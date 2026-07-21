"""SageMaker inference handler for the fine-tuned Qwen2.5-VL defect classifier.

Runs on the HuggingFace PyTorch Inference DLC (see infra/stacks/gpu_stack.py for
the exact image). The model.tar.gz that ships this file also carries ONLY the
~120MB LoRA adapter (models/qwen25vl-lora-v1); the ~8GB Qwen2.5-VL-3B base is NOT
baked in — it downloads from the HuggingFace hub inside model_fn at endpoint
start, so the first cold start is ~5-8 min (base download + adapter merge).

The classification logic (score_answers + softmax ranking) is a self-contained
copy of defectlens.eval.vlm_topk.score_answers and defectlens.serve.describer.
Describer.rank_classes, because the `defectlens` package is NOT installed in the
DLC — the tarball is just this handler + its deps + the adapter. The inlined
constants (HUMANIZED, QUESTION, UNIFIED_CLASSES, MAX_PIXELS) MUST stay in lockstep
with defectlens.train.qlora / defectlens.eval.vlm_topk: the answer set, prompt
text, and pixel budget are all part of the exact training-time prompt the 0.851
macro top-1 was measured with. A drift here silently degrades accuracy.

Module-import contract (mirrors qlora/vlm_topk): only stdlib loads at import
time. torch/transformers/peft/PIL are imported lazily inside the functions that
need them, so the pure payload/response helpers are unit-testable without a GPU,
a model, or those heavy deps (see tests/test_package_sagemaker.py).

SageMaker serving contract: the HF inference toolkit imports this module and
calls model_fn(model_dir) once, then input_fn -> predict_fn -> output_fn per
request. Request JSON: {"image_b64": "<base64 image>", "note": "<optional>"};
response JSON: {"classes": [["crack", 0.87], ["spalling", 0.05], ...]} (softmaxed
probabilities, descending), matching serve.api's {label, score} ordering.
"""
from __future__ import annotations

import base64
import json
import math
import re

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Vision-token pixel budget — MUST equal the value used at training and eval
# (defectlens.train.qlora / eval.vlm_topk default 589824 = 768^2). Image-token
# count is pixel-budget-dependent, so a different budget shifts the answer
# log-likelihoods and degrades the measured accuracy.
MAX_PIXELS = 589824

MAX_NOTE_CHARS = 500

# label -> humanized answer text (the eval answer set — kept identical to
# defectlens.train.qlora.HUMANIZED so the log-likelihood ranking is well-defined).
HUMANIZED = {
    "crack": "crack",
    "spalling": "spalling",
    "efflorescence": "efflorescence",
    "exposed_rebar": "exposed rebar",
    "corrosion_stain": "corrosion stain",
    "mold_algae": "mold or algae",
    "water_damage": "water damage",
    "peeling_paint": "peeling paint",
    "no_defect": "no defect",
    "finish_detachment": "finish detachment",
    "bulge_deformation": "bulging deformation",
    "insulator_damage": "insulator damage",
}
UNIFIED_CLASSES = list(HUMANIZED)

QUESTION = (
    "What building defect is shown in this image? Answer with one of: "
    "crack, spalling, efflorescence, exposed rebar, corrosion stain, "
    "mold or algae, water damage, peeling paint, no defect, "
    "finish detachment, bulging deformation, insulator damage."
)

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no torch/transformers/PIL involved)
# ---------------------------------------------------------------------------


def sanitize_note(note: str | None) -> str | None:
    """Strip chat-template control markers and cap length, mirroring serve.api.

    A None/blank note returns None, which build_messages turns into the EXACT
    training-time prompt (the serve layer's empty-note contract).
    """
    if not note or not note.strip():
        return None
    return re.sub(r"<\|[^>]*\|>", " ", note.strip())[:MAX_NOTE_CHARS]


def parse_input(data: dict) -> tuple[str, str | None]:
    """Validate the request payload into (image_b64, note). Pure + testable.

    Raises ValueError with a clear message on a missing/blank image_b64 so a
    malformed request fails fast with a 4xx-shaped error rather than deep inside
    the model call.
    """
    if not isinstance(data, dict):
        raise ValueError(f"request body must be a JSON object, got {type(data).__name__}")
    image_b64 = data.get("image_b64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ValueError("request is missing a non-empty 'image_b64' field")
    return image_b64, sanitize_note(data.get("note"))


def softmax_rank(loglik: dict[str, float]) -> list[list]:
    """Softmax the per-label log-likelihoods into (label, prob) pairs, descending.

    The softmax normalization mirrors serve.describer.Describer.rank_classes; the
    deterministic label-ascending tie-break mirrors eval.vlm_topk.rank_answers
    (describer.rank_classes has no explicit tie-break). Returns JSON-ready lists
    (not tuples) so the response serializes to [["crack", 0.87], ...].
    """
    if not loglik:
        return []
    z = max(loglik.values())
    weights = {label: math.exp(v - z) for label, v in loglik.items()}
    total = sum(weights.values())
    ranked = sorted(
        ((label, w / total) for label, w in weights.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [[label, prob] for label, prob in ranked]


def build_response(ranked: list[list]) -> dict:
    """Wrap the ranked (label, prob) pairs in the response envelope."""
    return {"classes": ranked}


def build_messages(image, label: str, note: str | None = None) -> list[dict]:
    """Qwen chat-format messages for one (image, label) pair — copy of
    defectlens.train.qlora.build_messages. `note` (already sanitized) is prefixed
    before the question; a None note yields the exact training-time prompt.
    """
    question = QUESTION
    if note and note.strip():
        clean = re.sub(r"<\|[^>]*\|>", " ", note.strip())[:MAX_NOTE_CHARS]
        question = f"Inspector note: {clean}\n{QUESTION}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": HUMANIZED[label]},
    ]


def decode_image(image_b64: str):
    """Decode a base64 image string into an RGB PIL.Image (lazy PIL import)."""
    from io import BytesIO

    from PIL import Image

    raw = base64.b64decode(image_b64)
    image = Image.open(BytesIO(raw))
    image.load()  # force full decode; catches truncated/corrupt payloads
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Model-facing (needs real torch/transformers/peft + a real model/image; not
# unit-tested — exercised only by the live endpoint's first-request smoke test).
# ---------------------------------------------------------------------------


def score_answers(model, processor, image, device: str, note: str | None = None) -> dict[str, float]:
    """Length-normalized teacher-forced answer log-likelihood per class.

    Exact copy of defectlens.eval.vlm_topk.score_answers: for each of the 9
    classes, build the training (image, answer) chat, measure the prompt-token
    boundary by re-encoding the prompt-only chat with the same image (image-token
    expansion is image-size-dependent), and sum the log-softmax over the answer
    span, divided by its token count.
    """
    import torch
    import torch.nn.functional as F

    scores: dict[str, float] = {}
    for label in UNIFIED_CLASSES:
        messages = build_messages(image, label, note=note)
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        encoded = processor(text=[full_text], images=[image], return_tensors="pt").to(device)
        prompt_len = (
            processor(text=[prompt_text], images=[image], return_tensors="pt")
            .input_ids.shape[1]
        )

        input_ids = encoded["input_ids"][0]
        seq_len = input_ids.shape[0]
        n_answer_tokens = seq_len - prompt_len
        if n_answer_tokens <= 0:
            scores[label] = float("-inf")
            continue

        with torch.no_grad():
            logits = model(**encoded).logits[0]

        answer_token_ids = input_ids[prompt_len:]
        pred_logits = logits[prompt_len - 1 : seq_len - 1].float()
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)
        scores[label] = (token_log_probs.sum() / n_answer_tokens).item()

    return scores


def model_fn(model_dir: str):
    """Load the base Qwen2.5-VL-3B (from the HF hub) + the LoRA adapter (from the
    tarball's model_dir root) once per endpoint instance.

    Returns the bundle predict_fn needs. bf16 on the GPU (ml.g5.xlarge / A10G),
    fp32 only on a CPU fallback. AutoProcessor is built with max_pixels=MAX_PIXELS
    to match training/eval — NOT the processor default.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL, dtype=dtype)
    # The adapter files (adapter_config.json + adapter_model.safetensors) sit at
    # the model.tar.gz root, which SageMaker extracts to model_dir.
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(QWEN_MODEL, max_pixels=MAX_PIXELS)
    return {"model": model, "processor": processor, "device": device}


def input_fn(request_body, content_type: str = "application/json") -> dict:
    """Deserialize the request body to a dict. Only JSON is supported.

    The content type may carry parameters (e.g. "application/json; charset=utf-8"),
    so compare only the media type, not the raw header.
    """
    media_type = (content_type or "").split(";")[0].strip()
    if media_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(data: dict, bundle: dict) -> dict:
    """Rank the 9 defect classes for the request image (+ optional note)."""
    image_b64, note = parse_input(data)
    image = decode_image(image_b64)
    loglik = score_answers(
        bundle["model"], bundle["processor"], image, bundle["device"], note=note
    )
    return build_response(softmax_rank(loglik))


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize the prediction to a JSON body string (allow_nan=False — NaN is
    invalid JSON).

    The HuggingFace inference toolkit expects output_fn to return ONLY the body;
    it sets the response content type separately via
    context.set_response_content_type, then wraps the return value as
    ``[response]``. Returning a (body, content_type) tuple would serialize the
    TUPLE into the async S3 output (e.g. ``["{...}", "application/json"]``) and
    break vlm_gateway.parse_output.
    """
    return json.dumps(prediction, allow_nan=False)
