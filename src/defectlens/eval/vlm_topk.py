"""Top-k defect recognition eval for Qwen2.5-VL via answer log-likelihood ranking.

For each test image, score the sequence log-likelihood of each of the 9
humanized class answers under the model (teacher-forced), rank descending,
compute macro top-1/top-3 on the frozen test split (spec §5).

Module-import contract: only pure-Python/lightweight deps load at import
time (mirrors defectlens.train.qlora). torch/transformers/peft load lazily
inside the functions that need them, so `import defectlens.eval.vlm_topk`
stays cheap (see tests/test_vlm_topk.py's import sanity check).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from defectlens.ingest import read_manifest
from defectlens.metrics import confusion_matrix, macro_topk_accuracy, per_class_topk_accuracy
from defectlens.serve.describer import QWEN_MODEL
from defectlens.taxonomy import UNIFIED_CLASSES
from defectlens.train.qlora import HUMANIZED, build_messages, subset_rows

# label -> humanized answer text, inverted; ANSWER_TO_LABEL[answer] -> label.
# Asserting bijectivity here (not just in tests) means a future HUMANIZED edit
# that introduces a duplicate answer string fails loudly at import time.
ANSWER_TO_LABEL = {answer: label for label, answer in HUMANIZED.items()}
assert len(ANSWER_TO_LABEL) == len(HUMANIZED), (
    "HUMANIZED answers must be unique (bijective label<->answer mapping) "
    "for log-likelihood ranking to be well-defined"
)

# ---------------------------------------------------------------------------
# Pure functions (TDD-covered; no torch/transformers/PIL involved)
# ---------------------------------------------------------------------------


def rank_answers(loglik: dict[str, float]) -> list[str]:
    """Labels sorted by their (length-normalized) answer log-likelihood, desc.

    Ties broken deterministically by label name (ascending) so ranking is
    reproducible regardless of dict insertion order or float equality.
    """
    return [label for label, _ in sorted(loglik.items(), key=lambda kv: (-kv[1], kv[0]))]


def _nan_to_none(value: float) -> float | None:
    """NaN (absent class) -> null in JSON; json.dumps NaN is invalid per RFC 8259."""
    return None if math.isnan(value) else value


def results_payload(
    y_true: list[str], ranked: list[list[str]], k_values: tuple[int, ...] = (1, 3)
) -> dict:
    """Macro/per-class top-k + confusion matrix, mirroring clip_zeroshot's JSON shape.

    `ranked` is one label-ranking (best first) per image, e.g. from
    rank_answers(score_answers(...)). Adds "model_kind": "vlm_loglik" so the
    output JSON is distinguishable from the CLIP baseline's.
    """
    top1 = [r[0] for r in ranked]
    payload: dict = {
        "model_kind": "vlm_loglik",
        "n_images": len(y_true),
    }
    for k in k_values:
        per = per_class_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=k)
        payload[f"macro_top{k}"] = _nan_to_none(
            macro_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=k)
        )
        payload[f"per_class_top{k}"] = {c: _nan_to_none(v) for c, v in per.items()}
    payload["confusion_matrix"] = confusion_matrix(y_true, top1, UNIFIED_CLASSES)
    payload["classes"] = UNIFIED_CLASSES
    return payload


# ---------------------------------------------------------------------------
# Model-facing (thin; exercised by the controller's local/GPU runs, not
# unit-tested — needs real torch/transformers + a real model/image).
# ---------------------------------------------------------------------------


def pick_device() -> str:
    """Duplicated from clip_zeroshot.pick_device (not imported) to keep this

    module's top-level import cheap — clip_zeroshot imports torch/numpy/PIL
    at module scope, which would defeat the import-sanity contract above.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def score_answers(model, processor, image, device: str) -> dict[str, float]:
    """Length-normalized teacher-forced log-likelihood of each of the 9 answers.

    For each label, builds the exact same (image, label) chat used in
    training (qlora.build_messages) and measures the prompt-token length the
    same way qlora.build_collate_fn does: re-encoding the prompt-only chat
    (add_generation_prompt=True) with the same image, since Qwen2.5-VL's
    image-token expansion is image-size-dependent. Tokens from that boundary
    to the end of the full sequence are the "answer" span (assistant text +
    any closing/eos tokens the chat template adds) — summed log-softmax over
    that span, divided by its token count so multi-token answers (e.g.
    "corrosion stain") aren't penalized relative to single-token ones
    (e.g. "crack").
    """
    import torch
    import torch.nn.functional as F

    scores: dict[str, float] = {}
    for label in UNIFIED_CLASSES:
        messages = build_messages(image, label)
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
            # Shouldn't happen (mirrors the assertion in qlora's collator) but
            # degrade gracefully rather than raising mid-eval.
            scores[label] = float("-inf")
            continue

        with torch.no_grad():
            logits = model(**encoded).logits[0]  # [seq_len, vocab]

        answer_token_ids = input_ids[prompt_len:]
        # logits[i] predicts token i+1, so the predictions for the answer
        # span start one position before it.
        pred_logits = logits[prompt_len - 1 : seq_len - 1].float()
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)
        scores[label] = (token_log_probs.sum() / n_answer_tokens).item()

    return scores


def _load_model_and_processor(adapter: Path | None, device: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL, dtype=dtype)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    return model, processor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument(
        "--adapter", type=Path, default=None, help="LoRA adapter dir (omit for base model)"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="balanced subset size (local validation)"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-name", type=str, default="vlm_topk.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def main(argv: list[str] | None = None) -> None:
    from PIL import Image
    from tqdm import tqdm

    args = build_arg_parser().parse_args(argv)

    rows = read_manifest(args.test_manifest)
    if args.subset:
        rows = subset_rows(rows, args.subset, seed=args.seed)
    adapter_desc = f"adapter={args.adapter}" if args.adapter else "base model"
    print(f"Evaluating on {len(rows)} rows ({adapter_desc})")

    device = pick_device()
    print(f"Device: {device}; model: {QWEN_MODEL}")
    model, processor = _load_model_and_processor(args.adapter, device)

    y_true = [r.unified_label for r in rows]
    ranked: list[list[str]] = []
    for row in tqdm(rows, desc="images"):
        image = Image.open(row.image_path).convert("RGB")
        loglik = score_answers(model, processor, image, device)
        ranked.append(rank_answers(loglik))

    payload = results_payload(y_true, ranked, k_values=(1, 3))
    payload.update(
        {
            "model": QWEN_MODEL,
            "adapter": str(args.adapter) if args.adapter else None,
            "manifest": str(args.test_manifest),
        }
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)
    out_json = args.out_dir / args.out_name
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"macro top-1: {_fmt(payload['macro_top1'])}  macro top-3: {_fmt(payload['macro_top3'])}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
