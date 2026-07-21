"""Qwen2.5-VL-3B serving component: fine-tuned classification + description.

One model, two modes (spec §7 + Phase 3 landing):
- rank_classes(): LoRA adapter ACTIVE — the fine-tuned classifier over the
  12-class v2 taxonomy (v1 adapter: macro top-1 0.851 on the frozen 9-class split).
- describe(): adapter DISABLED per call — the base model writes the
  condition description. The adapter was trained only on terse class
  answers and measurably degrades open-ended narration, so description
  quality comes from the base weights.

Optional component: set DEFECTLENS_NO_VLM=1 to disable (API returns an empty
description, classification falls back to the CLIP-fused ranking) — keeps
serving usable on low-RAM machines and in tests. ~7GB unified memory on MPS.
DEFECTLENS_ADAPTER overrides the adapter dir (default models/qwen25vl-lora-v1;
missing dir = base model only, classification stays CLIP-fused).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER = "models/qwen25vl-lora-v1"


def build_prompt(top_classes: list[str], audio_band: str | None = None) -> str:
    """Deterministic instruction naming the top classes.

    When an equipment-audio clip accompanies the photo, its calibrated band is
    named in one extra sentence so the narration covers both signals (Phase 5.3).
    """
    named = ", ".join(c.replace("_", " ") for c in top_classes[:3])
    prompt = (
        "Describe the visible condition of this building surface in 2-3 "
        f"sentences for an inspection report. Focus on signs of: {named}. "
        "Be factual and specific about what is visible; do not speculate "
        "about causes you cannot see."
    )
    if audio_band:
        prompt += (
            " An accompanying equipment-audio recording was assessed as "
            f"'{audio_band.replace('_', ' ')}'; note this alongside the visual findings."
        )
    return prompt


def vlm_disabled() -> bool:
    return os.environ.get("DEFECTLENS_NO_VLM", "") == "1"


def _decode_tail(processor, output_ids, input_len: int) -> str:
    """Trim off the prompt tokens and decode only the newly generated tail.

    `output_ids` is the full [batch, seq] sequence returned by
    model.generate() (prompt + continuation); `input_len` is the prompt
    token count (inputs.input_ids.shape[1]). Iterating per-row and slicing
    keeps this agnostic to whether output_ids is a torch tensor or a plain
    list of lists (as used in tests).
    """
    trimmed = [row[input_len:] for row in output_ids]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
    return decoded[0].strip()


class Describer:
    """Loads Qwen2.5-VL-3B once, then generates a condition description per image."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device: str | None = None
        self.adapter_loaded = False

    def load(self) -> None:
        if vlm_disabled():
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from defectlens.eval.clip_zeroshot import pick_device

        self.device = pick_device()
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL, dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL)

        adapter_dir = Path(os.environ.get("DEFECTLENS_ADAPTER", DEFAULT_ADAPTER))
        if adapter_dir.is_dir():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
            self.model = self.model.to(self.device).eval()
            self.adapter_loaded = True

    def rank_classes(self, image, note: str | None = None) -> list[tuple[str, float]]:
        """Fine-tuned classification over UNIFIED_CLASSES: (label, probability) descending.

        Same length-normalized answer log-likelihood ranking the Phase 3 eval
        measured 0.851 macro top-1 with; log-liks are softmaxed so the API
        reports probabilities. Returns [] when the adapter isn't loaded —
        callers fall back to the CLIP-fused ranking.

        Args:
            note: Optional inspector free-text to include in the classification prompt.
        """
        if not self.adapter_loaded:
            return []

        from defectlens.eval.vlm_topk import score_answers

        # score_answers returns {label: loglik} (labels, not answer texts)
        loglik = score_answers(self.model, self.processor, image, self.device, note=note)
        z = max(loglik.values())
        weights = {label: math.exp(v - z) for label, v in loglik.items()}
        total = sum(weights.values())
        return sorted(
            ((label, w / total) for label, w in weights.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )

    def describe(
        self, image, top_classes: list[str], audio_band: str | None = None
    ) -> str:
        if vlm_disabled() or self.model is None or self.processor is None:
            return ""

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_prompt(top_classes, audio_band)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        # Adapter OFF for generation: the classification fine-tune measurably
        # degrades free-text description quality; base weights write it.
        from contextlib import nullcontext

        ctx = self.model.disable_adapter() if self.adapter_loaded else nullcontext()
        with ctx, torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=120, do_sample=False
            )
        input_len = inputs.input_ids.shape[1]
        return _decode_tail(self.processor, output_ids, input_len)

    def chat(
        self, prompt: str, image=None, max_new_tokens: int = 400, images: list | None = None
    ) -> str:
        """Generic adapter-OFF generation for the agent workflow.

        Unlike describe(), the caller owns the prompt; image/images are
        optional so the same model does text-only synthesis steps. images
        carries a multi-photo walkthrough (one content entry per image, in
        order) - Qwen2.5-VL accepts multiple images in one message.
        """
        if vlm_disabled() or self.model is None or self.processor is None:
            return ""

        import torch

        if images is not None and image is not None:
            raise ValueError("pass either image or images, not both")
        imgs = list(images) if images is not None else ([image] if image is not None else [])

        content: list[dict] = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        kwargs = {"text": [text], "return_tensors": "pt"}
        if imgs:
            kwargs["images"] = imgs
        inputs = self.processor(**kwargs).to(self.device)
        from contextlib import nullcontext

        ctx = self.model.disable_adapter() if self.adapter_loaded else nullcontext()
        with ctx, torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        input_len = inputs.input_ids.shape[1]
        return _decode_tail(self.processor, output_ids, input_len)
