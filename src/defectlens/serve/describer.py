"""Natural-language condition description via Qwen2.5-VL-3B (spec §7).

Optional component: set DEFECTLENS_NO_VLM=1 to disable (API returns an empty
description and the UI hides the panel) — keeps serving usable on low-RAM
machines and in tests. fp16 on MPS needs ~7GB unified memory.
"""
from __future__ import annotations

import os

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt(top_classes: list[str]) -> str:
    """Deterministic instruction naming the top classes."""
    named = ", ".join(c.replace("_", " ") for c in top_classes[:3])
    return (
        "Describe the visible condition of this building surface in 2-3 "
        f"sentences for an inspection report. Focus on signs of: {named}. "
        "Be factual and specific about what is visible; do not speculate "
        "about causes you cannot see."
    )


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

    def describe(self, image, top_classes: list[str]) -> str:
        if vlm_disabled() or self.model is None or self.processor is None:
            return ""

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_prompt(top_classes)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=120, do_sample=False
            )
        input_len = inputs.input_ids.shape[1]
        return _decode_tail(self.processor, output_ids, input_len)
