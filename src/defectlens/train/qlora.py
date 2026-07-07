"""QLoRA/LoRA fine-tune of Qwen2.5-VL-3B on the frozen train split (spec §5).

Local validation: --quant none --subset 96 --max-steps 20 on MPS/CPU.
GPU run:          --quant 4bit on CUDA (bitsandbytes required).

Module-import contract: only pure-Python/lightweight deps (ingest, taxonomy,
describer's QWEN_MODEL constant) load at import time. torch/transformers/peft
/bitsandbytes are imported lazily inside the functions that need them, so
`import defectlens.train.qlora` stays cheap (see tests/test_train.py's import
sanity check).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from defectlens.ingest import ManifestRow, read_manifest
from defectlens.serve.describer import QWEN_MODEL
from defectlens.taxonomy import UNIFIED_CLASSES

HUMANIZED = {  # label -> answer text (also the eval answer set — keep in ONE place)
    "crack": "crack",
    "spalling": "spalling",
    "efflorescence": "efflorescence",
    "exposed_rebar": "exposed rebar",
    "corrosion_stain": "corrosion stain",
    "mold_algae": "mold or algae",
    "water_damage": "water damage",
    "peeling_paint": "peeling paint",
    "no_defect": "no defect",
}
QUESTION = (
    "What building defect is shown in this image? Answer with one of: "
    "crack, spalling, efflorescence, exposed rebar, corrosion stain, "
    "mold or algae, water damage, peeling paint, no defect."
)

assert set(HUMANIZED) == set(UNIFIED_CLASSES), "HUMANIZED must cover all unified classes"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ---------------------------------------------------------------------------
# Pure functions (TDD-covered; no torch/transformers/PIL involved)
# ---------------------------------------------------------------------------


def class_weights(labels: list[str], cap: float = 20.0) -> dict[str, float]:
    """Inverse-frequency weight per label, normalized so the majority class is 1.0.

    weight[label] = min(max_count / count[label], cap) — deterministic given
    the label multiset; rare classes are oversampled but capped at `cap`x.
    """
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {label: min(max_count / count, cap) for label, count in counts.items()}


def sample_weights(rows: list[ManifestRow]) -> list[float]:
    """Per-row sampling weight, looked up from class_weights over the rows' labels."""
    weights = class_weights([r.unified_label for r in rows])
    return [weights[r.unified_label] for r in rows]


def build_messages(image_path: str, label: str) -> list[dict]:
    """Qwen chat-format messages for one (image, label) training example.

    `image_path` is embedded as-is into the "image" content field: pass a
    path string (as these tests do) or an already-opened PIL.Image (as the
    training Dataset does) — the processor accepts either.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": QUESTION},
            ],
        },
        {"role": "assistant", "content": HUMANIZED[label]},
    ]


def subset_rows(rows: list[ManifestRow], n: int, seed: int = 42) -> list[ManifestRow]:
    """Deterministic per-class-balanced subset of `n` rows, round-robin over classes.

    Each class gets its own RNG seeded from (seed, label) — matching
    ingest.apply_caps's per-group RNG convention — so a class's shuffle order
    (and hence its first picks) is stable regardless of what other classes
    are present in `rows`. Classes are visited in sorted-label order each
    round; stops as soon as `n` rows are collected or every class is
    exhausted (never raises if there are fewer than `n` rows total).
    """
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[r.unified_label].append(r)

    labels = sorted(grouped)
    queues: dict[str, list[ManifestRow]] = {}
    for label in labels:
        class_rows = sorted(grouped[label], key=lambda r: r.image_path)
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(class_rows)
        queues[label] = class_rows

    out: list[ManifestRow] = []
    round_idx = 0
    while len(out) < n:
        picked_this_round = False
        for label in labels:
            if len(out) >= n:
                break
            queue = queues[label]
            if round_idx < len(queue):
                out.append(queue[round_idx])
                picked_this_round = True
        if not picked_this_round:
            break
        round_idx += 1
    return out


# ---------------------------------------------------------------------------
# Training assembly — thin, exercised by the controller's local/GPU runs, not
# unit-tested (needs real torch/transformers/peft + real images/model).
# ---------------------------------------------------------------------------


def _manifest_dataset(rows: list[ManifestRow]):
    """Torch Dataset over ManifestRows; __getitem__ opens the PIL image."""
    from PIL import Image
    from torch.utils.data import Dataset

    class ManifestDataset(Dataset):
        def __init__(self, rows: list[ManifestRow]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            row = self.rows[idx]
            image = Image.open(row.image_path).convert("RGB")
            return build_messages(image, row.unified_label), row.unified_label

    return ManifestDataset(rows)


def build_collate_fn(processor):
    """Collator: chat-template + processor encode, with assistant-only label masking.

    For each sample, the prompt-only chat text (up to and including the
    generation prompt) is encoded separately with the same image to get the
    exact prompt token length (image-token expansion is image-size-dependent,
    so this must be measured per-sample rather than assumed from text alone).
    Everything up to that length, plus padding, is masked to -100; the
    remaining tokens (the assistant answer + eos) are the training targets.
    """

    def _prompt_only(messages: list[dict]) -> list[dict]:
        return messages[:-1]

    def collate(batch):
        messages_list, _labels = zip(*batch)
        images = [m[0]["content"][0]["image"] for m in messages_list]

        full_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        prompt_texts = [
            processor.apply_chat_template(
                _prompt_only(m), tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]

        encoded = processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )
        prompt_lens = [
            processor(text=[pt], images=[img], return_tensors="pt").input_ids.shape[1]
            for pt, img in zip(prompt_texts, images)
        ]

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
            assert (labels[i] != -100).any(), (
                f"sample {i}: no unmasked answer tokens (prompt_len={plen} "
                ">= sequence length) — check chat template/image sizing"
            )
        encoded["labels"] = labels
        return encoded

    return collate


def load_base_model(quant: str):
    """Load Qwen2.5-VL-3B: 4-bit NF4 on CUDA (bitsandbytes) or fp32 elsewhere.

    --quant 4bit requires CUDA; guarded with a clear SystemExit rather than
    letting bitsandbytes fail obscurely on MPS/CPU.
    """
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    if quant == "4bit":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--quant 4bit requires a CUDA GPU (bitsandbytes) — use "
                "--quant none for local MPS/CPU validation"
            )
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "bitsandbytes not installed — pip install bitsandbytes on the "
                "GPU box (the DLAMI image already has it)"
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL, quantization_config=bnb_config, device_map="auto"
        )
        from peft import prepare_model_for_kbit_training

        # Standard QLoRA prep: casts norms to fp32, enables input-grads on the
        # (frozen, quantized) base so gradients actually flow into the LoRA
        # adapters — required for a quantized base, not exercised by the
        # --quant none local path.
        return prepare_model_for_kbit_training(model)
    if quant == "none":
        # fp32 (not fp16): fp16 optimizer state is unstable on MPS, and this
        # path is only ever a 20-step local correctness check.
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL, dtype=torch.float32
        )
    raise SystemExit(f"--quant must be '4bit' or 'none', got {quant!r}")


def apply_lora(model, r: int, alpha: int, dropout: float = 0.05):
    """Wrap `model` with a LoRA adapter over the LM's attention+MLP projections.

    Vision tower frozen (spec §5 "freeze vision tower v1"): Qwen2.5-VL's
    vision-block MLP submodules are also literally named gate_proj/up_proj/
    down_proj, so target_modules alone would silently attach adapters there
    too (verified against the loaded model's named_modules: matches live
    under both `model.language_model.layers.*` and `model.visual.blocks.*`).
    exclude_modules excludes the latter.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        exclude_modules=r".*visual.*",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _weighted_trainer_cls():
    """Return a Trainer subclass sampling batches via a seeded WeightedRandomSampler."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, train_sample_weights, sampler_seed, **kwargs):
            super().__init__(*args, **kwargs)
            self._train_sample_weights = train_sample_weights
            self._sampler_seed = sampler_seed

        def _get_train_sampler(self, train_dataset=None):
            import torch
            from torch.utils.data import WeightedRandomSampler

            weights = torch.tensor(self._train_sample_weights, dtype=torch.double)
            generator = torch.Generator()
            generator.manual_seed(self._sampler_seed)
            return WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True, generator=generator
            )

    return WeightedTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument(
        "--subset", type=int, default=None, help="balanced subset size (local validation)"
    )
    parser.add_argument(
        "--quant", choices=["4bit", "none"], required=True, help="4bit=CUDA only; none=MPS/CPU"
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qlora"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pixels", type=int, default=589824,
        help="processor image pixel budget (768^2 default; use 262144=512^2 on "
        "low-memory local runs). Vision tokens dominate sequence length — this "
        "is the main memory/cost knob. MUST match eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoProcessor, TrainingArguments

    args = build_arg_parser().parse_args(argv)

    rows = read_manifest(args.train_manifest)
    if args.subset:
        rows = subset_rows(rows, args.subset, seed=args.seed)
    print(f"Training on {len(rows)} rows (quant={args.quant})")

    processor = AutoProcessor.from_pretrained(QWEN_MODEL, max_pixels=args.max_pixels)
    model = load_base_model(args.quant)
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    dataset = _manifest_dataset(rows)
    collate_fn = build_collate_fn(processor)
    weights = sample_weights(rows)

    on_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=5,
        report_to=[],
        seed=args.seed,
        bf16=on_cuda,
        dataloader_num_workers=0,  # MPS safety
        remove_unused_columns=False,
    )

    trainer_cls = _weighted_trainer_cls()
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=processor,
        train_sample_weights=weights,
        sampler_seed=args.seed,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = Path(args.output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    # The last log_history entry is often the train-summary dict (train_runtime,
    # etc.) with no "loss" key — walk backwards for the last real step log.
    final_loss = next(
        (entry["loss"] for entry in reversed(trainer.state.log_history) if "loss" in entry),
        None,
    )
    print(f"Final loss: {final_loss}")
    print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
