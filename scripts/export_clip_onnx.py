"""Export the CLIP image + text encoders to ONNX (fp32 + int8) - Phase 2.

Offline / one-time (needs torch; serving does NOT). Produces, under --out-dir:
  clip_image.onnx[.data]      clip_image_int8.onnx      (image encoder)
  clip_text.onnx[.data]       clip_text_int8.onnx       (text encoder)
  tokenizer.json + friends    (loaded torch-free via the `tokenizers` lib at serving)

The serving-side OnnxClipEncoder runs these with onnxruntime and NO torch/
transformers import (the Phase 2 cold-start win). The int8 variants add a ~2-4x
warm CPU speedup; the frozen-split accuracy gate (scripts/.. / eval) decides int8
vs the fp32 fallback (fp32 is numerically identical to torch CLIP - spike: cosine
1.0).

Uses the LEGACY TorchScript exporter (dynamo=False): the dynamo exporter's graph
breaks onnxruntime's dynamic-quantizer shape inference (verified: 1024-vs-768
InferenceError), while the legacy graph quantizes cleanly.

Usage:
  python scripts/export_clip_onnx.py                      # both encoders, fp32+int8
  python scripts/export_clip_onnx.py --no-int8            # fp32 only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import CLIPModel, CLIPTokenizerFast

MODEL = "openai/clip-vit-large-patch14"
OPSET = 14  # legacy exporter target; CLIP ops are well within this


class _ImageEncoder(torch.nn.Module):
    def __init__(self, clip: CLIPModel) -> None:
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values):
        # transformers 5.x: get_image_features -> BaseModelOutputWithPooling whose
        # pooler_output IS the projected features (see clip_zeroshot._features).
        return self.clip.get_image_features(pixel_values=pixel_values).pooler_output


class _TextEncoder(torch.nn.Module):
    def __init__(self, clip: CLIPModel) -> None:
        super().__init__()
        self.clip = clip

    def forward(self, input_ids, attention_mask):
        return self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output


def _export(module: torch.nn.Module, args_tuple, path: Path, input_names, output_name):
    dynamic = {n: {0: "batch"} for n in input_names}
    dynamic[output_name] = {0: "batch"}
    torch.onnx.export(
        module,
        args_tuple,
        str(path),
        input_names=input_names,
        output_names=[output_name],
        dynamic_axes=dynamic,
        opset_version=OPSET,
        dynamo=False,  # legacy graph so int8 dynamic-quant shape-infer works
    )
    print(f"  exported {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("models/cloud_artifacts/onnx")
    )
    ap.add_argument("--no-int8", action="store_true", help="skip int8 quantization")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} (torch, offline) ...")
    clip = CLIPModel.from_pretrained(args.model).eval()

    # dummy inputs (batch=1); dynamic batch axis is exported.
    px = torch.zeros(1, 3, 224, 224)
    ids = torch.ones(1, 77, dtype=torch.long)
    mask = torch.ones(1, 77, dtype=torch.long)

    img_fp32 = args.out_dir / "clip_image.onnx"
    txt_fp32 = args.out_dir / "clip_text.onnx"
    print("Exporting encoders (legacy exporter) ...")
    _export(_ImageEncoder(clip).eval(), (px,), img_fp32, ["pixel_values"], "image_embeds")
    _export(
        _TextEncoder(clip).eval(),
        (ids, mask),
        txt_fp32,
        ["input_ids", "attention_mask"],
        "text_embeds",
    )

    if not args.no_int8:
        print("Quantizing to int8 (dynamic) ...")
        for src in (img_fp32, txt_fp32):
            dst = src.with_name(src.stem + "_int8.onnx")
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            print(f"  int8 {dst.name}")

    # Save the tokenizer so serving can load it torch-free via the `tokenizers`
    # lib (transformers' CLIPTokenizerFast import pulls torch - avoided at serving).
    CLIPTokenizerFast.from_pretrained(args.model).save_pretrained(str(args.out_dir))
    print(f"Saved tokenizer + {['fp32', 'fp32+int8'][not args.no_int8]} encoders to {args.out_dir}")


if __name__ == "__main__":
    main()
