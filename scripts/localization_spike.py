"""Localization spike: does base Qwen2.5-VL draw usable defect boxes? ($0, MPS)

Samples 4 test-split crops per spike class + the 8 realistic field photos,
runs the BASE model (adapter deliberately not loaded) with the grounding
prompt, renders box overlays, and writes a hand-rating HTML sheet.

Pass criteria (spec, fixed up front): >=70% of photos rated "useful"
(box tightly covers the defect) among photos whose primary class is correct.

Run:  DEFECTLENS_ADAPTER=/nonexistent .venv/bin/python scripts/localization_spike.py
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectlens.ingest import read_manifest  # noqa: E402
from defectlens.localization import (  # noqa: E402
    GROUNDING_PROMPT,
    input_size_from_grid,
    parse_boxes,
    rescale_box,
)

SPIKE_CLASSES = [
    "crack",
    "spalling",
    "corrosion_stain",
    "insulator_damage",
    "finish_detachment",
]
REALISTIC_DIR = Path("data/raw/realistic")
# Realistic field photos are full scenes - ground the class named in the file
# where obvious, else default to "crack" (rater judges usefulness regardless).
REALISTIC_CLASS = {
    "clifton_viaduct_cracks.jpg": "crack",
    "cochem_facade.jpg": "finish_detachment",
    "cracked_wall.jpg": "crack",
    "fema_louisiana_interior.jpg": "water_damage",
    "funchal_carbonation_rebar.jpg": "exposed_rebar",
    "kellokoski_wall_detail.jpg": "peeling_paint",
    "peeling_paint_closeup.jpg": "peeling_paint",
    "water_damaged_pub_wall.jpg": "water_damage",
}


def sample_manifest_rows(rows, n_per_class: int, seed: int):
    """Deterministic n-per-class sample, mirroring scripts/spot_check.py."""
    picked = []
    for label in SPIKE_CLASSES:
        group = sorted(
            (r for r in rows if r.unified_label == label),
            key=lambda r: r.image_path,
        )
        rng = random.Random(f"{seed}:{label}")
        picked.extend(rng.sample(group, min(n_per_class, len(group))))
    return picked


def ground_image(describer, processor, image, class_name: str):
    """One grounding generation on the BASE model. Returns (boxes, input_size).

    Mirrors Describer.chat() but uses a max_pixels-bounded processor (see main)
    and keeps the processor inputs so image_grid_thw (the smart-resized extent)
    is available for coordinate mapping. Generation still runs on the shared
    describer.model / describer.device.
    """
    import torch

    prompt = GROUNDING_PROMPT.format(name=class_name.replace("_", " "))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(describer.device)
    input_size = input_size_from_grid(inputs["image_grid_thw"][0])
    with torch.no_grad():
        output_ids = describer.model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )
    raw = processor.batch_decode(
        [row[inputs.input_ids.shape[1]:] for row in output_ids],
        skip_special_tokens=True,
    )[0]
    orig_size = (image.height, image.width)
    boxes = [
        {**b, "bbox_2d": rescale_box(b["bbox_2d"], input_size, orig_size)}
        for b in parse_boxes(raw)
    ]
    return boxes, input_size, raw


def render_overlay(image, boxes, dest: Path) -> None:
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox_2d"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=4)
        draw.text((x1 + 4, max(0, y1 - 16)), b["label"], fill=(255, 40, 40))
    canvas.save(dest)


def write_rating_sheet(entries: list[dict], out: Path) -> None:
    """Static HTML: overlay image + useful/loose/wrong radios + JSON export."""
    rows = []
    for i, e in enumerate(entries):
        rows.append(
            f'<div class="item"><h3>{i + 1}. {html.escape(e["name"])} '
            f'({html.escape(e["class"])}, {len(e["boxes"])} boxes)</h3>'
            f'<img src="{html.escape(e["overlay"])}" loading="lazy">'
            + "".join(
                f'<label><input type="radio" name="r{i}" value="{v}">{v}</label>'
                for v in ("useful", "loose", "wrong", "no_boxes")
            )
            + "</div>"
        )
    body = "\n".join(rows)
    out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Localization spike rating</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2rem auto}}
img{{max-width:100%;display:block;margin:.5rem 0}}
.item{{border-bottom:1px solid #ccc;padding:1rem 0}}label{{margin-right:1rem}}</style>
<h1>Localization spike - rate each overlay</h1>
<p>useful = box tightly covers the defect; loose = covers but sloppy/huge;
wrong = misses the defect; no_boxes = model returned none.
Pass = useful/(rated) >= 0.70.</p>
{body}
<button onclick="exportRatings()">Export ratings JSON</button><pre id="out"></pre>
<script>
function exportRatings() {{
  const n = {len(entries)}, r = [];
  for (let i = 0; i < n; i++) {{
    const c = document.querySelector(`input[name=r${{i}}]:checked`);
    r.push(c ? c.value : null);
  }}
  const useful = r.filter(v => v === "useful").length;
  const rated = r.filter(Boolean).length;
  document.getElementById("out").textContent = JSON.stringify(
    {{ratings: r, useful, rated, pass_ratio: rated ? useful / rated : 0}}, null, 2);
}}
</script>""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/localization_spike"))
    parser.add_argument("--n-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor

    from defectlens.serve.describer import QWEN_MODEL, Describer

    targets = [
        (Path(r.image_path), r.unified_label)
        for r in sample_manifest_rows(read_manifest(args.manifest), args.n_per_class, args.seed)
    ] + [(REALISTIC_DIR / name, cls) for name, cls in sorted(REALISTIC_CLASS.items())]

    describer = Describer()
    describer.load()
    assert not describer.adapter_loaded, (
        "Spike must run the BASE model - set DEFECTLENS_ADAPTER=/nonexistent"
    )

    # Bound grounding to the production pixel budget. Describer builds its own
    # processor bare (no max_pixels), so full-resolution field photos explode
    # the vision-token count and OOM MPS attention. The future product path
    # (deploy/sagemaker/inference.py) pins max_pixels=589824 (=768^2, the
    # train/eval budget MAX_PIXELS); measuring grounding at any other resolution
    # would score the model at a scale the product never runs. Match it here.
    grounding_processor = AutoProcessor.from_pretrained(QWEN_MODEL, max_pixels=589824)

    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    entries, raw_log = [], {}
    for path, cls in targets:
        image = Image.open(path).convert("RGB")
        boxes, input_size, raw = ground_image(describer, grounding_processor, image, cls)
        overlay_name = f"{path.stem}__{cls}.png"
        render_overlay(image, boxes, args.out / overlay_name)
        entries.append(
            {"name": path.name, "class": cls, "boxes": boxes, "overlay": overlay_name}
        )
        raw_log[path.name] = {"raw": raw, "input_size": input_size, "boxes": boxes}
        print(f"{path.name}: {len(boxes)} boxes")
        if describer.device == "mps":
            torch.mps.empty_cache()  # codified MPS fragmentation fix

    (args.out / "raw_outputs.json").write_text(json.dumps(raw_log, indent=2))
    write_rating_sheet(entries, args.out / "rating_sheet.html")
    print(f"Rate: open {args.out / 'rating_sheet.html'}")


if __name__ == "__main__":
    main()
