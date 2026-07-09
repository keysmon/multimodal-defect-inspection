#!/usr/bin/env python3
"""Build the frontend example-gallery assets from the CC BY source datasets.

Copies six visually-verified inspection crops (4 SDNET2018 + 2 METU) into
``frontend/public/gallery/``, downscaled to a web-friendly JPEG, and writes an
``ATTRIBUTION.md`` naming both datasets, their authors, and licenses (both are
CC BY 4.0). The source images live under the gitignored ``data/raw/``; the six
filenames below were hand-picked for clear crack / no-crack signal.

Run from the repo root::

    python scripts/build_gallery_assets.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "frontend" / "public" / "gallery"

MAX_EDGE = 512  # thumbnail() only downscales; the sources are already <=256px
JPEG_QUALITY = 85
SIZE_CAP_BYTES = 300 * 1024

# (source path relative to data/raw, output filename)
EXAMPLES = [
    ("sdnet2018/cracked/W__CW__7069-101.jpg", "sdnet-wall-crack.jpg"),
    ("sdnet2018/cracked/P__CP__001-100.jpg", "sdnet-pavement-crack.jpg"),
    ("sdnet2018/non_cracked/W__UW__7069-100.jpg", "sdnet-wall-clean.jpg"),
    ("sdnet2018/non_cracked/D__UD__7001-100.jpg", "sdnet-deck-clean.jpg"),
    ("ood_crack/Positive/00001.jpg", "metu-crack.jpg"),
    ("ood_crack/Negative/00001.jpg", "metu-clean.jpg"),
]

ATTRIBUTION = """# Gallery image attribution

The example images in this folder ship with the DefectLens frontend so the demo
works without an upload. Each was downscaled to a web-friendly JPEG; no other
modification was made. Both source datasets are licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

## SDNET2018

- Files: `sdnet-wall-crack.jpg`, `sdnet-pavement-crack.jpg`,
  `sdnet-wall-clean.jpg`, `sdnet-deck-clean.jpg`
- Authors: Maguire, M., Dorafshan, S., & Thomas, R. J. (Utah State University, 2018)
- Title: "SDNET2018: A concrete crack image dataset for machine learning applications"
- Source: https://digitalcommons.usu.edu/all_datasets/48/
- License: CC BY 4.0

## METU campus crack dataset

- Files: `metu-crack.jpg`, `metu-clean.jpg`
- Authors: Özgenel, Ç. F., & Gönenç Sorguç, A. (Middle East Technical University)
- Title: "Concrete Crack Images for Classification" (Mendeley Data, V2,
  doi:10.17632/5y9wdsg2zt.2); collected on the METU campus and introduced in
  Özgenel & Gönenç Sorguç (2018), "Performance Comparison of Pretrained
  Convolutional Neural Networks on Crack Detection in Buildings", ISARC 2018.
- Source: https://data.mendeley.com/datasets/5y9wdsg2zt
- License: CC BY 4.0
"""


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_name in EXAMPLES:
        src = RAW / src_rel
        if not src.exists():
            raise SystemExit(
                f"missing source image: {src} — regenerate data/raw first "
                "(see docs/datasets.md)"
            )
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_EDGE, MAX_EDGE))
            dst = OUT / dst_name
            im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
            width, height = im.size
        size_bytes = dst.stat().st_size
        if size_bytes >= SIZE_CAP_BYTES:
            raise SystemExit(f"{dst_name} is {size_bytes} bytes, over the 300KB cap")
        print(f"{dst_name:26s} {width}x{height}  {size_bytes / 1024:5.1f} KB")

    (OUT / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    print(f"wrote {OUT / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    build()
