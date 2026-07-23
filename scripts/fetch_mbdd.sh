#!/usr/bin/env bash
# Fetch MBDD2025 (Multi-material Building Defect Detection, CC BY 4.0)
# to ~/datasets/mbdd2025. https://zenodo.org/records/15622584
# 14,471 UAV images, 5 defect classes (crack, leakage, corrosion, abscission,
# bulge) with detection annotations; classification crops are derived by
# scripts/prepare_mbdd.py.
set -euo pipefail

DEST="${HOME}/datasets/mbdd2025"
URL="https://zenodo.org/api/records/15622584/files/MBDD2025.zip/content"
# Zenodo publishes md5:b2dfdce060ef687c327b1f8203b52636 for MBDD2025.zip;
# the sha256 below was computed from the first verified download (2026-07-21).
SHA="db37469e0ee59be132d0e3773affec89a1c49fad3a873a9d47e7221bcfc3f95e"

# macOS ships `shasum`; Ubuntu (e.g. the DLAMI) ships `sha256sum`. Support both.
sha256_check() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c -
  else
    sha256sum -c -
  fi
}

mkdir -p "${DEST}"
cd "${DEST}"
if [ ! -f MBDD2025.zip ]; then
  echo "Downloading MBDD2025 (~2.5 GB)..."
  # -f: HTTP errors fail (don't save an error page as the archive). On any
  # curl failure, remove the partial file so the next run re-downloads cleanly.
  curl -fSL -o MBDD2025.zip "${URL}" || { rm -f MBDD2025.zip; echo "download failed" >&2; exit 1; }
fi
if [ "${SHA}" = "__PIN_ON_FIRST_DOWNLOAD__" ]; then
  echo "First download — pin this sha256 into fetch_mbdd.sh:" >&2
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 MBDD2025.zip; else sha256sum MBDD2025.zip; fi
  exit 1
fi
# Self-heal a corrupt/partial archive: on checksum mismatch, delete it and tell
# the user to re-run (otherwise the [ ! -f ] guard would never re-fetch it).
if ! echo "${SHA}  MBDD2025.zip" | sha256_check; then
  echo "sha256 mismatch — removing bad archive; re-run this script to re-download." >&2
  rm -f MBDD2025.zip
  exit 1
fi
[ -d MBDD2025 ] || unzip -q MBDD2025.zip
echo "OK: ${DEST}"
