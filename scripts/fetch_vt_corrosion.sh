#!/usr/bin/env bash
# Fetch the VT Corrosion Condition State dataset (CC0) to ~/datasets/vt_corrosion.
# https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663
# (DOI 10.7294/16624663) — 440 bridge-inspection images annotated with AASHTO/BIRM
# corrosion condition states [good, fair, poor, severe].
set -euo pipefail

DEST="${HOME}/datasets/vt_corrosion"
URL="https://ndownloader.figshare.com/files/31729733"
ZIP="corrosion_condition_state.zip"
# figshare publishes md5:bbc9b8b8a5bb065e9ace028893dfd983 for the archive;
# the sha256 below was computed from the first verified download (2026-07-21).
SHA="45f0ec8b26f1c09d707f3010af359a28e0985d385d2bf6d98b5d4dd308e9dbe5"

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
if [ ! -f "${ZIP}" ]; then
  echo "Downloading VT corrosion condition state dataset (~333 MB)..."
  curl -fSL -o "${ZIP}" "${URL}" || { rm -f "${ZIP}"; echo "download failed" >&2; exit 1; }
fi
if [ "${SHA}" = "__PIN_ON_FIRST_DOWNLOAD__" ]; then
  echo "First download — pin this sha256 into fetch_vt_corrosion.sh:" >&2
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "${ZIP}"; else sha256sum "${ZIP}"; fi
  exit 1
fi
if ! echo "${SHA}  ${ZIP}" | sha256_check; then
  echo "sha256 mismatch — removing bad archive; re-run this script to re-download." >&2
  rm -f "${ZIP}"
  exit 1
fi
[ -d extracted ] || { mkdir -p extracted && unzip -q "${ZIP}" -d extracted; }
echo "OK: ${DEST}/extracted"
