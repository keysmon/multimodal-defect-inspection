#!/usr/bin/env bash
# Fetch the Insulator-Defect Detection dataset (CC BY 4.0) to ~/datasets/insulator.
# https://figshare.com/articles/dataset/VOC_zip/21200986
# (DOI 10.6084/m9.figshare.21200986) — 1,600 grid transmission-insulator images
# with VOC detection annotations (normal / pollution-flashover / broken).
# NOTE: grid insulators, NOT electrical panels — caveat stated in the README.
set -euo pipefail

DEST="${HOME}/datasets/insulator"
URL="https://ndownloader.figshare.com/files/37587370"
ZIP="VOC.zip"
# figshare publishes md5:1cd1776a7ea48bbc69c51c656727d915 for the archive;
# the sha256 below was computed from the first verified download (2026-07-21).
SHA="71b3c7f469ebd4f9349558409b09616c4768ef5e1ee2a67083747781a9f3934d"

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
  echo "Downloading insulator VOC dataset (~2.4 GB)..."
  curl -fSL -o "${ZIP}" "${URL}" || { rm -f "${ZIP}"; echo "download failed" >&2; exit 1; }
fi
if [ "${SHA}" = "__PIN_ON_FIRST_DOWNLOAD__" ]; then
  echo "First download — pin this sha256 into fetch_insulator.sh:" >&2
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
