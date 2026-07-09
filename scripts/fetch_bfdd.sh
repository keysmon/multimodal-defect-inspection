#!/usr/bin/env bash
# Fetch BFDD (Building Facade Defect Dataset, CC BY 4.0) to ~/datasets/bfdd.
# https://data.mendeley.com/datasets/9ych7czvyg/1
set -euo pipefail

DEST="${HOME}/datasets/bfdd"
URL="https://data.mendeley.com/public-files/datasets/9ych7czvyg/files/c1c5144b-cb20-4687-b514-d0bbec12209e/file_downloaded"
SHA="43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a"

mkdir -p "${DEST}"
cd "${DEST}"
if [ ! -f bfdd.tar.gz ]; then
  echo "Downloading BFDD (~528 MB)..."
  # -f: HTTP errors fail (don't save an error page as the tarball). On any
  # curl failure, remove the partial file so the next run re-downloads cleanly.
  curl -fSL -o bfdd.tar.gz "${URL}" || { rm -f bfdd.tar.gz; echo "download failed" >&2; exit 1; }
fi
# Self-heal a corrupt/partial tarball: on checksum mismatch, delete it and tell
# the user to re-run (otherwise the [ ! -f ] guard would never re-fetch it).
if ! echo "${SHA}  bfdd.tar.gz" | shasum -a 256 -c -; then
  echo "sha256 mismatch — removing bad tarball; re-run this script to re-download." >&2
  rm -f bfdd.tar.gz
  exit 1
fi
[ -d Dataset_1x ] || tar -xzf bfdd.tar.gz
echo "OK: ${DEST}/Dataset_1x"
