#!/usr/bin/env bash
# Fetch DCASE 2020 Task 2 dev data (fan + pump) - MIMII subset, CC BY-NC-SA 4.0.
# Layout after run:
#   ~/datasets/dcase2020t2/{fan,pump}/{train,test}/*.wav
#   data/raw/audio/{fan,pump} -> symlinks into the above
set -euo pipefail

DEST="${HOME}/datasets/dcase2020t2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$DEST"

# macOS ships bash 3.2 (no associative arrays) — use a lookup function.
md5_for() {
  case "$1" in
    dev_data_fan.zip)  echo 649bdfc06263ae7a838963f43b6641e6 ;;
    dev_data_pump.zip) echo 90e7091ef722b7238a7f1009365779cd ;;
    *) echo unknown ;;
  esac
}

for NAME in dev_data_fan.zip dev_data_pump.zip; do
  ZIP="${DEST}/${NAME}"
  if [[ ! -f "$ZIP" ]]; then
    echo "== downloading ${NAME} =="
    curl -L -o "$ZIP" "https://zenodo.org/records/3678171/files/${NAME}?download=1"
  fi
  echo "== verifying ${NAME} =="
  GOT=$(md5 -q "$ZIP")
  [[ "$GOT" == "$(md5_for "$NAME")" ]] || { echo "MD5 MISMATCH for ${NAME}: ${GOT}" >&2; exit 1; }
  MACHINE="${NAME#dev_data_}"; MACHINE="${MACHINE%.zip}"
  if [[ ! -d "${DEST}/${MACHINE}" ]]; then
    echo "== extracting ${NAME} =="
    unzip -q "$ZIP" -d "$DEST"
  fi
done

mkdir -p "${REPO_ROOT}/data/raw/audio"
for MACHINE in fan pump; do
  ln -sfn "${DEST}/${MACHINE}" "${REPO_ROOT}/data/raw/audio/${MACHINE}"
done
echo "== done =="
find "${REPO_ROOT}/data/raw/audio/" -name "*.wav" | head -3
for MACHINE in fan pump; do
  printf "%s: train=%s test=%s\n" "$MACHINE" \
    "$(ls "${DEST}/${MACHINE}/train" | wc -l | tr -d ' ')" \
    "$(ls "${DEST}/${MACHINE}/test" | wc -l | tr -d ' ')"
done
