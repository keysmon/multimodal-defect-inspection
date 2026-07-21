#!/usr/bin/env bash
# Fetch the realistic walkthrough golden-set photos (Wikimedia Commons).
#
# The frozen dataset-crop golden set (walkthrough_golden.json) is deliberately
# hard: 256px context-free crops. This set is the realistic counterpart - full
# field photos with context, licensed CC BY / CC BY-SA / public domain.
# Attribution: data/manifests/walkthrough_realistic_attribution.md
#
# Usage (from the repo root): bash scripts/fetch_realistic_walkthrough.sh
set -euo pipefail

DEST="data/raw/realistic"
UA="DefectLensEval/1.0 (research; github keysmon/multimodal-defect-inspection)"
mkdir -p "$DEST"

fetch() { # name url sha256
  local name="$1" url="$2" sha="$3" path
  path="$DEST/$name"
  if [ ! -f "$path" ]; then
    echo "fetching $name"
    curl -sfL -A "$UA" -o "$path" "$url"
  fi
  echo "$sha  $path" | shasum -a 256 -c - >/dev/null || {
    echo "sha256 mismatch for $name" >&2
    exit 1
  }
}

fetch water_damaged_pub_wall.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/f/f8/A_water_damaged_public_house_wall_in_Broadstairs_Kent_England.jpg" \
  034f2ba660bff4459e957d6650c3637a4b3436731f1e1b7c9155279868281120

fetch fema_louisiana_interior.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/0/0e/FEMA_-_17647_-_Photograph_by_Patsy_Lynch_taken_on_10-18-2005_in_Louisiana.jpg" \
  9efb49b7becc7d027cd73e7745c45681dca8cf0909b3104eb15db2dfd35eba77

fetch peeling_paint_closeup.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/4/4e/Close-up_of_peeling_paint_on_the_wall_in_an_old_abandoned_building_%2848652561603%29.jpg" \
  d330744eedd3f674d474ebfd03f255168b3e69dc2e3cce24ea8116e7c4feb260

fetch cochem_facade.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/3/38/Cochem%2C_Unterbachstra%C3%9Fe%2C_Geb%C3%A4ude_--_2018_--_0053.jpg" \
  eecea6ac8f68a691c502eb2aaab7ef852a6ce4cd1906a5b09a1e6897df2c0269

fetch funchal_carbonation_rebar.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/3/36/Funchal_Carbonatation_Rebar.JPG" \
  2eacb14c89ba0165ab04c275038459b3ea431b25eba16d9a9502b758c2b6187b

fetch cracked_wall.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/7/79/A_cracked_wall.jpg" \
  de559778e909e476a8305d122d7b8fa48a59a20c51042fca2eba20df032bf051

fetch clifton_viaduct_cracks.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/c/cc/Clifton_viaduct_cracks_in_road_tunnel.jpg" \
  27bc39ca637c948d32266c0ca55dd28d4994eb8ea9a3c5308154127a43da46fe

fetch kellokoski_wall_detail.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/d/d6/A_wall_detail_of_building_3_of_Kellokosken_Ruukki_in_Kellokoski%2C_Tuusula%2C_Finland%2C_2022_April.jpg" \
  9a04be6f37453307ed9af48e433acbc24872b2757c70b7bc75cbe53e0fc71cd1

echo "realistic walkthrough photos ready in $DEST"
