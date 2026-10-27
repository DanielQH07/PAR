#!/bin/bash
# Final 3-member zip (layout of 107; 178 = best test 0.8006): MetaCLIP B soup3 (base) + PE-B soup3 on the colour attributes only + PE-L member;
# colour attributes 1/3 each, the others MetaCLIP 0.5 + PE-L 0.5; Color-Other +0.10, Age-Adult -0.05.
# Usage (repo root): bash src/build_submission.sh BASE_ZIP PEL_PT ID NAME [PEB_PT (default exports/pebsoup3/pebsoup3.pt)]
set -e
base=$1 pel=$2 id=$3 name=$4 peb=${5:-exports/pebsoup3/pebsoup3.pt}
tmp=builds/ensemble_tmp_$name
python src/build_ensemble.py --base "$base" --equal --out "$tmp" --pe "$peb":all --pe "$pel":all \
  --bias UpperBody-Color-Other=0.10 --bias LowerBody-Color-Other=0.10 --bias Age-Adult=-0.05 --no-save
python src/tweak_zip.py "$tmp" "$name" "$id" --colour-weights member_0.pth=1,$(basename "$peb")=1,$(basename "$pel")=1 \
  --other-weights member_0.pth=1,$(basename "$pel")=1
rm -rf "$tmp" "$tmp.zip"
