#!/bin/bash
# contact sheet: one row per material = reference | shader ball | raking close-up | 3x3 front-on.
# usage: sheet.sh <renderdir> <refdir> <out.jpg>
#   <renderdir> holds <Name>_ball.jpg, <Name>_close.jpg, <Name>_front.jpg from lookdev.py
#   <refdir>    holds <Name>.jpg, the reference each material was matched to
# keep the references somewhere durable: a temp folder is emptied by a reboot
R="$1"; REF="$2"; OUT="$3"
H="${HB:-$(dirname "$(command -v hython)")}/hoiiotool"
T=$(mktemp -d); rows=()
for b in "$R"/*_ball.jpg; do
  n=$(basename "$b" _ball.jpg); lbl=$(echo "$n" | tr '_' ' ')
  ref="$REF/$n.jpg"; [ -f "$ref" ] || ref="$b"
  "$H" "$ref" --fit:pad=1 720x480 "$R/${n}_ball.jpg" --fit:pad=1 720x480 "$R/${n}_close.jpg" --fit:pad=1 720x480 "$R/${n}_front.jpg" --fit:pad=1 720x480 \
    --mosaic:pad=8 4x1 --ch R,G,B --origin +0+56 --fullsize 2904x544+0+0 --croptofull \
    --text:x=12:y=40:size=32:color=1,1,1 "$lbl" --text:x=740:y=40:size=26:color=0.6,0.6,0.6 "shader ball" \
    --text:x=1468:y=40:size=26:color=0.6,0.6,0.6 "raking close-up" --text:x=2196:y=40:size=26:color=0.6,0.6,0.6 "3x3 tiling, front on" \
    -d uint8 -o "$T/$n.tif" && rows+=("$T/$n.tif")
done
[ ${#rows[@]} -gt 0 ] || { echo "no renders found in $R"; exit 1; }
"$H" "${rows[@]}" --mosaic:pad=10 1x${#rows[@]} -o "$OUT" && echo "wrote $OUT (${#rows[@]} rows)"
rm -rf "$T"
