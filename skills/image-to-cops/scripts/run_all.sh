#!/bin/bash
# bake then render every material, one process at a time, each under the memory guard.
# stops at the first failure, kill or refusal, so a problem costs one item and not the machine.
# usage: run_all.sh <recipes.py> <count> <resolution> <texdir> <renderdir> [first]
S="$(cd "$(dirname "$0")" && pwd)"
REC="$1"; COUNT="$2"; RES="$3"; TEX="$4"; OUT="$5"; FIRST="${6:-1}"
HY="${HB:+$HB/}hython"; LOG="$OUT/_guard.log"; mkdir -p "$TEX" "$OUT"
for i in $(seq "$FIRST" "$COUNT"); do
  "$S/guard.sh" "$LOG" "bake$i" 900 "$HY" "$S/bake.py" "$REC" "$i" "$RES" "$TEX" b16 || { echo "STOPPED at bake$i" | tee -a "$LOG"; exit 1; }
  NAME=$(grep -o 'BAKED [^ ]*' "$LOG.bake$i.out" | tail -1 | cut -d' ' -f2)
  "$S/guard.sh" "$LOG" "shot$i" 900 "$HY" "$S/lookdev.py" "$TEX" "$NAME" "$OUT" || { echo "STOPPED at shot$i" | tee -a "$LOG"; exit 1; }
done
echo "ALL DONE" | tee -a "$LOG"
