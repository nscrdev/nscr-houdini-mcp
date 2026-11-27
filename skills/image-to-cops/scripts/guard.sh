#!/bin/bash
# run one heavy command under a memory guard. usage: guard.sh <logfile> <label> <timeout_s> <cmd...>
# refuses to start, or kills the command, when system memory leaves the safe zone.
LOG="$1"; LABEL="$2"; TMO="$3"; shift 3
freepct(){ memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2);print $2}'; }
swapmb(){ sysctl -n vm.swapusage | awk '{gsub("M","",$6);print int($6)}'; }
F0=$(freepct); S0=$(swapmb)
# swap already in use stays allocated long after the memory is free again, so judge swap by its growth
if [ "${F0:-0}" -lt 45 ]; then echo "$LABEL REFUSED free=${F0}% swap=${S0}M" | tee -a "$LOG"; exit 3; fi
T0=$(date +%s); "$@" > "$LOG.$LABEL.out" 2>&1 & PID=$!
MINF=$F0; MAXS=$S0; KILLED=""
while kill -0 $PID 2>/dev/null; do
  sleep 4; F=$(freepct); S=$(swapmb); [ -n "$F" ] && [ "$F" -lt "$MINF" ] && MINF=$F; [ "$S" -gt "$MAXS" ] && MAXS=$S
  EL=$(( $(date +%s)-T0 ))
  if [ -n "$F" ] && [ "$F" -lt 25 ]; then KILLED="free ${F}%"; fi
  if [ $((S-S0)) -gt 3000 ]; then KILLED="swap grew $((S-S0))M"; fi
  if [ "$EL" -gt "$TMO" ]; then KILLED="timeout ${EL}s"; fi
  if [ -n "$KILLED" ]; then kill -9 $PID 2>/dev/null; break; fi
done
wait $PID 2>/dev/null; RC=$?
echo "$LABEL rc=$RC $(( $(date +%s)-T0 ))s free_start=${F0}% free_min=${MINF}% swap_start=${S0}M swap_max=${MAXS}M ${KILLED:+KILLED $KILLED} $(tail -1 "$LOG.$LABEL.out" | cut -c1-90)" | tee -a "$LOG"
[ -n "$KILLED" ] && exit 4; exit $RC
