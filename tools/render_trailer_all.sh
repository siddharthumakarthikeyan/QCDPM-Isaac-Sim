#!/bin/bash
# Render every trailer shot (path traced), one Isaac process per recording, ~1 s per frame.
#   tools/render_trailer_all.sh [recordings...]      default: track plan tower pyramid compound
# A launch that writes no frame within 3 minutes is treated as hung, killed and retried (seen once at startup).
cd "$(dirname "$0")/.."
RECS=${@:-track plan tower pyramid compound}
for r in $RECS; do
  for attempt in 1 2 3; do
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/render_shots.py --rec $r > logs/shots_$r.log 2>&1 &
    pid=$!
    start=$(date +%s)
    hung=0
    while kill -0 $pid 2>/dev/null; do
      sleep 20
      # hung = started more than 3 minutes ago and has not written a single frame since it started
      age=$(( $(date +%s) - start ))
      new=$(find renders/trailer -name "f_*.png" -newermt "@$start" 2>/dev/null | head -1)
      if [ $age -gt 180 ] && [ -z "$new" ]; then hung=1; pkill -9 -P $pid; kill -9 $pid; break; fi
    done
    wait $pid 2>/dev/null; code=$?
    [ $hung -eq 0 ] && break
    echo "$r hung on attempt $attempt, retrying" >> logs/shots_all.log
  done
  echo "$r exit $code" >> logs/shots_all.log
done
echo done >> logs/shots_all.log
