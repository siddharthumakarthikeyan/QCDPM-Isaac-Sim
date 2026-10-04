#!/bin/bash
cd "$(dirname "$0")/.."
for cfg in "32 3" "16 2" "8 2"; do set -- $cfg
  ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh tools/render_shots.py --rec tower --only carry --out t_$1_$2 --spp $1 --subframes $2 --max-frames 16 > logs/timing_$1_$2.log 2>&1
  echo "spp $1 sub $2: $(grep '\[shots\]' logs/timing_$1_$2.log)" >> logs/timing.txt
done
echo done >> logs/timing.txt
