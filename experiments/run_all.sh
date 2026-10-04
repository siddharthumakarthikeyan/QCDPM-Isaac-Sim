#!/usr/bin/env bash
# Runs every Isaac experiment sequentially (headless), then the comparison against theory.
set -u
ISAAC_SIM=${ISAAC_SIM:-$HOME/isaac-sim-standalone-6.0.1-linux-x86_64}
cd "$(dirname "$0")/.."
python3 analysis/theory.py > results/theory.txt 2>&1
for s in E1_straight E1_crossed E2_reconfig E3_tilt35 E3_tilt45; do
  echo "=== $s"
  "$ISAAC_SIM/python.sh" experiments/run_experiment.py --scenario "$s" > "results/$s.log" 2>&1 \
    && grep "\[exp\]" "results/$s.log" || { echo "FAILED: $s (see results/$s.log)"; grep -m5 -A3 Traceback "results/$s.log"; }
done
echo "=== ALL DONE"
