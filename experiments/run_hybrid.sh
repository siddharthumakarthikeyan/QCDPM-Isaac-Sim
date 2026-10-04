#!/usr/bin/env bash
# E3 pose with drone winches in tension mode (hybrid) at several tilt limits.
set -u
ISAAC_SIM=${ISAAC_SIM:-$HOME/isaac-sim-standalone-6.0.1-linux-x86_64}
cd "$(dirname "$0")/.."
for t in 35 40 45 50; do
  log="results/hyb_tilt$t.log"
  "$ISAAC_SIM/python.sh" experiments/run_experiment.py --scenario E3 --tilt $t --hybrid > "$log" 2>&1 && grep "\[exp\]" "$log" \
    || { echo "FAILED: $t"; grep -m3 -A5 Traceback "$log"; }
done
echo "=== HYBRID DONE"
