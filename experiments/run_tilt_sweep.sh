#!/usr/bin/env bash
# E3 pose (drones at 3.0 m) with different drone tilt limits, without and with drone-aware tension caps.
set -u
ISAAC_SIM=${ISAAC_SIM:-$HOME/isaac-sim-standalone-6.0.1-linux-x86_64}
cd "$(dirname "$0")/.."
run() {
  "$ISAAC_SIM/python.sh" experiments/run_experiment.py --scenario E3 "$@" > "results/sweep_$(echo "$@" | tr ' -' '_').log" 2>&1 \
    && grep "\[exp\]" "results/sweep_$(echo "$@" | tr ' -' '_').log" || echo "FAILED: $*"
}
for t in 35 38 40 42 45 50; do run --tilt $t; done
for t in 40 45 50; do run --tilt $t --drone-aware; done
echo "=== SWEEP DONE"
