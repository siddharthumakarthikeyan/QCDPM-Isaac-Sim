#!/usr/bin/env bash
# Stop any running Isaac Sim process started from this project (GUI or headless).
# Run it on its own (not chained with a launch in the same shell command).
pkill -f "kit/python/bin/python3 .*(isaac_cdpr|tests/isaac_|run_sim\.py|run_task\.py|experiments/run_|tools/render)" \
  && echo "stopped" || echo "nothing running"
