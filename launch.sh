#!/usr/bin/env bash
# Launch the mobile-CDPR sim with Isaac Sim 6.0's bundled ROS 2 Humble (Python 3.12).
#   ISAAC_SIM=~/isaac-sim-standalone-6.0.1-linux-x86_64 ./launch.sh [--headless] [--demo] [--no-sensors] ...
set -e
ISAAC_SIM=${ISAAC_SIM:-$HOME/isaac-sim-standalone-6.0.1-linux-x86_64}
[ -x "$ISAAC_SIM/python.sh" ] || { echo "set ISAAC_SIM to your Isaac Sim 6.0 install (not found: $ISAAC_SIM)"; exit 1; }
unset AMENT_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH   # never mix system py3.10 ROS into Isaac
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ISAAC_SIM/exts/isaacsim.ros2.core/humble/lib"
export PYTHONPATH="$ISAAC_SIM/exts/isaacsim.ros2.core/humble/rclpy"
exec "$ISAAC_SIM/python.sh" "$(cd "$(dirname "$0")" && pwd)/run_sim.py" "$@"
