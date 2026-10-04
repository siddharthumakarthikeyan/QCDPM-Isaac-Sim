#!/usr/bin/env bash
# Build the C++ core as a Python module for the system Python (analysis, tests) and for Isaac Sim's Python.
# The modules land in cdpr_sim/ next to the Python package.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ISAAC_SIM=${ISAAC_SIM:-$HOME/isaac-sim-standalone-6.0.1-linux-x86_64}
PYBIND=$(python3 -m pybind11 --cmakedir)

build() {  # name, python executable
  cmake -S "$HERE" -B "$HERE/build/$1" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND" \
        -DPython_EXECUTABLE="$2" -DPYBIND11_FINDPYTHON=ON -DCMAKE_INSTALL_PREFIX="$HERE/../cdpr_sim" > /dev/null
  cmake --build "$HERE/build/$1" -j"$(nproc)"
  cmake --install "$HERE/build/$1" > /dev/null
}

build system "$(command -v python3)"
if [ -x "$ISAAC_SIM/kit/python/bin/python3" ]; then
  build isaac "$ISAAC_SIM/kit/python/bin/python3"
else
  echo "Isaac Sim not found at $ISAAC_SIM: built for the system Python only"
fi
ls -1 "$HERE/../cdpr_sim/"cdpr_core*.so
