#!/usr/bin/env bash
set -e
mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
echo "Built OpenMP extension at $(pwd)"
