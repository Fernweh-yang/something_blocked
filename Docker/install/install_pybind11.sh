#!/usr/bin/env bash

set -e

echo "Installing pybind11..."

cd /opt
git clone https://github.com/pybind/pybind11.git --branch v2.11.0 --single-branch --depth 1
cd pybind11
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
make install -j$(nproc)
rm -rf /opt/pybind11

echo "Installing pybind11... DONE"
