#!/usr/bin/env bash

set -e

echo "Installing spdlog..."

cd /opt
git clone https://github.com/gabime/spdlog.git --branch v1.14.0 --single-branch --depth 1
cd spdlog
mkdir build && cd build
cmake .. -DSPDLOG_BUILD_EXAMPLE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
make install -j$(nproc)
rm -rf /opt/spdlog

echo "Installing spdlog... DONE"
