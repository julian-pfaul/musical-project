#!/usr/bin/sh

set -e

libtorch_path="$(cat LIBTORCH_PATH)/share/cmake/Torch"

echo ${libtorch_path}

mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH="${libtorch_path}" ..
cmake --build . --config Release
