#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=1
export CCACHE_DIR=/ccache

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/

echo set\(USE_OPENCL ON\) >>config.cmake
if [[ ${GPU} == cuda* ]]; then
	echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
	echo set\(CMAKE_CUDA_ARCHITECTURES "80;90"\) >>config.cmake
	echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
	echo set\(USE_CUDA ON\) >>config.cmake
fi
if [[ -f ${ADRENOACCL_SDK}/include/adrenoaccl.h ]]; then
    echo set\(USE_ADRENO_ACCL \"${ADRENOACCL_SDK}\"\) >> config.cmake
fi

cmake .. && make -j`nproc`
