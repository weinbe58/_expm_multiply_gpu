#!/bin/bash

NVCC=$1
BUILD_DIR=$2
COMPILE_FLAGS=(-rdc=true -m64 -O3 --compiler-options '-fPIC' -I$BUILD_DIR/includes)
GENCODE_FLAGS=(-gencode arch=compute_35,code=sm_35)
LDFLAGS=(-lcudart -lcusparse)

$NVCC ${COMPILE_FLAGS[*]} ${LDFLAGS[*]} ${GENCODE_FLAGS[*]} --shared $BUILD_DIR/expm_multiply_cuda.cu -o $BUILD_DIR/lib/libexpm_multiply_cuda.so || exit 1


