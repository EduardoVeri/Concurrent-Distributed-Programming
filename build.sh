#!/bin/bash

# Verify if the current directory is the root of the project
if [ ! -f "build.sh" ]; then
    echo "Please run this script from the root of the project"
    exit 1
fi

# Create the build directory
mkdir -p build
cd build

# Build the shared library
gcc ../src/*.c -I../inc -o libDiffusionEquation.so -fopenmp -shared -DBUILD_SHARED
nvcc -Xcompiler -fPIC -shared -I../inc ../src/cuda.cu -o libCUDAdiffusionEquation.so -arch=sm_50

# Build the executables
gcc ../src/sequential.c ../src/utils.c -I../inc -o sequential
gcc ../src/omp.c ../src/utils.c -I../inc -o omp -fopenmp
nvcc ../src/cuda.cu ../src/utils.c -I../inc -o cuda -arch=sm_50
