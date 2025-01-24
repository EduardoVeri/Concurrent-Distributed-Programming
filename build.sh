#!/bin/bash

# Verify if the current directory is the root of the project
if [ ! -f "build.sh" ]; then
    echo -e "\e[1;31mERROR: This script must be run from the project's root directory.\e[0m"
    exit 1
fi

NVCC=$(which nvcc)
if [ -z "$NVCC" ]; then
    echo -e "\e[1;33mWARNING: NVIDIA CUDA Compiler (nvcc) not found.\e[0m"
    echo -e "\e[1;33mPlease install the NVIDIA CUDA Toolkit to build the CUDA version.\e[0m"
fi

MPI=$(which mpicc)
if [ -z "$MPI" ]; then
    echo -e "\e[1;33mWARNING: Message Passing Interface (MPI) Compiler (mpicc) not found.\e[0m"
    echo -e "\e[1;33mPlease install the OpenMPI library to build the MPI version.\e[0m"
fi

# Create the build directory
mkdir -p build
cd build

# Build the shared library
gcc $(ls ../src/*.c | grep -v 'mpi.c') -I../inc -o libDiffusionEquation.so -fopenmp -shared -DBUILD_SHARED
echo -e "\e[1;32mSuccessfully built libDiffusionEquation.so\e[0m"
echo -e "\e[1;33mWARNING: The shared library does not support the MPI version yet.\e[0m"

if [ ! -z "$NVCC" ]; then
    nvcc -Xcompiler -fPIC -shared -I../inc ../src/cuda.cu -o libCUDAdiffusionEquation.so -arch=sm_50
    echo -e "\e[1;32mSuccessfully built libCUDAdiffusionEquation.so\e[0m"
fi

# Build the executables
gcc ../src/sequential.c ../src/utils.c -I../inc -o sequential
echo -e "\e[1;32mSuccessfully built sequential\e[0m"
gcc ../src/omp.c ../src/utils.c -I../inc -o omp -fopenmp
echo -e "\e[1;32mSuccessfully built omp\e[0m"

if [ ! -z "$NVCC" ]; then
    nvcc ../src/cuda.cu ../src/utils.c -I../inc -o cuda -arch=sm_50
    echo -e "\e[1;32mSuccessfully built cuda\e[0m"
fi

if [ ! -z "$MPI" ]; then
    mpicc ../src/mpi.c ../src/utils.c -I../inc -o mpi -fopenmp
    echo -e "\e[1;32mSuccessfully built mpi\e[0m"
fi