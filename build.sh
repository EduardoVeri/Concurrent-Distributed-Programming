#!/bin/bash

# Verify if the current directory is the root of the project
if [ ! -f "build.sh" ]; then
    echo -e "\e[1;31mERROR: This script must be run from the project's root directory.\e[0m"
    exit 1
fi

RED="\e[1;31m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RESET="\e[0m"

NVCC=$(which nvcc)
if [ -z "$NVCC" ]; then
    echo -e "${YELLOW}WARNING: NVIDIA CUDA Compiler (nvcc) not found.${RESET}"
    echo -e "${YELLOW}Please install the NVIDIA CUDA Toolkit to build the CUDA version.${RESET}"
fi

MPI=$(which mpicc)
if [ -z "$MPI" ]; then
    echo -e "${YELLOW}WARNING: Message Passing Interface (MPI) Compiler (mpicc) not found.${RESET}"
    echo -e "${YELLOW}Please install the OpenMPI library to build the MPI version.${RESET}"
fi

# Create the build directory
mkdir -p build
cd build

# Build the shared library
gcc $(ls ../src/*.c | grep -v 'mpi.c') -I../inc -o libDiffusionEquation.so -fopenmp -shared -DBUILD_SHARED
echo -e "${GREEN}Successfully built libDiffusionEquation.so${NC}"
echo -e "${YELLOW}WARNING: The shared library does not support the MPI version yet.${RESET}"

if [ ! -z "$NVCC" ]; then
    nvcc -Xcompiler -fPIC -shared -I../inc ../src/cuda.cu -o libCUDAdiffusionEquation.so -arch=sm_50
    echo -e "${GREEN}Successfully built libCUDAdiffusionEquation.so${RESET}"
fi

# Build the executables
gcc ../src/sequential.c ../src/utils.c -I../inc -o sequential
echo -e "${GREEN}Successfully built sequential${RESET}"
gcc ../src/omp.c ../src/utils.c -I../inc -o omp -fopenmp
echo -e "${GREEN}Successfully built omp${RESET}"

if [ ! -z "$NVCC" ]; then
    nvcc ../src/cuda.cu ../src/utils.c -I../inc -o cuda -arch=sm_50
    echo -e "${GREEN}Successfully built cuda${RESET}"
fi

if [ ! -z "$MPI" ]; then
    mpicc ../src/mpi.c ../src/utils.c -I../inc -o mpi -fopenmp
    echo -e "${GREEN}Successfully built mpi${RESET}"
fi
