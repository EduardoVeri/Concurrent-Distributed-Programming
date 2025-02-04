#!/bin/bash

RED="\e[1;31m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RESET="\e[0m"

# Verify if the current directory is the root of the project
if [ ! -f "build.sh" ]; then
    echo -e "\e[1;31mERROR: This script must be run from the project's root directory.\e[0m"
    exit 1
fi

# Parse the command line arguments
while getopts "veh" opt; do
    case ${opt} in
        v )
            echo -e "${YELLOW}Building in verbose mode${RESET}"
            VERBOSE=1
            define_flag="-DVERBOSE"
            ;;
        e )
            echo -e "${YELLOW}Building in evaluating mode${RESET}"
            EVALUATE=1
            define_flag="-DEVALUATE"
            ;;
        h )
            echo "Usage: build.sh [-v] [-e] [-h]"
            echo "Options:"
            echo "  -v    Build in verbose mode"
            echo "  -e    Build in evaluating mode"
            echo "  -h    Show this help message"
            exit 0
            ;;
        \? )
            echo "Usage: build.sh [-v] [-e] [-h]"
            exit 1
            ;;
    esac
done


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
gcc $(ls ../src/*.c | grep -v 'mpi.c') -I../inc -o libDiffusionEquation.so -fopenmp -shared -DBUILD_SHARED $define_flag
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully built libDiffusionEquation.so${RESET}"
else
    echo -e "${RED}ERROR: Failed to build libDiffusionEquation.so${RESET}"
fi
echo -e "${YELLOW}WARNING: The shared library does not support the MPI version yet.${RESET}"

if [ ! -z "$NVCC" ]; then
    nvcc -Xcompiler -fPIC -shared -I../inc ../src/cuda.cu -o libCUDAdiffusionEquation.so -arch=sm_50 $define_flag
    if [ $? -eq 0 ]; then
         echo -e "${GREEN}Successfully built libCUDAdiffusionEquation.so${RESET}"
    else
        echo -e "${RED}ERROR: Failed to build libCUDAdiffusionEquation.so${RESET}"
    fi
fi

# Build the executables
gcc ../src/sequential.c ../src/utils.c -I../inc -o sequential $define_flag
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully built sequential${RESET}"
else
    echo -e "${RED}ERROR: Failed to build sequential${RESET}"
fi

gcc ../src/omp.c ../src/utils.c -I../inc -o omp -fopenmp $define_flag
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully built omp${RESET}"
else
    echo -e "${RED}ERROR: Failed to build omp${RESET}"
fi

if [ ! -z "$NVCC" ]; then
    nvcc ../src/cuda.cu ../src/utils.c -I../inc -o cuda -arch=sm_50 $define_flag
    if [ $? -eq 0 ]; then
         echo -e "${GREEN}Successfully built cuda${RESET}"
    else
        echo -e "${RED}ERROR: Failed to build cuda${RESET}"
    fi
fi

if [ ! -z "$MPI" ]; then
    mpicc ../src/mpi.c ../src/utils.c -I../inc -o mpi -fopenmp $define_flag
    if [ $? -eq 0 ]; then
         echo -e "${GREEN}Successfully built mpi${RESET}"
    else
        echo -e "${RED}ERROR: Failed to build mpi${RESET}"
    fi
fi
