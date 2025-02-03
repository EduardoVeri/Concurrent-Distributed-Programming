#!/bin/bash

RED="\e[1;31m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RESET="\e[0m"

if [ -f "run_simulations.sh" ]; then
    cd ..
fi

cd build
rm -r results
mkdir -p results
cd results

N=50
N_EVAL=15
N_STEPS=50

echo -e "${YELLOW}Running simulations...${RESET}"
../sequential $N_EVAL $N $N_STEPS 0.1 0.01 1.0 > seq.txt
echo -e "${GREEN}Sequential simulation completed${RESET}"

# for i in 1 2 4 8 16 32; do
#     echo -e "${YELLOW}Running OpenMP simulation with $i threads...${RESET}"
#     ../omp $N_EVAL $N $N_STEPS 0.1 0.01 1.0 $i > omp_$i.txt
#     echo -e "${GREEN}OpenMP simulation with $i threads completed${RESET}"
# done

echo -e "${YELLOW}Running OpenMP simulation with 4 thread...${RESET}"
../omp $N_EVAL $N $N_STEPS 0.1 0.01 1.0 4 > omp_4.txt
echo -e "${GREEN}OpenMP simulation with 4 thread completed${RESET}"

echo -e "${YELLOW}Running CUDA simulation...${RESET}"
../cuda $N_EVAL $N $N_STEPS 0.1 0.01 1.0 > cuda.txt
echo -e "${GREEN}CUDA simulation completed${RESET}"

for i in 1 2 4 8 16 32; do
    echo -e "${YELLOW}Running MPI simulation with $i processes...${RESET}"
    mpirun -np $i --oversubscribe ../mpi $N_EVAL $N $N_STEPS 0.1 0.01 1.0 1 > omp_$i.txt
    echo -e "${GREEN}MPI simulation with $i processes completed${RESET}"
done
