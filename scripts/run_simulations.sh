#!/bin/bash

if [ -f "run_simulations.sh" ]; then
    cd ..
fi

cd build
mkdir -p results
cd results

N=50
N_EVAL=15
N_STEPS=50

../sequential $N_EVAL $N $N_STEPS 0.1 0.01 1.0 > seq.txt

for i in 1 2 4 8 16 32; do
    ../omp $N_EVAL $N $N_STEPS 0.1 0.01 1.0 $i > omp_$i.txt
done

../cuda $N_EVAL $N $N_STEPS 0.1 0.01 1.0 > cuda.txt