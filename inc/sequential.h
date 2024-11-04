#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

typedef struct {
    int N;
    double D;
    double DELTA_T;
    double DELTA_X;
} DiffEqArgs;


void sequential_diff_eq(double **C, double **C_new, DiffEqArgs *args);

#endif