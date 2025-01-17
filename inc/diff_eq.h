#ifndef DIFF_EQUATION_H
#define DIFF_EQUATION_H

typedef struct {
    int N;
    double D;
    double DELTA_T;
    double DELTA_X;
} DiffEqArgs;

#ifdef __cplusplus
extern "C"{
#endif

double sequential_diff_eq(double **C, double **C_new, DiffEqArgs *args);
double omp_diff_eq(double **C, double **C_new, DiffEqArgs *args);
double cuda_diff_eq(DiffEqArgs *args);

#ifdef __cplusplus
}
#endif

#endif