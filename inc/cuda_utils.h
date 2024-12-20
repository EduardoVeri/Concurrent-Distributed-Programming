#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "diff_eq.h"

#ifdef __cplusplus
extern "C"{
#endif

void cuda_init(double *h_C_flat, double *h_C_new_flat, DiffEqArgs *args);
void cuda_get_result(double *h_C_flat, int N);
void cuda_finalize();

#ifdef __cplusplus
}
#endif
#endif
