#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double **create_matrix(int N);
double **create_matrix_and_init(int N);
void free_matrix(double **matrix, int N);
void salvar_matriz(double **matriz, int linhas, int colunas, const char *nome_arquivo);

#ifdef __cplusplus
}
#endif

#endif
