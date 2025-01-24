#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double **create_matrix(int N);
double **create_matrix_and_init(int N);
void free_matrix(double **matrix, int N);
void salvar_matriz(double **matriz, int linhas, int colunas, const char *nome_arquivo);
double *create_flatten_matrix(int N);
double *create_flatten_matrix_and_init(int N);
void free_flatten_matrix(double *matrix);
double **create_submatrix(int rows, int cols);
void free_submatrix(double **mat, int rows);

#endif
