// Write everything in with english, please.

#include <stdio.h>
#include <stdlib.h>
#include "sequential.h"

void sequential_diff_eq(double **C, double **C_new, DiffEqArgs *args){
    int N = args->N;
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C_new[i][j] = C[i][j] + D * DELTA_T * ((C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) / (DELTA_X * DELTA_X));
        }
    }

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C[i][j] = C_new[i][j];
        }
    }
}

double ** create_matrix(int N){
    double **matrix = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    return matrix;
}

double ** create_matrix_and_init(int N){
    double **matrix = create_matrix(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int N){
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

#ifndef BUILD_SHARED
int main(int argc, char *argv[]) {

    // Check arguments
    if (argc != 6) {
        printf("Usage: %s <N> <T> <D> <DELTA_T> <DELTA_X>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    double D = atof(argv[3]);
    double DELTA_T = atof(argv[4]);
    double DELTA_X = atof(argv[5]);

    // Create matrix
    double **C = create_matrix_and_init(N); 
    double **C_new = create_matrix(N);

    // Initial condition
    C[N / 2][N / 2] = 1.0;

    // Create struct with arguments
    DiffEqArgs args = {N, D, DELTA_T, DELTA_X};

    // Call the function T times
    for (int t = 0; t < T; t++) {
        sequential_diff_eq(C, C_new, &args);
    }

    // Show concentration at the center
    printf("Concentração final no centro: %f\n", C[N / 2][N / 2]);

    // show_matrix(C);

    return 0;
}
#endif