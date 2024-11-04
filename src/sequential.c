#include <stdio.h>
#include "sequential.h"

#define N 100   // Tamanho da grade
#define T 1000  // Número de iterações
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0
// #define BUILD_SHARED

void show_matrix(double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%f ", C[i][j]);
        printf("\n");
    }
}

void sequential_diff_eq(double C[N][N], double C_new[N][N]) {
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * ((C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) / (DELTA_X * DELTA_X));
            }
        }
        // Atualizar matriz para a próxima iteração
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }
    }
}

#ifndef BUILD_SHARED
int main() {
    double C[N][N] = {0};      // Concentração inicial
    double C_new[N][N] = {0};  // Concentração para a próxima iteração
    // Inicializar uma concentração alta no centro
    C[N / 2][N / 2] = 1.0;
    // Executar a equação de difusão
    diff_eq(C, C_new);
    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", C[N / 2][N / 2]);

    // show_matrix(C);

    return 0;
}
#endif