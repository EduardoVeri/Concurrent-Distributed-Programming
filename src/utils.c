#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

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

double * create_flatten_matrix(int N){
    double *matrix = (double *)malloc(N * N * sizeof(double));
    return matrix;
}

double * create_flatten_matrix_and_init(int N){
    double *matrix = create_flatten_matrix(N);
    for (int i = 0; i < N * N; i++) {
        matrix[i] = 0;
    }
    return matrix;
}

void free_flatten_matrix(double *matrix){
    free(matrix);
}

void salvar_matriz(double **matriz, int linhas, int colunas, const char *nome_arquivo) {
    FILE *arquivo = fopen(nome_arquivo, "w");
    if (!arquivo) {
        perror("Erro ao abrir o arquivo para salvar matriz");
        return;
    }

    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {
            fprintf(arquivo, "%.6f ", matriz[i][j]); // Salva com 6 casas decimais
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);
    printf("Matriz salva no arquivo: %s\n", nome_arquivo);
}

#ifdef __cplusplus
}
#endif