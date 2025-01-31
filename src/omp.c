#include <omp.h>

#include "diff_eq.h"
#include "utils.h"

double omp_diff_eq(double **C, double **C_new, DiffEqArgs *args) {
    int N = args->N;
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;
    double difmedio = 0.;

#pragma omp parallel for collapse(2) reduction(+ : difmedio)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C_new[i][j] = C[i][j] + D * DELTA_T * ((C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) / (DELTA_X * DELTA_X));
            difmedio += fabs(C_new[i][j] - C[i][j]);
        }
    }

    return difmedio / ((N - 2) * (N - 2));
}

#ifndef BUILD_SHARED
int main(int argc, char *argv[]) {
    struct timeval start_parallel, end_parallel;

    // Check arguments
    if (argc != 8) {
        printf("Usage: %s <I> <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>\n", argv[0]);
        return 1;
    }

    int I = atoi(argv[1]);
    int N = atoi(argv[2]);
    int T = atoi(argv[3]);
    double D = atof(argv[4]);
    double DELTA_T = atof(argv[5]);
    double DELTA_X = atof(argv[6]);
    int NUM_THREADS = atoi(argv[7]);

    // Set number of threads
    omp_set_num_threads(NUM_THREADS);

    for (int i = 0; i < I; i++) {

        // Create matrix
        double **C = create_matrix_and_init(N);
        double **C_new = create_matrix(N);

        // Initial condition
        C[N / 2][N / 2] = 1.0;

        // Create struct with arguments
        DiffEqArgs args = {N, D, DELTA_T, DELTA_X};

        gettimeofday(&start_parallel, NULL);

        // Call the function T times
        for (int t = 0; t < T; t++) {
            double difmedio = omp_diff_eq(C, C_new, &args);

            // Swap pointers
            double **temp = C;
            C = C_new;
            C_new = temp;

#ifdef VERBOSE
            if ((t % 100) == 0)
                printf("interacao %d - diferenca = %g\n", t, difmedio);
#endif
        }

        gettimeofday(&end_parallel, NULL);

#ifdef EVALUATE
        // print in seconds
        printf("%f\n", (double)(end_parallel.tv_sec - start_parallel.tv_sec) + (double)(end_parallel.tv_usec - start_parallel.tv_usec) / 1000000);
#endif
        // Show concentration at the center
#ifdef VERBOSE
        printf("Concentração final no centro: %f\n", C[N / 2][N / 2]);
#endif

        // salvar_matriz(C, N, N, "matriz_omp.txt");

        // show_matrix(C);
        free_matrix(C, N);
        free_matrix(C_new, N);

    }

    return 0;
}
#endif