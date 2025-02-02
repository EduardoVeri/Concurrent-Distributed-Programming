#include "diff_eq.h"
#include "utils.h"

double sequential_diff_eq(double **C, double **C_new, DiffEqArgs *args) {
    int N = args->N;
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;
    double difmedio = 0.;

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
    // Check arguments
    if (argc != 7) {
        printf("Usage: %s <I> <N> <T> <D> <DELTA_T> <DELTA_X>\n", argv[0]);
        return 1;
    }

    int I = atoi(argv[1]);
    int N = atoi(argv[2]);
    int T = atoi(argv[3]);
    double D = atof(argv[4]);
    double DELTA_T = atof(argv[5]);
    double DELTA_X = atof(argv[6]);

    struct timeval start_parallel, end_parallel;

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
            double difmedio = sequential_diff_eq(C, C_new, &args);

            // Swap matrices
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
        // Show time in seconds
        printf("%f\n", get_elapsed_time(start_parallel, end_parallel));
#endif

#ifdef VERBOSE
        // Show concentration at the center
        printf("Concentração final no centro: %f\n", C[N / 2][N / 2]);
#endif

        // show_matrix(C);
        free_matrix(C, N);
        free_matrix(C_new, N);
    }
    return 0;
}
#endif