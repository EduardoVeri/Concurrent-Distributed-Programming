#include <mpi.h>
#include <omp.h>

#include "diff_eq.h"
#include "utils.h"


double mpi_omp_diff_eq(double **C, double **C_new, DiffEqArgs *args,
                       int localN, int N, int rank, int size) {
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;
    double difmedio_local = 0.0;

    MPI_Request reqs[4];
    int req_count = 0;

    if (rank > 0) {
        MPI_Irecv(C[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(C[1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    if (rank < size - 1) {
        MPI_Irecv(C[localN + 1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(C[localN], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);


    double local_sum = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : local_sum)
    for (int i = 1; i <= localN; i++) {
        for (int j = 1; j < N - 1; j++) {
            C_new[i][j] =
                C[i][j] +
                D * DELTA_T *
                    ((C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4.0 * C[i][j]) / (DELTA_X * DELTA_X));

            local_sum += fabs(C_new[i][j] - C[i][j]);
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    double difmedio_global = global_sum / ((N - 2) * (N - 2));

    return difmedio_global;
}


#ifndef BUILD_SHARED
int main(int argc, char *argv[]) {
    struct timeval start_parallel, end_parallel;

    int required = MPI_THREAD_FUNNELED;  
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
        fprintf(stderr, "Error: The MPI library does not provide the required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check arguments
    if (argc != 7) {
        if (rank == 0) {
            printf("Usage: mpirun -np <num_procs> %s <I> <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int I = atoi(argv[1]);
    int N = atoi(argv[2]);
    int T = atoi(argv[3]);
    double D = atof(argv[4]);
    double DELTA_T = atof(argv[5]);
    double DELTA_X = atof(argv[6]);
    int num_threads = atoi(argv[7]);

    // Set number of threads
    omp_set_num_threads(num_threads);

    int localN = N / size;

    // Create the local matrices of size (localN+2) x N:
    //   +2 to store top and bottom halos.
    double **C = create_submatrix(localN + 2, N);
    double **C_new = create_submatrix(localN + 2, N);

    // Initialize everything to 0.0
    for (int i = 0; i < localN + 2; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    int global_start = rank * localN;
    int global_end = global_start + localN - 1;

    int mid_row = N / 2;
    int mid_col = N / 2;
    if (mid_row >= global_start && mid_row <= global_end) {
        int local_i = mid_row - global_start + 1;  // +1 because of halo offset
        C[local_i][mid_col] = 1.0;
    }

    DiffEqArgs args;
    args.N = N;
    args.D = D;
    args.DELTA_T = DELTA_T;
    args.DELTA_X = DELTA_X;

    // Time-step loop
    gettimeofday(&start_parallel, NULL);
    for (int t = 0; t < T; t++) {
        double difmedio = mpi_omp_diff_eq(C, C_new, &args, localN, N, rank, size);

        double **temp = C;
        C = C_new;
        C_new = temp;

#ifdef VERBOSE
        if ((t % 100) == 0 && rank == 0) {
            printf("Iteração %d - diferença média global = %g\n", t, difmedio);
        }
#endif
    }

    gettimeofday(&end_parallel, NULL);

#ifdef VERBOSE
    // Print the final concentration at the center from the rank that owns it
    if (mid_row >= global_start && mid_row <= global_end) {
        int local_i = mid_row - global_start + 1;
        printf("Rank %d => Concentração final no centro: %f\n", rank, C[local_i][mid_col]);
    }
#endif


    free_submatrix(C, localN + 2); 
    free_submatrix(C_new, localN + 2);

    double parallel_time =
        ((end_parallel.tv_sec * 1000000 + end_parallel.tv_usec) -
         (start_parallel.tv_sec * 1000000 + start_parallel.tv_usec)) /
        1000.0;

#ifdef EVALUATE
    if (rank == 0) {
        printf("Tempo paralelo:  %lf ms\n", parallel_time);
    }
#endif

    MPI_Finalize();
    return 0;
}
#endif
