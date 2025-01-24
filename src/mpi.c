#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "diff_eq.h"  // Where you have your struct DiffEqArgs, etc.
#include "utils.h"    // Where you have create_matrix, free_matrix, etc.

//-----------------------------------------------------------------------
// Hybrid MPI+OpenMP implementation of the diffusion step.
//
// We assume a 1D decomposition in the 'i' (row) dimension.
//
// Global domain: 0 .. N-1 in the 'i' dimension (rows), 0 .. N-1 in 'j' dimension (cols).
// Each rank handles a local chunk of the rows: [start_i, end_i-1], inclusive.
//
// We'll store (localN + 2) x N for each rank, to include
//  + 1 halo row at the top    (index 0)
//  + localN rows for real data (index 1 .. localN)
//  + 1 halo row at the bottom (index localN+1)
//
// Then interior is [1..localN] in 'i', [0..N-1] in 'j'.
//
// Communication occurs for the top/bottom halo with neighbors.
//
// `C_new[i][j] = ...` is done for i in [1..localN], j in [1..N-1].
//
// Finally, we do an MPI_Allreduce to compute the global difmedio.
//
double mpi_omp_diff_eq(double **C, double **C_new, DiffEqArgs *args,
                       int localN, int N, int rank, int size) {
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;
    double difmedio_local = 0.0;

    // 1) Exchange boundary rows (halo) with neighbors
    //    top neighbor:   rank-1 (if rank > 0)
    //    bottom neighbor: rank+1 (if rank < size-1)
    MPI_Request reqs[4];
    int req_count = 0;

    // Send my top row (row=1) to rank-1, receive into row=0
    if (rank > 0) {
        MPI_Irecv(C[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(C[1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    // Send my bottom row (row=localN) to rank+1, receive into row=localN+1
    if (rank < size - 1) {
        MPI_Irecv(C[localN + 1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(C[localN], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    // Wait for boundary exchange to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // 2) Perform the diffusion step locally, using OpenMP for parallelization
    //    Skip the halo layers -> i in [1..localN], j in [1..N-1]
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

    // 3) Compute global average difference (difmedio)
    //    We have local_sum of differences in our subdomain -> reduce to a global sum
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // The total number of "active" interior cells in the entire domain is (N-2)*(N-2).
    // (assuming we're ignoring boundary i=0, i=N-1, j=0, j=N-1).
    double difmedio_global = global_sum / ((N - 2) * (N - 2));

    return difmedio_global;
}

//-----------------------------------------------------------------------
// Main function (hybrid MPI+OpenMP driver).
// mpirun -np <num_procs> ./mpi_omp_diff_eq <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>
//
// Example run:
//   mpirun -np 4 ./mpi_omp_diff_eq 1000 1000 0.1 0.1 0.1 4
//
// Make sure N is divisible by the number of processes for simple 1D decomposition.
#ifndef BUILD_SHARED
int main(int argc, char *argv[]) {
    struct timeval start, end, start_parallel, end_parallel;
    gettimeofday(&start, NULL);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check arguments
    if (argc != 7) {
        if (rank == 0) {
            printf("Usage: mpirun -np <num_procs> %s <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    double D = atof(argv[3]);
    double DELTA_T = atof(argv[4]);
    double DELTA_X = atof(argv[5]);
    int NUM_THREADS = atoi(argv[6]);

    // Set number of threads for OpenMP
    omp_set_num_threads(NUM_THREADS);

    // For simplicity, assume N is divisible by size (number of processes).
    // localN is the number of "real" rows each process will handle (excluding halo).
    int localN = N / size;

    // Create the local matrices of size (localN+2) x N:
    //   +2 to store top and bottom halos.
    double **C = create_submatrix(localN + 2, N);  // You need to implement or adapt your create_matrix
    double **C_new = create_submatrix(localN + 2, N);

    // Initialize everything to 0.0
    for (int i = 0; i < localN + 2; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    // Set initial condition in the global center (N/2, N/2) if it belongs to this rank.
    // Global center row = mid_row = N/2
    // Our local chunk covers rows [rank*localN .. (rank+1)*localN - 1] in global indexing.
    int global_start = rank * localN;
    int global_end = global_start + localN - 1;

    int mid_row = N / 2;
    int mid_col = N / 2;
    if (mid_row >= global_start && mid_row <= global_end) {
        int local_i = mid_row - global_start + 1;  // +1 because of halo offset
        C[local_i][mid_col] = 1.0;
    }

    // Create struct with arguments
    DiffEqArgs args;
    args.N = N;
    args.D = D;
    args.DELTA_T = DELTA_T;
    args.DELTA_X = DELTA_X;

    // Time-step loop
    gettimeofday(&start_parallel, NULL);
    for (int t = 0; t < T; t++) {
        double difmedio = mpi_omp_diff_eq(C, C_new, &args, localN, N, rank, size);

        // Swap C and C_new
        double **temp = C;
        C = C_new;
        C_new = temp;

        // Print from rank=0 every 100 iterations
        if ((t % 100) == 0 && rank == 0) {
            printf("Iteração %d - diferença média global = %g\n", t, difmedio);
        }
    }
    gettimeofday(&end_parallel, NULL);

    // Print the final concentration at the center from the rank that owns it
    if (mid_row >= global_start && mid_row <= global_end) {
        int local_i = mid_row - global_start + 1;
        printf("Rank %d => Concentração final no centro: %f\n", rank, C[local_i][mid_col]);
    }

    // Optionally gather results or do further analysis here
    // e.g. you could gather the entire matrix with MPI_Gather if needed.

    free_submatrix(C, localN + 2);  // You need to implement or adapt your free_matrix
    free_submatrix(C_new, localN + 2);

    // Timing
    gettimeofday(&end, NULL);
    double total_time =
        ((end.tv_sec * 1000000 + end.tv_usec) -
         (start.tv_sec * 1000000 + start.tv_usec)) /
        1000.0;
    double parallel_time =
        ((end_parallel.tv_sec * 1000000 + end_parallel.tv_usec) -
         (start_parallel.tv_sec * 1000000 + start_parallel.tv_usec)) /
        1000.0;
    double sequential_time = total_time - parallel_time;

    if (rank == 0) {
        printf("Tempo total:     %lf ms\n", total_time);
        printf("Tempo paralelo:  %lf ms\n", parallel_time);
        printf("Tempo sequencial:%lf ms\n", sequential_time);
    }

    MPI_Finalize();
    return 0;
}
#endif
