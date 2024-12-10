#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "diff_eq.h"
#include "utils.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t errr = (err); \
        if (errr != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
            cudaGetErrorString(errr), errr, __FILE__, __LINE__); \
            exit(errr); \
        } \
    } while (0)

// Kernels as before...
__global__ void compute_kernel(double *C, double *C_new, int N, double D, double DELTA_T, double DELTA_X) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        double val = C[i*N + j] + D * DELTA_T * 
            ((C[(i+1)*N + j] + C[(i-1)*N + j] + C[i*N + (j+1)] + C[i*N + (j-1)] - 4.0 * C[i*N + j]) / (DELTA_X * DELTA_X));
        C_new[i*N + j] = val;
    }
}

__global__ void diff_kernel(double *C, double *C_new, double *block_sums, int N) {
    extern __shared__ double sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x; 
    int block_size = blockDim.x * blockDim.y;

    double diff_val = 0.0;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        double old_val = C[i*N + j];
        double new_val = C_new[i*N + j];
        diff_val = fabs(new_val - old_val);
        C[i*N + j] = new_val;
    }

    sdata[thread_id] = diff_val;
    __syncthreads();

    // reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        int grid_width = gridDim.x;
        int block_index = blockIdx.y * grid_width + blockIdx.x;
        block_sums[block_index] = sdata[0];
    }
}

double cuda_diff_eq(double **C_host, double **C_new_host, DiffEqArgs *args, int T) {
    int N = args->N;
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;

    size_t size = N * N * sizeof(double);

    // Flatten host arrays once
    double *C_flat = (double*)malloc(size);
    double *C_new_flat = (double*)malloc(size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_flat[i*N + j] = C_host[i][j];
            C_new_flat[i*N + j] = C_new_host[i][j];
        }
    }

    // Allocate device memory once
    double *d_C, *d_C_new;
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C_new, size));
    CUDA_CHECK(cudaMemcpy(d_C, C_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_new, C_new_flat, size, cudaMemcpyHostToDevice));

    dim3 blockDim(32,32);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);

    int num_blocks = gridDim.x * gridDim.y;
    double *d_block_sums;
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(double)));

    // Allocate once on host for block sums
    double *h_block_sums = (double*)malloc(num_blocks * sizeof(double));

    size_t smem_size = blockDim.x * blockDim.y * sizeof(double);

    // Main loop
    for (int t = 0; t < T; t++) {
        compute_kernel<<<gridDim, blockDim>>>(d_C, d_C_new, N, D, DELTA_T, DELTA_X);
        CUDA_CHECK(cudaDeviceSynchronize());

        diff_kernel<<<gridDim, blockDim, smem_size>>>(d_C, d_C_new, d_block_sums, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Only copy and compute difmedio every 100 steps to reduce overhead
        if (t % 100 == 0) {
            CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
            double total_diff = 0.0;
            for (int i = 0; i < num_blocks; i++) {
                total_diff += h_block_sums[i];
            }
            double difmedio = total_diff / ((N-2)*(N-2));
            printf("interacao %d - diferenca = %g\n", t, difmedio);
        }
    }

    // Final copy of C back to host
    CUDA_CHECK(cudaMemcpy(C_flat, d_C, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_host[i][j] = C_flat[i*N + j];
        }
    }

    double final_val = C_host[N/2][N/2];

    // Cleanup
    free(C_flat);
    free(C_new_flat);
    free(h_block_sums);
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_new));
    CUDA_CHECK(cudaFree(d_block_sums));

    return final_val;
}

#ifndef BUILD_SHARED
int main(int argc, char *argv[]) {
    struct timeval start, end, start_parallel, end_parallel;
    gettimeofday(&start, NULL);

    if (argc != 7) {
        printf("Usage: %s <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    double D = atof(argv[3]);
    double DELTA_T = atof(argv[4]);
    double DELTA_X = atof(argv[5]);
    int NUM_THREADS = atoi(argv[6]); // Not used in CUDA

    double **C = create_matrix_and_init(N); 
    double **C_new = create_matrix(N);
    C[N/2][N/2] = 1.0;

    DiffEqArgs args = {N, D, DELTA_T, DELTA_X};
    gettimeofday(&start_parallel, NULL);

    double final_val = cuda_diff_eq(C, C_new, &args, T);

    gettimeofday(&end_parallel, NULL);
    printf("Concentração final no centro: %f\n", final_val);

    // salvar_matriz(C, N, N, "matriz_cuda.txt");
    free_matrix(C, N);
    free_matrix(C_new, N);

    gettimeofday(&end, NULL);
    double total_time = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000.0;
    double parallel_time = ((end_parallel.tv_sec * 1000000 + end_parallel.tv_usec) - (start_parallel.tv_sec * 1000000 + start_parallel.tv_usec)) / 1000.0;
    double sequential_time = total_time - parallel_time;

    printf("Tempo total: %lf ms\n", total_time);
    printf("Tempo paralelo (CUDA): %lf ms\n", parallel_time);
    printf("Tempo sequencial: %lf ms\n", sequential_time);

    return 0;
}
#endif
