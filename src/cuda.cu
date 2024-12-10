// diff_eq_optimized.cu

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
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

// Structure to hold diffusion equation parameters
typedef struct {
    int N;
    float D;
    float DELTA_T;
    float DELTA_X;
} DiffEqArgs;

// Utility function to create and initialize a matrix
float** create_matrix_and_init(int N) {
    float **matrix = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (float*)calloc(N, sizeof(float));
    }
    // Initialize center
    matrix[N/2][N/2] = 1.0f;
    return matrix;
}

// Utility function to create a matrix
float** create_matrix(int N) {
    float **matrix = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (float*)calloc(N, sizeof(float));
    }
    return matrix;
}

// Utility function to free a matrix
void free_matrix(float **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Optimized kernel combining compute and reduction
__global__ void compute_and_diff_kernel(const float *C, float *C_new, float *block_sums, int N, float D, float DELTA_T, float DELTA_X) {
    // Define shared memory for reduction
    extern __shared__ float sdata[];

    // Calculate global indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Linear index for 1D representation
    int idx = i * N + j;

    // Initialize difference value
    float diff_val = 0.0f;

    // Perform computation if within bounds
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        float center = C[idx];
        float up = C[(i-1)*N + j];
        float down = C[(i+1)*N + j];
        float left = C[i*N + (j-1)];
        float right = C[i*N + (j+1)];

        // Compute new value using diffusion equation
        float new_val = center + D * DELTA_T * ((up + down + left + right - 4.0f * center) / (DELTA_X * DELTA_X));
        C_new[idx] = new_val;

        // Compute absolute difference
        diff_val = fabsf(new_val - center);
    }

    // Load difference value into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = diff_val;
    __syncthreads();

    // Perform reduction in shared memory
    // Reduce within a block
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        block_sums[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}

// Host function to perform optimized diffusion equation computation
float cuda_diff_eq_optimized(float **C_host, float **C_new_host, DiffEqArgs *args, int T) {
    int N = args->N;
    float D = args->D;
    float DELTA_T = args->DELTA_T;
    float DELTA_X = args->DELTA_X;

    size_t size = N * N * sizeof(float);

    // Flatten host arrays
    float *C_flat = (float*)malloc(size);
    float *C_new_flat = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_flat[i*N + j] = C_host[i][j];
            C_new_flat[i*N + j] = C_new_host[i][j];
        }
    }

    // Allocate device memory
    float *d_C, *d_C_new;
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C_new, size));
    CUDA_CHECK(cudaMemcpy(d_C, C_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_new, C_new_flat, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // Reduced block size for better occupancy
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);

    int num_blocks = gridDim.x * gridDim.y;
    float *d_block_sums;
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float)));

    // Allocate host memory for block sums
    float *h_block_sums = (float*)malloc(num_blocks * sizeof(float));

    size_t smem_size = blockDim.x * blockDim.y * sizeof(float);

    // Create CUDA stream for overlapping (optional)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize pointers for pointer swapping
    float *current = d_C;
    float *next = d_C_new;

    // Main loop
    for (int t = 0; t < T; t++) {
        // Launch combined compute and reduction kernel
        compute_and_diff_kernel<<<gridDim, blockDim, smem_size, stream>>>(current, next, d_block_sums, N, D, DELTA_T, DELTA_X);
        CUDA_CHECK(cudaGetLastError());

        // Every 100 steps, copy block sums to host and compute difmedio
        if (t % 100 == 0) {
            // Synchronize to ensure kernel has finished
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Copy block sums to host
            CUDA_CHECK(cudaMemcpyAsync(h_block_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Compute total difference on host
            float total_diff = 0.0f;
            for (int i = 0; i < num_blocks; i++) {
                total_diff += h_block_sums[i];
            }
            float difmedio = total_diff / ((N-2)*(N-2));
            printf("Iteration %d - Difference = %g\n", t, difmedio);
        }

        // Swap pointers for next iteration
        float *temp = current;
        current = next;
        next = temp;
    }

    // Ensure all kernels have finished
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy final data back to host
    CUDA_CHECK(cudaMemcpy(C_flat, current, size, cudaMemcpyDeviceToHost));

    // Reconstruct host matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_host[i][j] = C_flat[i*N + j];
        }
    }

    float final_val = C_host[N/2][N/2];

    // Cleanup
    free(C_flat);
    free(C_new_flat);
    free(h_block_sums);
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_new));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return final_val;
}

int main(int argc, char *argv[]) {
    struct timeval start, end, start_parallel, end_parallel;
    gettimeofday(&start, NULL);

    if (argc != 7) {
        printf("Usage: %s <N> <T> <D> <DELTA_T> <DELTA_X> <NUM_THREADS>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    float D = atof(argv[3]);
    float DELTA_T = atof(argv[4]);
    float DELTA_X = atof(argv[5]);
    int NUM_THREADS = atoi(argv[6]); // Not used in CUDA

    // Initialize matrices
    float **C = create_matrix_and_init(N);
    float **C_new = create_matrix(N);

    DiffEqArgs args = {N, D, DELTA_T, DELTA_X};
    gettimeofday(&start_parallel, NULL);

    // Perform diffusion computation
    float final_val = cuda_diff_eq_optimized(C, C_new, &args, T);

    gettimeofday(&end_parallel, NULL);
    printf("Final concentration at center: %f\n", final_val);

    // Optionally save the matrix to a file
    // Implement salvar_matriz if needed

    free_matrix(C, N);
    free_matrix(C_new, N);

    gettimeofday(&end, NULL);
    double total_time = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000.0;
    double parallel_time = ((end_parallel.tv_sec * 1000000 + end_parallel.tv_usec) - (start_parallel.tv_sec * 1000000 + start_parallel.tv_usec)) / 1000.0;
    double sequential_time = total_time - parallel_time;

    printf("Total Time: %lf ms\n", total_time);
    printf("Parallel Time (CUDA): %lf ms\n", parallel_time);
    printf("Sequential Time: %lf ms\n", sequential_time);

    return 0;
}
