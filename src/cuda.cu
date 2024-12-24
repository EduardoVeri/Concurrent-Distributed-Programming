// cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "diff_eq.h"


#define CUDA_CHECK(err) \
    do { \
        cudaError_t errr = (err); \
        if (errr != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
            cudaGetErrorString(errr), errr, __FILE__, __LINE__); \
            exit(errr); \
        } \
    } while (0)

// Device pointers (module-level variables)
double *d_C = nullptr, *d_C_new = nullptr, *d_block_sums = nullptr;
double *h_block_sums = nullptr;
int num_blocks = 0;
cudaStream_t stream;
bool initialized = false;

// Define block and grid dimensions as global variables
#define blockDimX 16
#define blockDimY 16

__global__ void diffusion_kernel(double *C, double *C_new, double *block_sums, int N, double D, double DELTA_T, double DELTA_X) {
    extern __shared__ double sdata[]; // Shared memory for diff_val
    
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    double diff_val = 0.0f;

    if (i < N - 1 && j < N - 1) {
        int idx = i * N + j;
        double up = C[(i - 1) * N + j];
        double down = C[(i + 1) * N + j];
        double left = C[i * N + (j - 1)];
        double right = C[i * N + (j + 1)];
        double center = C[idx];
        
        C_new[idx] = center + D * DELTA_T * ((up + down + left + right - 4 * center) / (DELTA_X * DELTA_X));
        
        diff_val = fabs(C_new[idx] - center);
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = diff_val;
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile double* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        block_sums[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}

extern "C" {

void cuda_init(double *h_C_flat, double *h_C_new_flat, DiffEqArgs *args) {
    int N = args->N;
    size_t size = N * N * sizeof(double);
    num_blocks = ((N - 2 + blockDimX - 1) / blockDimX) * ((N - 2 + blockDimY - 1) / blockDimY);
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C_new, size));
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_C, h_C_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_new, h_C_new_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreate(&stream));

    h_block_sums = (double*)malloc(num_blocks * sizeof(double));
    if (h_block_sums == nullptr) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        exit(1);
    }

    initialized = true;

}

double cuda_diff_eq(DiffEqArgs *args) {
    int N = args->N;
    double D = args->D;
    double DELTA_T = args->DELTA_T;
    double DELTA_X = args->DELTA_X;

    // Check if the device pointers are initialized
    if (!initialized) {
        fprintf(stderr, "Error: Device pointers are not initialized\n");
        exit(1);
    }

    // Define grid dimensions
    dim3 blockDim(blockDimX, blockDimY); 
    dim3 gridDim((N + blockDim.x - 2) / blockDim.x, (N + blockDim.y - 2) / blockDim.y);

    size_t smem_size = blockDim.x * blockDim.y * sizeof(double);

    // Launch the kernel
    diffusion_kernel<<<gridDim, blockDim, smem_size, stream>>>(d_C, d_C_new, d_block_sums, N, D, DELTA_T, DELTA_X);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_block_sums, d_block_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total_diff = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_diff += h_block_sums[i];
    }
    double difmedio = total_diff / ((N-2)*(N-2));

    // Swap device pointers
    double *temp = d_C;
    d_C = d_C_new;
    d_C_new = temp;

    return difmedio;
}

void cuda_get_result(double *h_C_flat, int N) {
    size_t size = N * N * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_C_flat, d_C, size, cudaMemcpyDeviceToHost));
}

void cuda_finalize() {
    if (d_C) {
        CUDA_CHECK(cudaFree(d_C));
        d_C = nullptr;
    }
    if (d_C_new) {
        CUDA_CHECK(cudaFree(d_C_new));
        d_C_new = nullptr;
    }
    if (d_block_sums) {
        CUDA_CHECK(cudaFree(d_block_sums));
        d_block_sums = nullptr;
    }
    if (h_block_sums) {
        free(h_block_sums);
        h_block_sums = nullptr;
    }
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
    }
    cudaDeviceReset(); // Reset the device
    initialized = false;
}

} // extern "C"

#ifndef BUILD_SHARED
int main(int argc, char* argv[]) {
    // Check arguments
    if (argc != 6) {
        printf("Usage: %s <N> <T> <D> <DELTA_T> <DELTA_X>\n", argv[0]);
        return 1;
    }

    // Parse arguments
    int N = atoi(argv[1]);
    int T = atof(argv[2]);
    double D = atof(argv[3]);
    double DELTA_T = atof(argv[4]);
    double DELTA_X = atof(argv[5]);

    size_t size = N * N * sizeof(double);
    double *C_flat = (double*)malloc(size);
    double *C_new_flat = (double*)malloc(size);

    C_flat[N*N/2 + N/2] = 1.0;

    // Initialize device data
    DiffEqArgs args = {N, D, DELTA_T, DELTA_X};
    cuda_init(C_flat, C_new_flat, &args);

    // Main loop
    for (int t = 0; t < T; t++) {
        double difmedio = cuda_diff_eq(&args);
        if (t % 100 == 0)
            printf("Iteration %d - Difference = %g\n", t, difmedio);
    }

    // Copy final data back to host
    cuda_get_result(C_flat, N);

    // Print final at center
    printf("Final value: %g\n", C_flat[N*N/2 + N/2]);

    // Cleanup
    free(C_flat);
    free(C_new_flat);
    cuda_finalize();

    return 0;
}
#endif