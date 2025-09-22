#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 8;
    int h_a[N] = {1,2,3,4,5,6,7,8};
    int h_b[N] = {8,7,6,5,4,3,2,1};
    int h_c[N];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 4;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
