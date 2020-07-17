#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

template <typename T> __global__ void prefix_sum(T *A, int len) {
  for (int i = 1; i < len; i++) {
    A[i] += A[i - 1];
  }

  int oX = threadIdx.x + blockDim.x * blockIdx.x;
  int oY = threadIdx.y + blockDim.y * blockIdx.y;

  printf("Thread(%d,%d,%d,%d,%d,%d), oXY(%d,%d)\n", threadIdx.x, threadIdx.y,
         threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, oX, oY);
}

int main() {

  size_t sz;
  cudaDeviceGetLimit(&sz, cudaLimitPrintfFifoSize);
  std::cout << sz << std::endl;
  sz = 1048576 * 10;
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);
  int len = 128;

  int32_t *A_h = (int32_t *)malloc(sizeof(int32_t) * len);
  memset(A_h, 0, sizeof(int32_t) * len);
  A_h[0] = 1;
  int32_t *A_d;
  cudaMalloc(&A_d, sizeof(int32_t) * len);
  cudaMemcpy(A_d, A_h, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

  prefix_sum<int32_t><<<dim3(8, 4, 1), dim3(8, 32, 1)>>>(A_d, len);

  cudaMemcpy(A_h, A_d, sizeof(int32_t) * len, cudaMemcpyDeviceToHost);

  printf("======= result ==========\n");
  for (int i = 0; i < len; i++) {
    printf("A[%d] = %d\n", i, A_h[i]);
  }

  return 0;
}
