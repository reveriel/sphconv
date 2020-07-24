#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>


#ifdef DEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
#define printTensor_int(...) printTensor<int>(...)
#define printTensor_k_int(...) printTensor_k<int>(...)
#else
#define gpuErrchk(ans)
#define printTensor_int(...)
#define printTensor_k_int(...)
#endif


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define isThread(thread_x, thread_y, thread_z, block_x, block_y, block_z) \
  (threadIdx.x  == thread_x && threadIdx.y == thread_y && threadIdx.z == thread_z \
  && blockIdx.x == block_x && blockIdx.y == block_y && blockIdx.z == block_z)

int divUp(int x, int y) { return (x + y - 1) / y; };


template <typename Index>
__device__ __inline__ Index OutSpatial(Index k, Index x, Index s, Index d, Index pad)
{
  // forgive me. do nothing with the dillation
  // TODO
  if ((x + pad - k) % s == 0)
    return (x + pad - k)/ s;
  return -1;
}


// Print a some tile, for debugging.
template <typename T>
void printTensor(at::Tensor A, std::string name, int batch_idx, int H_start,
                 int H_end, int W_start, int W_end) {

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== BEGIN" << std::endl;

  int dim = A.dim();
  if (dim == 3) {

    for (int x = H_start; x < H_end; x++) {
      std::cout << "  [ ";
      for (int y = W_start; y < W_end - 1; y++) {
        std::cout << A[batch_idx][x][y].item<T>() << ", ";
      }
      std::cout << A[batch_idx][x][W_end - 1].item<T>() << " ]" << std::endl;
    }
  } else if (dim == 4) {
    int size1 = A.size(1);
    for (int i = 0; i < size1; i++) {
      std::cout << " [ == " << i << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][i][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][i][x][W_end - 1].item<T>() << " ]" << std::endl;
      }
      std::cout << " ] ==" << i << std::endl;
    }
  } else if (dim == 5) {
    // we only print dim == 4 tensor, print channel 0
    int size1 = A.size(1);
    for (int i = 0; i < size1; i++) {
      std::cout << " [ ==" << i << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][0][i][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][0][i][x][W_end - 1].item<T>() << " ]";
      }
      std::cout << " ] ==" << i << std::endl;
    }
  }

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== END" << std::endl;
}

// Print a some tile, for debugging.
// the second dimension is kernel
template <typename T>
void printTensor_k(at::Tensor A, std::string name, int batch_idx, int H_start,
                 int H_end, int W_start, int W_end) {

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== BEGIN" << std::endl;

  int dim = A.dim();
  if (dim == 4) {
    int kernel_volume = A.size(1);
    for (int k = 0; k < kernel_volume; k++) {
      std::cout << " [ == " << k << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][k][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][k][x][W_end - 1].item<T>() << " ]" << std::endl;
      }
      std::cout << " ] ==" << k << std::endl;
    }
  } else if (dim == 5) {
    // we only print dim == 4 tensor, print channel 0
    int kernel_volume = A.size(1);
    int len = A.size(4);
    for (int k = 0; k < kernel_volume; k++) {
      std::cout << " [ ==" << k << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end ; y++) {
          std::cout << "(";
          for (int i = 0; i < A.size(4); i++ ) {
            std::cout << A[batch_idx][k][x][y][i].item<T>() << " ";
          }
          std::cout << "),";
        }
        std::cout <<  " ]";
      }
      std::cout << " ] ==" << k << std::endl;
    }
  }

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== END" << std::endl;
}