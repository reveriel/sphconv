#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

// this is related to tile_size and batchsize, and input data
// #define TILE_N_MAX 128
// #define TILE_N_MAX 2048

static const int zeros[] = {0, 0, 0, 0, 0};

__host__ __device__ __forceinline__ constexpr int divUp(int x, int y)
{
    return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int linearIdx(int x0, int x1, int x2, int x3, int D1, int D2, int D3)
{
    return ((x0 * D1 + x1) * D2 + x2) * D3 + x3;
}

__host__ __device__ __forceinline__ int linearIdx(int x0, int x1, int x2, int D1, int D2)
{
    return (x0 * D1 + x1) * D2 + x2;
}

__host__ __device__ __forceinline__ int linearIdx(int x0, int x1, int D1)
{
    return x0 * D1 + x1;
}

__device__ __forceinline__ int getLinearTileIdx(int TileSize0, int TileSize1, int x, int y, int TileGridW)
{
    int tileIdxX = x / TileSize0;
    int tileIdxY = y / TileSize1;
    return linearIdx(tileIdxX, tileIdxY, TileGridW);
}

// #define DEBUG

#ifdef DEBUG
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#define printTensor_int(...) printTensor<int>(...)
#define printTensor_k_int(...) printTensor_k<int>(...)

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#else
#define gpuErrchk(ans)
#define printTensor_int(...)
#define printTensor_k_int(...)

#define CHECK_CUDA(x)
#define CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x)

#endif

#define ASSERT_RT_ERR(expr, ...)                              \
    {                                                         \
        if (!(expr)) {                                        \
            std::stringstream __macro_s;                      \
            __macro_s << __FILE__ << " " << __LINE__ << "\n"; \
            __macro_s << #expr << " assert faild. ";          \
            tv::sstream_print(__macro_s, __VA_ARGS__);        \
            throw std::runtime_error(__macro_s.str());        \
        }                                                     \
    }

#define gpuUnimplementedErr                          \
    {                                                \
        printf("function %s unimplimented. %s:%d\n", \
               __FUNCTION__, __FILE__, __LINE__);    \
    }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define isThread(thread_x, thread_y, thread_z, block_x, block_y, block_z) \
    (threadIdx.x == thread_x && threadIdx.y == thread_y &&                \
     threadIdx.z == thread_z && blockIdx.x == block_x &&                  \
     blockIdx.y == block_y && blockIdx.z == block_z)

#define PRINT_SHAPE(x) (printShape(x, #x))

inline void printShape(torch::Tensor A, std::string name)
{
    printf("%s shape = (", name.c_str());
    for (int i = 0; i < A.dim() - 1; i++) {
        printf("%ld,", A.size(i));
    }
    printf("%ld)\n", A.size(A.dim() - 1));
}

// Print a some tile, for debugging.
template <typename T>
void printTensor(at::Tensor A, std::string name, int batch_idx, int H_start,
                 int H_end, int W_start, int W_end)
{

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
                std::cout << A[batch_idx][i][x][W_end - 1].item<T>() << " ]"
                          << std::endl;
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
                   int H_end, int W_start, int W_end)
{

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
                std::cout << A[batch_idx][k][x][W_end - 1].item<T>() << " ]"
                          << std::endl;
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
                for (int y = W_start; y < W_end; y++) {
                    std::cout << "(";
                    for (int i = 0; i < A.size(4); i++) {
                        std::cout << A[batch_idx][k][x][y][i].item<T>() << " ";
                    }
                    std::cout << "),";
                }
                std::cout << " ]";
            }
            std::cout << " ] ==" << k << std::endl;
        }
    }

    std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
              << " ======== END" << std::endl;
}