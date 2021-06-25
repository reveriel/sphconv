#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#include "debug_utils.h"
#include "timer.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

/**
 * @brief fill grid = 1 at non empty voxels
 *
 * @tparam IType
 */
template <typename IType>
__global__ void fillGridKernel(
    const GpuTensor<IType, 2> indicesZYX, // [NNZ, 4], 4: b,z,y,x
    GpuTensor<IType, 4> grid)             // [B, H, W, D]
{
    int NNZ = indicesZYX.size(0);
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n >= NNZ)
        return;

    IType b = indicesZYX[n][0];
    IType x = indicesZYX[n][1]; // this is convention disagreement..  not a bug
    IType y = indicesZYX[n][2]; // this is convention disagreement..  not a bug
    IType z = indicesZYX[n][3]; // this is convention disagreement..  not a bug

    grid[b][x][y][z] = IType(1);
}

template <typename IType, typename DType>
__global__ void reorderFeatureKernel(
    const GpuTensor<IType, 3> zPtr,       // [B, H, W]
    const GpuTensor<IType, 2> indicesZYX, // [NNZ, 4], bzyx
    const GpuTensor<DType, 2> feature,    // [NNZ ,C]
    GpuTensor<IType, 3> fiberSize,        // [B, H, W]
    GpuTensor<IType, 1> zIndices,         // [NNZ]
    GpuTensor<DType, 2> outFeature)       // [NNZ, C]
{
    int NNZ = indicesZYX.size(0);
    int C = feature.size(1);
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n >= NNZ)
        return;

    IType b = indicesZYX[n][0];
    IType x = indicesZYX[n][1]; // this is convention disagreement..  not a bug
    IType y = indicesZYX[n][2]; // this is convention disagreement..  not a bug
    IType z = indicesZYX[n][3]; // this is convention disagreement..  not a bug

    IType val_pos = zPtr[b][x][y];
    IType fiber_pos = atomicAdd(&fiberSize[b][x][y], -1);

    for (int c = 0; c < C; c++) {
        outFeature[val_pos - fiber_pos][c] = feature[n][c];
    }

    zIndices[val_pos - fiber_pos] = z;
    // fiberSize[b][x][y] = fiber_pos - 1;
}

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor
init_tensor(
    const torch::Tensor feature,    // [NNZ, C]
    const torch::Tensor indicesZYX, // [NNZ, 4]
    int batchSize,
    std::vector<int64_t> spatialShape, // H, W, D
    torch::Tensor outFeature,          // [NNZ, C]
    torch::Tensor zIndices)            //
{

    CHECK_INPUT(feature);
    CHECK_INPUT(indicesZYX);
    CHECK_INPUT(outFeature);
    CHECK_INPUT(zIndices);

    int NNZ = feature.size(0);
    torch::Tensor grid = torch::zeros({batchSize, spatialShape[0], spatialShape[1], spatialShape[2]},
                                      torch::dtype(torch::kInt32).device(indicesZYX.device()));
    // fill grid
    // set occupied voxels to 1
    fillGridKernel<int32_t><<<divUp(NNZ, 512), 512>>>(
        indicesZYX.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // sum on each D fiber
    torch::Tensor fiberSize = torch::sum(grid, 3, false, torch::kInt32);
    torch::Tensor zPtr = torch::cumsum(fiberSize.view({-1}), 0, torch::kInt32)
                             .view({batchSize, spatialShape[0], spatialShape[1]});

    reorderFeatureKernel<int32_t, float><<<divUp(NNZ, 512), 512>>>(
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        indicesZYX.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        fiberSize.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return zPtr;
}

template <typename IType, typename DType>
__global__ void toDenseKernel(
    const GpuTensor<DType, 2> feature,  // [NNZ ,C]
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    GpuTensor<DType, 5> out,            // [B, H, W, D, C]
    int C_BLOCK)
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = feature.size(1);
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= H || y >= W)
        return;

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);
        for (int pos = zStart; pos < zEnd; pos++) {
            IType z = zIndices[pos];
            // grid[b][x][y][z] = pos;
            for (int c = threadIdx.z; c < C; c += C_BLOCK) {
                out[b][x][y][z][c] = feature[pos][c];
            }
        }
    }
}

template <typename IType, typename DType>
__global__ void toDenseBackwardKernel(
    const GpuTensor<DType, 5> d_featureOut, // [B H W D C]
    const GpuTensor<IType, 3> zPtr,
    const GpuTensor<IType, 1> zIndices,
    GpuTensor<DType, 2> d_feature, // [NNZ, C]
    int C_BLOCK)
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = d_feature.size(1);
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= H || y >= W)
        return;

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);
        for (int pos = zStart; pos < zEnd; pos++) {
            IType z = zIndices[pos];
            // grid[b][x][y][z] = pos;
            for (int c = threadIdx.z; c < C; c += C_BLOCK) {
                d_feature[pos][c] = d_featureOut[b][x][y][z][c];
            }
        }
    }
}

torch::Tensor to_dense(
    const torch::Tensor feature,  // [NNZ, C]
    const torch::Tensor zIndices, // [NNZ]
    const torch::Tensor zPtr,     // [B, H, W]
    int D,
    torch::Tensor out) // [B, H, W, D, C]
{
    CHECK_INPUT(feature);
    CHECK_INPUT(zIndices);
    CHECK_INPUT(zPtr);
    CHECK_INPUT(out);

    int H_BLOCK = 1;
    int W_BLOCK = 16;
    int C_BLOCK = 16;

    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = feature.size(1);

    if (C <= 4) {
        C_BLOCK = 4;
    } else if (C <= 8) {
        C_BLOCK = 8;
    } else {
        C_BLOCK = 16;
    }

    W_BLOCK = 512 / C_BLOCK;

    dim3 gridSize = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

    toDenseKernel<int32_t, float><<<gridSize, blockSize>>>(
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        C_BLOCK);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return out;
}

torch::Tensor to_dense_backward(
    const torch::Tensor d_featureOut, // [B D W H C]
    const torch::Tensor zIndices,     // [NNZ]
    const torch::Tensor zPtr          // [B, H, W]
)
{
    CHECK_INPUT(d_featureOut);
    CHECK_INPUT(zIndices);
    CHECK_INPUT(zPtr);

    int H_BLOCK = 1;
    int W_BLOCK = 16;
    int C_BLOCK = 16;

    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = d_featureOut.size(4);
    int NNZ = zIndices.size(0);

    torch::Tensor d_feature =
        torch::zeros({NNZ, C}, torch::dtype(torch::kFloat32).device(d_featureOut.device()));

    if (C <= 4) {
        C_BLOCK = 4;
    } else if (C <= 8) {
        C_BLOCK = 8;
    } else {
        C_BLOCK = 16;
    }

    W_BLOCK = 512 / C_BLOCK;

    dim3 gridSize = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

    toDenseBackwardKernel<int32_t, float><<<gridSize, blockSize>>>(
        d_featureOut.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        d_feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        C_BLOCK);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return d_feature;
}

} // namespace sphconv
