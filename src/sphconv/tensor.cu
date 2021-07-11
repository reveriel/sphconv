#include "sphconv/debug_utils.h"
#include "timer.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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
    const GpuTensor<IType, 2> indicesBZYX, // [NNZ, 4], 4: b,z,y,x
    GpuTensor<IType, 4> grid)              // [B, H, W, D]
{
    int NNZ = indicesBZYX.size(0);
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n >= NNZ)
        return;

    IType b = indicesBZYX[n][0];
    IType x = indicesBZYX[n][1]; // this is convention disagreement..  not a bug
    IType y = indicesBZYX[n][2]; // this is convention disagreement..  not a bug
    IType z = indicesBZYX[n][3]; // this is convention disagreement..  not a bug

    grid[b][x][y][z] = IType(1);
}

template <typename IType, typename DType>
__global__ void reorderFeatureKernel(
    const GpuTensor<IType, 3> zPtr,        // [B, H, W]
    const GpuTensor<IType, 2> indicesBZYX, // [NNZ, 4], bzyx
    const GpuTensor<DType, 2> feature,     // [NNZ ,C]
    GpuTensor<IType, 3> fiberSize,         // [B, H, W]
    GpuTensor<IType, 1> zIndices,          // [NNZ]
    GpuTensor<DType, 2> outFeature,        // [NNZ, C]
    GpuTensor<IType, 1> permutation)       // [NNZ]
{
    int NNZ = indicesBZYX.size(0);
    int C = feature.size(1);
    int n = threadIdx.y + blockDim.y * blockIdx.x;
    if (n >= NNZ)
        return;

    IType b = indicesBZYX[n][0];
    IType x = indicesBZYX[n][1]; // this is convention disagreement..  not a bug
    IType y = indicesBZYX[n][2]; // this is convention disagreement..  not a bug
    IType z = indicesBZYX[n][3]; // this is convention disagreement..  not a bug

    IType val_pos = zPtr[b][x][y];
    IType fiber_pos;
    if (threadIdx.x == 0) {
        fiber_pos = atomicAdd(&fiberSize[b][x][y], -1);
    }
    fiber_pos = __shfl_sync(0xffffffff, fiber_pos, 0, blockDim.x);
    int new_pos = val_pos - fiber_pos;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        outFeature[new_pos][c] = feature[n][c];
    }

    // record the permutation, for backward
    if (threadIdx.x == 0) {
        permutation[n] = new_pos;
        zIndices[new_pos] = z;
    }
}

std::vector<torch::Tensor>
init_tensor(
    const torch::Tensor feature,     // [NNZ, C]
    const torch::Tensor indicesBZYX, // [NNZ, 4]
    int batchSize,
    std::vector<int64_t> spatialShape) // H, W, D
{
    CHECK_INPUT(feature);
    CHECK_INPUT(indicesBZYX);
    int NNZ = feature.size(0);
    int C = feature.size(1);

    torch::Tensor outFeature = torch::empty(
        {NNZ, C}, torch::dtype(feature.dtype()).device(feature.device()));
    torch::Tensor zIndices = torch::zeros(
        {NNZ}, torch::dtype(indicesBZYX.dtype()).device(indicesBZYX.device()));
    torch::Tensor grid = torch::zeros(
        {batchSize, spatialShape[0], spatialShape[1], spatialShape[2]},
        torch::dtype(torch::kInt32).device(indicesBZYX.device()));
    // fill grid
    // set occupied voxels to 1
    fillGridKernel<int32_t><<<divUp(NNZ, 64), 64>>>(
        indicesBZYX.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // sum on each D fiber
    torch::Tensor fiberSize = torch::sum(grid, 3, false, torch::kInt32);
    torch::Tensor zPtr = torch::cumsum(fiberSize.view({-1}), 0, torch::kInt32)
                             .view({batchSize, spatialShape[0], spatialShape[1]});

    torch::Tensor permutation = torch::empty(
        {NNZ}, torch::dtype(torch::kInt32).device(feature.device()));

    reorderFeatureKernel<int32_t, float><<<divUp(NNZ, 64), dim3(8, 64)>>>(
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        indicesBZYX.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        fiberSize.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        permutation.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {outFeature, zIndices, zPtr, permutation};
}

template <typename IType, typename DType>
__global__ void reorderFeatureBackwardKernel(
    const GpuTensor<DType, 2> d_featureOut, // [NNZ, C]
    const GpuTensor<IType, 1> permutation,  // [NNZ]
    GpuTensor<DType, 2> d_feature)          // [NNZ, C]
{
    int NNZ = d_feature.size(0);
    int C = d_feature.size(1);
    int n = threadIdx.y + blockDim.y * blockIdx.x;
    if (n >= NNZ)
        return;
    int new_pos = permutation[n];

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        d_feature[n][c] = d_featureOut[new_pos][c];
    }
}

torch::Tensor init_tensor_backward(
    const torch::Tensor d_featureOut, // [NNZ, C]
    const torch::Tensor permutation)  // [NNZ]
{
    int NNZ = d_featureOut.size(0);
    int C = d_featureOut.size(1);
    torch::Tensor d_feature = torch::empty(
        {NNZ, C}, torch::dtype(d_featureOut.dtype()).device(d_featureOut.device()));

    reorderFeatureBackwardKernel<int32_t, float><<<divUp(NNZ, 64), dim3(8, 64)>>>(
        d_featureOut.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        permutation.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        d_feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    return d_feature;
}

template <typename IType, typename DType>
__global__ void toDenseKernel(
    const GpuTensor<DType, 2> feature,  // [NNZ ,C]
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    GpuTensor<DType, 5> out)            // [B, H, W, D, C]
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = feature.size(1);
    int x = threadIdx.y + blockDim.y * blockIdx.x;
    int y = threadIdx.z + blockDim.z * blockIdx.y;
    if (x >= H || y >= W)
        return;

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];
        for (int pos = zStart; pos < zEnd; pos++) {
            IType z = zIndices[pos];
            // grid[b][x][y][z] = pos;
            for (int c = threadIdx.x; c < C; c += blockDim.x) {
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
    GpuTensor<DType, 2> d_feature) // [NNZ, C]
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = d_feature.size(1);
    int x = threadIdx.y + blockDim.y * blockIdx.x;
    int y = threadIdx.z + blockDim.z * blockIdx.y;
    if (x >= H || y >= W)
        return;

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];
        for (int pos = zStart; pos < zEnd; pos++) {
            IType z = zIndices[pos];
            // grid[b][x][y][z] = pos;
            for (int c = threadIdx.x; c < C; c += blockDim.x) {
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

    int H_BLOCK = 2;
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

    dim3 gridSize = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
    dim3 blockSize = dim3(C_BLOCK, H_BLOCK, W_BLOCK);

    toDenseKernel<int32_t, float><<<gridSize, blockSize>>>(
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return out;
}

torch::Tensor to_dense_backward(
    const torch::Tensor d_featureOut, // [B D W H C]
    const torch::Tensor zIndices,     // [NNZ]
    const torch::Tensor zPtr)         // [B, H, W]
{
    CHECK_INPUT(d_featureOut);
    CHECK_INPUT(zIndices);
    CHECK_INPUT(zPtr);

    int H_BLOCK = 2;
    int W_BLOCK = 16;
    int C_BLOCK = 16;

    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int C = d_featureOut.size(4);
    int NNZ = zIndices.size(0);

    torch::Tensor d_feature = torch::zeros(
        {NNZ, C}, torch::dtype(torch::kFloat32).device(d_featureOut.device()));

    if (C <= 4) {
        C_BLOCK = 4;
    } else if (C <= 8) {
        C_BLOCK = 8;
    } else {
        C_BLOCK = 16;
    }

    dim3 gridSize = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
    dim3 blockSize = dim3(C_BLOCK, H_BLOCK, W_BLOCK);

    toDenseBackwardKernel<int32_t, float><<<gridSize, blockSize>>>(
        d_featureOut.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        d_feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return d_feature;
}

} // namespace sphconv
