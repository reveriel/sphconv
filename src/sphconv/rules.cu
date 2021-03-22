
#include <debug_utils.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "indice.cu.h"
#include "timer.h"

using namespace std;

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

/**
 * @brief init the grid[B, H, W, D],
 *  grid is a mapping from spatial location to its target glaobal physical location
 *
 * fill grid with global indices
 *
 * TODO: fill local indices
 *
 * @return __global__
 */
template <typename Index>
__global__ void prepareSubMGridKernel(
    const GpuTensor<Index, 1> zIndices,
    const GpuTensor<Index, 3> zPtr,
    GpuTensor<Index, 4> grid,
    int B, int H, int W)
{
    Index x = threadIdx.x + blockDim.x * blockIdx.x;
    Index y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= H || y >= W)
        return;

    // for each voxel
    for (int b = 0; b < B; b++)
    {
        int zEnd = zPtr[b][x][y];
        int zStart = (x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        for (int zi = zStart; zi < zEnd; zi++)
        {
            Index z = zIndices[zi];
            grid[b][x][y][z] = zi;
        }
    }
}


/***
 *  fill rules,
        rules: [NTile, K*K*K, 4, DMax]
 */
template <typename Index>
__global__ void getSubMRulesKernel(
    const GpuTensor<Index, 1> zIndices,
    const GpuTensor<Index, 3> zPtr,
    const GpuTensor<Index, 4> grid,
    GpuTensor<Index, 4> rules,
    GpuTensor<Index, 2> indiceNum,  // number active index, [NTile, KKK]
    int B,
    int H, int W,
    int oD, int oH, int oW,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padD, int padH, int padW,
    int dD,  int dH, int dW)
{
    Index x = threadIdx.x + blockDim.x * blockIdx.x;
    Index y = threadIdx.y + blockDim.y * blockIdx.y;
    Index k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    Index k_D = k / (KH * KW);
    Index k_H = (k / KW) % KH;
    Index k_W = k % KW;

    Index oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    Index oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    Index nTile = 0; // TODO

    for (int b = 0; b < B;  b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        for (int zi = zStart; zi < zEnd; zi++)
        {
            Index z = zIndices[zi];
            Index oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                break;

            Index counter = atomicAdd(&indiceNum[nTile][k], Index(1));

            // grid[b][x][y][z] = zi;
            // rules: [NTile, K*K*K, 4, DMax]
            rules[nTile][k][0][counter] = zi;
            rules[nTile][k][1][counter] = grid[b][oX][oY][oZ];
        }
    }
}



/**
 *  tile_size: tile_size is on the output feature map.
 *
 *
 * return rules
 *  rules: [NTile, K*K*K, 4, DMax]
 *
 * ROADMAP:
 * 1. only generate global indices
 * 2. generate both local indices and global indices
 *
 * ref:   getIndicePair
 */
std::vector<torch::Tensor>
get_rules_subm(torch::Tensor zIndices,               //  [NNZ]
                torch::Tensor zPtr,                   // [B, H, W]
                torch::Tensor grid,                   // [B, H, W, D]
                int batchSize,
                std::vector<int64_t> spatialShape,    // H, W, D
                std::vector<int64_t> outSpatialShape, // H, W, D
                std::vector<int64_t> kernelSize,
                std::vector<int64_t> stride,
                std::vector<int64_t> padding,
                std::vector<int64_t> dilation)
{
    grid.fill_(-1);

    const int H_BLOCK = 4;
    const int W_BLOCK = 16;
    int num_active = zIndices.size(0);

    dim3 gridSize = dim3(divUp(spatialShape[0], H_BLOCK), divUp(spatialShape[1], W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, 1);

    prepareSubMGridKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        batchSize,
        spatialShape[0], spatialShape[1]);

    printTensor<int>(grid, "grid", 0, 0, outSpatialShape[0], 0, outSpatialShape[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());
    blockSize = dim3(H_BLOCK, W_BLOCK, kernelVolume);

    int NTile = 1; // TODO, number of Tiles
    // allocate rules and indice Num
    torch::Tensor rules =
        torch::full({NTile, kernelVolume, 2, zIndices.size(0)}, -1, torch::dtype(torch::kInt32).device(zIndices.device()));
    // rules is allocated larger, to be trimed lalter
    // TODO, change 2 to 4, numAct
    // TODO, last dimension...

    torch::Tensor indiceNum =
        torch::zeros({NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));

    getSubMRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        indiceNum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        batchSize,
        spatialShape[0], spatialShape[1],
        outSpatialShape[0], outSpatialShape[1], outSpatialShape[2],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // outZPtr
    // outIndices ?

    return {zIndices, zPtr, rules, indiceNum};
}

} // namespace sphconv
