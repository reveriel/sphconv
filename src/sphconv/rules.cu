
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
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        // diverge here, but we assume it's quick
        for (int pos = zStart; pos < zEnd; pos++)
        {
            Index z = zIndices[pos];
            grid[b][x][y][z] = pos;
        }
    }
}

/**
 * @brief init the grid[B, H, W, D],
 *  grid is a mapping from spatial location to its target global physical location
 *
 * fill grid with global indices,
 *
 *  we first fill it with 0,
 * fill output cell to ones
 * sum along  D
 *
 * TODO: fill local indices
 *
 * @return __global__
 */
template <typename Index>
__global__ void prepareGridKernel(
    const GpuTensor<Index, 1> zIndices,
    const GpuTensor<Index, 3> zPtr,
    GpuTensor<Index, 4> grid, // [B, oH, oW, oD]
    int B, int H, int W,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD)
{
    Index oH = grid.size(1);
    Index oW = grid.size(2);
    Index oD = grid.size(3);

    Index x = threadIdx.x + blockDim.x * blockIdx.x;
    Index y = threadIdx.y + blockDim.y * blockIdx.y;
    Index k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    // (KH, KW, KD)
    // k =  kx * KH  ky kz
    Index k_H = k / (KW * KD);
    Index k_W = (k / KD) % KW;
    Index k_D = k % KD;

    Index oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    Index oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    /// for each input voxel, fill its output to 1
    for (int b = 0; b < B; b++)
    {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        // diverge here, but we assume it's quick
        for (int pos = zStart; pos < zEnd; pos++)
        {
            Index z = zIndices[pos];
            Index oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            grid[b][oX][oY][oZ] = pos;
        }
    }
}

/**
 *  create ozIndices
 *   -- recompute all oZ, like we did in prepareGridKernel
 *      is there a way to reuse info in 'grid' ?
 *
 *      YES: but we don't know if it would be faster or not.
 *      we can scan on grid,
 *      TODO: implement both and compare
 */
template <typename Index>
__global__ void getOzIndicesAndRulesKernel(
    const GpuTensor<Index, 1> zIndices, // [NNZ]
    GpuTensor<Index, 1> ozIndices,      // [NNZ']
    const GpuTensor<Index, 3> zPtr,     // [B, H, W]
    GpuTensor<Index, 3> ozPtr,          // [B, oH, oW]
    GpuTensor<Index, 3> fiberSize,      // [B, oH, oW]
    const GpuTensor<Index, 4> grid,
    GpuTensor<Index, 4> rules,    // [NTile, KKK, 4(2), DMax]
    GpuTensor<Index, 2> ruleSize, // number active index, [NTile, KKK]
    int B, int H, int W,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD)
{
    Index oH = grid.size(1);
    Index oW = grid.size(2);
    Index oD = grid.size(3);

    Index x = threadIdx.x + blockDim.x * blockIdx.x;
    Index y = threadIdx.y + blockDim.y * blockIdx.y;
    Index k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    // (KH, KW, KD)
    // k =  kx * KH  ky kz
    Index k_H = k / (KW * KD);
    Index k_W = (k / KD) % KW;
    Index k_D = k % KD;

    Index oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    Index oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    Index nTile = 0; // TODO

    for (int b = 0; b < B; b++)
    {

        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        for (int pos = zStart; pos < zEnd; pos++)
        {
            Index z = zIndices[pos];
            Index oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            Index global_out_idx = grid[b][oX][oY][oZ];

            Index counter = atomicAdd(&ruleSize[nTile][k], Index(1));

            rules[nTile][k][0][counter] = pos;
            rules[nTile][k][1][counter] = global_out_idx;

            Index pos = ozPtr[b][oX][oY];
            Index fiberIdx = fiberSize[b][oX][oY];
            ozIndices[pos - fiberIdx] = oZ;
            fiberSize[b][oX][oY] = fiberIdx - 1;
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
    GpuTensor<Index, 2> ruleSize, // number active index, [NTile, KKK]
    int B, int H, int W,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD)
{
    Index oH = grid.size(1);
    Index oW = grid.size(2);
    Index oD = grid.size(3);

    Index x = threadIdx.x + blockDim.x * blockIdx.x;
    Index y = threadIdx.y + blockDim.y * blockIdx.y;
    Index k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    // (KH, KW, KD)
    // k =  kx * KH  ky kz
    Index k_H = k / (KW * KD);
    Index k_W = (k / KD) % KW;
    Index k_D = k % KD;

    Index oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    Index oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    Index nTile = 0; // TODO

    for (int b = 0; b < B; b++)
    {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        // diverge here
        for (int pos = zStart; pos < zEnd; pos++)
        {
            Index z = zIndices[pos];
            Index oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            Index global_out_idx = grid[b][oX][oY][oZ];
            if (global_out_idx < 0)
                continue;

            Index counter = atomicAdd(&ruleSize[nTile][k], Index(1));

            // grid[b][x][y][z] = pos;
            // rules: [NTile, K*K*K, 4, DMax]
            rules[nTile][k][0][counter] = pos;
            rules[nTile][k][1][counter] = global_out_idx;
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

    // printTensor<int>(grid, "grid", 0, 0, outSpatialShape[0], 0, outSpatialShape[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());
    blockSize = dim3(H_BLOCK, W_BLOCK, kernelVolume);

    int NTile = 1; // TODO, number of Tiles
    // allocate rules and indice Num
    torch::Tensor rules =
        torch::full({NTile, kernelVolume, 2, zIndices.size(0)},
                   /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));
    // rules is allocated larger, to be trimed lalter
    // TODO, change 2 to 4, numAct
    // TODO, last dimension... is NNZ now, But not NNZ if NTile > 1

    torch::Tensor ruleSize =
        torch::zeros({NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));

    getSubMRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        batchSize,
        spatialShape[0], spatialShape[1],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // outZPtr
    // outIndices ?

    return {zIndices, zPtr, rules, ruleSize};
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
get_rules(torch::Tensor zIndices, //  [NNZ]
          torch::Tensor zPtr,     // [B, H, W]
          torch::Tensor grid,     // [B, oH, oW, oD]
          int batchSize,
          std::vector<int64_t> spatialShape,    // H, W, D
          std::vector<int64_t> outSpatialShape, // oH, oW, oD
          std::vector<int64_t> kernelSize,
          std::vector<int64_t> stride,
          std::vector<int64_t> padding,
          std::vector<int64_t> dilation)
{
    grid.fill_(0);

    const int H_BLOCK = 4;
    const int W_BLOCK = 16;
    int num_active = zIndices.size(0);

    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());
    dim3 gridSize = dim3(divUp(spatialShape[0], H_BLOCK), divUp(spatialShape[1], W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, kernelVolume);

    printf(" befaore preapre geridf kernel a\n");

    prepareGridKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        batchSize, spatialShape[0], spatialShape[1],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]);

    // printTensor<int>(grid, "grid", 0, 0, outSpatialShape[0], 0, outSpatialShape[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    torch::Tensor fiberSize = torch::sum(grid, {3}, false, torch::kInt32); // [B, oH, oW]
    PRINT_SHAPE(fiberSize);
    torch::Tensor ozPtr = torch::cumsum(fiberSize.reshape({-1}), 0)
                              .reshape({batchSize, outSpatialShape[0], outSpatialShape[1]})
                              .toType(torch::kInt32);
    // [B, oH, oW]
    PRINT_SHAPE(ozPtr);
    PRINT_SHAPE(grid);
    grid += ozPtr; // now grid is filled with global output index

    int NTile = 1; // TODO, number of Tiles

    torch::Tensor rules =
        torch::full({NTile, kernelVolume, 2, zIndices.size(0)}, -1, torch::dtype(torch::kInt32).device(zIndices.device()));
    // rules is allocated larger, to be trimed lalter
    // TODO, change 2 to 4, numAct
    // TODO, last dimension... is NNZ now, But not NNZ if NTile > 1
    PRINT_SHAPE(rules);

    torch::Tensor ruleSize =
        torch::zeros({NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));
    PRINT_SHAPE(ruleSize);

    int outNNZ = ozPtr.view({-1}).index({-1}).item<int>();
    PRINT_SHAPE(ozPtr);

    torch::Tensor ozIndices = torch::empty({outNNZ}, torch::dtype(torch::kInt32).device(zIndices.device()));
    PRINT_SHAPE(ozIndices);

    getOzIndicesAndRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        ozIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        ozPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        fiberSize.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        batchSize, spatialShape[0], spatialShape[1],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        dilation[0], dilation[1], dilation[2],
        padding[0], padding[1], padding[2]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // outZPtr
    // outIndices ?

    return {ozIndices, ozPtr, rules, ruleSize};
}

} // namespace sphconv
