
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "timer.h"
#include "sphconv/debug_utils.h"
#include "assert.h"

using namespace std;
using namespace torch::indexing;

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

int huristic_tile_n_max(int inTileSizeH, int inTileSizeW, int B, int H, int W, int D, int NNZ)
{
    double sparsity = (double)NNZ / (double)(B * H * W * D);

    double size = B * inTileSizeH * inTileSizeW * D * sparsity * 2;

    return (int)(size + 64) & (-1 << 4);
}

__host__ __device__ __forceinline__ int getInTileSize(int outTileSize, int stride, int kernelSize)
{
    assert(stride <= kernelSize);
    return stride * (outTileSize - 1) + kernelSize;
}

template <typename IType>
__device__ __forceinline__ IType OutSpatial(IType k, IType x, IType s, IType d,
                                            IType pad)
{
    // forgive me. do nothing with the dillation
    // TODO: dilation
    if ((x + pad - k) % s == 0)
        return (x + pad - k) / s;
    return -1;
}

/**
 * @brief init the grid[B, H, W, D],
 *  grid is a mapping from spatial location to its target glaobal physical location
 *
 * fill grid with global indices
 *
 * @return __global__
 */
template <typename IType>
__global__ void prepareSubMGridKernel(
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    // TODO replace zPtr with exclusiveScan
    GpuTensor<IType, 4> grid) // [B, H, W, D]
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= H || y >= W)
        return;

    // for each voxel
    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : *(&zPtr[b][x][y] - 1);

        // diverge here, but we assume it's quick
        for (int pos = zStart; pos < zEnd; pos++) {
            int z = zIndices[pos];
            grid[b][x][y][z] = pos;
        }
    }
}

/**
 * @brief init the grid[B, oH, oW, oD],
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
template <typename IType>
__global__ void prepareGridKernel(
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    GpuTensor<IType, 4> grid,           // [B, oH, oW, oD]
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD)
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int oH = grid.size(1);
    int oW = grid.size(2);
    int oD = grid.size(3);

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z;

    if (x >= H || y >= W)
        return;

    // (KH, KW, KD)
    // k =  kx * KH  ky kz
    int k_H = k / (KW * KD);
    int k_W = (k / KD) % KW;
    int k_D = k % KD;

    int oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    int oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    /// for each input voxel, fill its output to 1
    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];

        // diverge here, but we assume it's quick
        for (int pos = zStart; pos < zEnd; pos++) {
            int z = zIndices[pos];
            int oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            grid[b][oX][oY][oZ] = 1;
        }
    }
}

/**
 *  for std conv
 *  create ozIndices and Rule s.
 */
template <typename IType>
__global__ void getOzIndicesAndRulesKernel(
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    GpuTensor<IType, 1> ozIndices,      // [NNZ']
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    const GpuTensor<IType, 3> ozPtr,    // [B, oH, oW]
    const GpuTensor<IType, 4> grid,     // [B, oH, oW, oD]
    GpuTensor<IType, 4> rules,          // [NTile, KKK, 2, DMax]
    GpuTensor<IType, 2> ruleSize,       // number active index, [NTile, KKK]
    int D,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD,
    int outTileH, int outTileW)
{
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int oH = grid.size(1);
    int oW = grid.size(2);
    int oD = grid.size(3);

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    int k_H = k / (KW * KD);
    int k_W = (k / KD) % KW;
    int k_D = k % KD;

    int oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    int oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    int TileGridW = divUp(oW, outTileW);
    int nTile = getLinearTileIdx(outTileH, outTileW, oX, oY, TileGridW);
    // printf("nTile(%d), oxy(%d,%d)\n", nTile, oX, oY);
    // nTile = 1;

    int tile_n_max = rules.size(3);

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];

        for (int globalInIdx = zStart; globalInIdx < zEnd; globalInIdx++) {
            int z = zIndices[globalInIdx];
            int oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            int globalOutIdx = grid[b][oX][oY][oZ] - 1;
            int counter = atomicAdd(&ruleSize[nTile][k], 1);

            if (counter < tile_n_max) {
                rules[nTile][k][0][counter] = globalInIdx;
                rules[nTile][k][1][counter] = globalOutIdx;
            } else {
                printf("overflow counter:(%d/%d), global i/o:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), oShape(%d,%d,%d), std\n",
                       counter, tile_n_max, globalInIdx, globalOutIdx,
                       nTile, x, y, k, outTileH, outTileW, oH, oW, oD);
            }
            ozIndices[globalOutIdx] = oZ;
        }
    } // b
}

/***
 *  fill rules,
        rules: [NTile, K*K*K, 4, DMax]
 */
template <typename IType>
__global__ void getSubMRulesKernel(
    const GpuTensor<IType, 1> zIndices, // [NNZ]
    const GpuTensor<IType, 3> zPtr,     // [B, H, W]
    const GpuTensor<IType, 4> grid,     // [B, oH, oW, oD]
    GpuTensor<IType, 4> rules,          // [NTile, KKK, 2, DMax]
    GpuTensor<IType, 2> ruleSize,       // number active index, [NTile, KKK]
    int D,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD,
    int outTileH, int outTileW)
{
    int tile_n_max = rules.size(3);
    int B = zPtr.size(0);
    int H = zPtr.size(1);
    int W = zPtr.size(2);
    int oH = grid.size(1);
    int oW = grid.size(2);
    int oD = grid.size(3);

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= H || y >= W)
        return;

    int k = threadIdx.z + blockDim.z * blockIdx.z;
    int k_H = k / (KW * KD);
    int k_W = (k / KD) % KW;
    int k_D = k % KD;

    int oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    int oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    // int TileGridH = divUp(oH, outTileH);
    int TileGridW = divUp(oW, outTileW);
    int nTile = getLinearTileIdx(outTileH, outTileW, oX, oY, TileGridW);
    // printf("nTile(%d), oxy(%d,%d)\n", nTile, oX, oY);
    // nTile = 1;

    for (int b = 0; b < B; b++) {
        int zEnd = zPtr[b][x][y];
        int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];

        // diverge here
        for (int globalInIdx = zStart; globalInIdx < zEnd; globalInIdx++) {
            int z = zIndices[globalInIdx];
            int oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            int globalOutIdx = grid[b][oX][oY][oZ];
            if (globalOutIdx < 0)
                continue;

            // printf(" Tile(%d,%d), in(%d,%d,%d), out(%d,%d,%d), pair(%d,%d), k=%d\n",
            //        oX / outTileH, oY / outTileW,
            //        x, y, z, oX, oY, oZ,
            //        globalInIdx, globalOutIdx, k);

            int counter = atomicAdd(&ruleSize[nTile][k], 1);

            if (counter < tile_n_max) {
                rules[nTile][k][0][counter] = globalInIdx;
                rules[nTile][k][1][counter] = globalOutIdx;
            } else {
                printf("overflow counter:(%d/%d), global i/o:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), inShape(%d,%d,%d), std\n",
                       counter, tile_n_max, globalInIdx, globalOutIdx,
                       nTile, x, y, k, outTileH, outTileW, oH, oW, oD);
            }
        }
    } // b
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
get_rules_subm(const torch::Tensor zIndices, //  [NNZ]
               const torch::Tensor zPtr,     // [B, H, W]
               int batchSize,
               std::vector<int64_t> spatialShape,    // H, W, D
               std::vector<int64_t> outSpatialShape, // H, W, D
               std::vector<int64_t> kernelSize,
               std::vector<int64_t> stride,
               std::vector<int64_t> padding,
               std::vector<int64_t> dilation,
               std::vector<int64_t> tileSize)
{

    torch::Tensor grid = torch::full({batchSize, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]},
                                     /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

    dim3 gridSize(divUp(spatialShape[0], 16), divUp(spatialShape[1], 32));
    dim3 blockSize(16, 32, 1);
    prepareSubMGridKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());

    // number of tiles
    int NTile = divUp(outSpatialShape[0], tileSize[0]) * divUp(outSpatialShape[1], tileSize[1]);

    // allocate rules and indice Num
    int tile_n_max = huristic_tile_n_max(
        getInTileSize(tileSize[0], stride[0], kernelSize[0]),
        getInTileSize(tileSize[1], stride[1], kernelSize[1]),
        batchSize, spatialShape[0], spatialShape[1], spatialShape[2], zIndices.size(0));
    // printf("tile_n_max = %d\n", tile_n_max);

    torch::Tensor rules = torch::full(
        {NTile, kernelVolume, 2, tile_n_max},
        /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

    torch::Tensor ruleSize = torch::zeros(
        {NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));

    gridSize = dim3(divUp(spatialShape[0], 4), divUp(spatialShape[1], 8), kernelVolume);
    blockSize = dim3(4, 8, 1);
    getSubMRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        spatialShape[2],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        tileSize[0], tileSize[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

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
get_rules(const torch::Tensor zIndices, //  [NNZ]
          const torch::Tensor zPtr,     // [B, H, W]
          int batchSize,
          std::vector<int64_t> spatialShape,    // H, W, D
          std::vector<int64_t> outSpatialShape, // oH, oW, oD
          std::vector<int64_t> kernelSize,
          std::vector<int64_t> stride,
          std::vector<int64_t> padding,
          std::vector<int64_t> dilation,
          std::vector<int64_t> tileSize)
{
    torch::Tensor grid = torch::zeros({batchSize, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]},
                                      torch::dtype(torch::kInt32).device(zIndices.device()));
    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());

    dim3 gridSize(divUp(spatialShape[0], 2), divUp(spatialShape[1], 16), 1);
    dim3 blockSize(2, 16, kernelVolume);
    prepareGridKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2]);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    grid = torch::cumsum(grid, 3, torch::kInt32); // [B, oH, oW, oD]

    // std::cout << "grid(2) = " << grid << std::endl;
    // here we want non inclusive scan, but pytorch only provides this.
    torch::Tensor ozPtr = torch::cumsum(grid.index({Slice(), Slice(), Slice(), -1}).reshape({-1}), 0, torch::kInt32)
                              .reshape({batchSize, outSpatialShape[0], outSpatialShape[1]});
    // [B, oH, oW]
    torch::Tensor exclusiveScan = ozPtr.roll(1);
    exclusiveScan.index_put_({0, 0, 0}, 0);
    grid += exclusiveScan.unsqueeze(-1); // now grid is filled with global output index
    // std::cout << "grid(3) = " << grid << std::endl;

    int NTile = divUp(outSpatialShape[0], tileSize[0]) * divUp(outSpatialShape[1], tileSize[1]);

    int tile_n_max = huristic_tile_n_max(
        getInTileSize(tileSize[0], stride[0], kernelSize[0]),
        getInTileSize(tileSize[1], stride[1], kernelSize[1]),
        batchSize, spatialShape[0], spatialShape[1], spatialShape[2], zIndices.size(0));

    torch::Tensor rules = torch::full(
        {NTile, kernelVolume, 2, tile_n_max},
        /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

    torch::Tensor ruleSize = torch::zeros(
        {NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));
    // PRINT_SHAPE(ruleSize);

    int outNNZ = ozPtr.view({-1}).index({-1}).item<int>();
    torch::Tensor ozIndices = torch::empty({outNNZ}, torch::dtype(torch::kInt32).device(zIndices.device()));
    // PRINT_SHAPE(ozIndices);

    gridSize = dim3(divUp(spatialShape[0], 2), divUp(spatialShape[1], 16), kernelVolume);
    blockSize = dim3(2, 16, 1);
    getOzIndicesAndRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        ozIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        ozPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        spatialShape[2],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        tileSize[0], tileSize[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {ozIndices, ozPtr, rules, ruleSize};
}

} // namespace sphconv
