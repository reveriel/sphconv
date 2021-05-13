
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "timer.h"
#include "debug_utils.h"
#include "assert.h"

using namespace std;
using namespace torch::indexing;


namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

__host__ __device__ __forceinline__
int getInTileSize(int outTileSize, int stride, int kernelSize)
{
    assert(stride <= kernelSize);
    return stride * (outTileSize - 1) + kernelSize;
}

__host__ __device__ __forceinline__
int linearIdx(int x0, int x1, int x2, int x3, int D1, int D2, int D3) {
    return ((x0 * D1 + x1) * D2 + x2) * D3 + x3;
}

__host__ __device__ __forceinline__
int linearIdx(int x0, int x1, int x2, int D1, int D2) {
    return (x0 * D1 + x1) * D2 + x2;
}

__host__ __device__ __forceinline__
int linearIdx(int x0, int x1, int D1) {
    return x0 * D1 + x1;
}

template <typename IType>
__device__ __forceinline__
int getLocalShift(const GpuTensor<IType, 3> &zPtr, // [B, H, W]
                  int TileSizeH, int TileSizeW,
                  int H, int W, int base,
                  int b, int x, int y,
                  int padH, int padW)
{
    // int tileIdxX = (x + padH) / TileSizeH;
    int tileIdxY = (y + padW) / TileSizeW;
    // int x0 = tileIdxX * TileSizeH;
    int y0 = max((tileIdxY * TileSizeW) - padW, 0);

    // last element outside the tile,
    int a = ((b == 0 && x == 0 && y0 == 0) ?  0 : zPtr[b][x][y0 - 1]);

    return (-a + base);
}

// virtual index to real index
// virtual index is counted including padding
__device__ __forceinline__
int v2r(int v, int pad) {
    return max(v - pad, 0);
}

template <typename IType>
__device__ __forceinline__
int getLocalInShift(const GpuTensor<IType, 3> &zPtr, // [B, H, W]
                  int TileSizeH, int TileSizeW,
                  int H, int W, int base,
                  int b, int x, int y,
                  int padH, int padW,
                  int oY, int outTileSizeW, int sW)
{
    // int tileIdxX = (x + padH) / TileSizeH;
    int tileIdxY = oY / outTileSizeW;
    // int x0 = tileIdxX * TileSizeH;
    int y0 = (tileIdxY * sW * outTileSizeW);
    int ry0 = v2r(y0, padW);

    // last element outside the tile,
    int a = ((b == 0 && x == 0 && ry0 == 0) ?  0 : zPtr[b][x][ry0 - 1]);

    return (-a + base);
}


__device__ __forceinline__
int getLinearTileIdx(int TileSize0, int TileSize1, int x, int y, int TileGridW)
{
    int tileIdxX = x / TileSize0;
    int tileIdxY = y / TileSize1;
    return linearIdx(tileIdxX, tileIdxY, TileGridW);
}

/** number of num zero in this line (x), at this tile */
template <typename IType>
__device__ __forceinline__
int updateBase(const GpuTensor<IType, 3> &zPtr, int H, int W,
               int b, int x, int y, int TileSizeH, int TileSizeW, int padH, int padW)
{
    int tileIdxY = (y + padW) / TileSizeW;
    // y0 is the first y in this line (x) in this tile.
    int y0 = max((tileIdxY * TileSizeW) - padW, 0); // wrong

    return zPtr[b][x][min(y0 + TileSizeW - 1, W - 1)] - ((b == 0 && x == 0 && y0 == 0) ? 0 : zPtr[b][x][y0 - 1]);
}

/** number of num zero in this line (x), at this tile */
template <typename IType>
__device__ __forceinline__
int updateBaseIn(const GpuTensor<IType, 3> &zPtr, int H, int W,
               int b, int x, int y, int TileSizeH, int TileSizeW, int padH, int padW, int oY, int outTileSizeW, int sW)
{
    int tileIdxY = oY / outTileSizeW;
    // y0 is the first y in this line (x) in this tile.
    int y0 = (tileIdxY * sW * outTileSizeW);
    int ry0 = v2r(y0, padW);

    return zPtr[b][x][min( v2r(y0 + TileSizeW - 1, padW), W - 1)] - ((b == 0 && x == 0 && ry0 == 0) ? 0 : zPtr[b][x][ry0 - 1]);
}


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
template <typename IType>
__global__ void prepareSubMGridKernel(
    const GpuTensor<IType, 1> zIndices,
    const GpuTensor<IType, 3> zPtr, // TODO replace zPtr with exclusiveScan
    GpuTensor<IType, 4> grid,
    int B, int H, int W)
{
    IType x = threadIdx.x + blockDim.x * blockIdx.x;
    IType y = threadIdx.y + blockDim.y * blockIdx.y;
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
            IType z = zIndices[pos];
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
    const GpuTensor<IType, 1> zIndices,
    const GpuTensor<IType, 3> zPtr,
    GpuTensor<IType, 4> grid, // [B, oH, oW, oD]
    int B, int H, int W,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD)
{
    IType oH = grid.size(1);
    IType oW = grid.size(2);
    IType oD = grid.size(3);

    IType x = threadIdx.x + blockDim.x * blockIdx.x;
    IType y = threadIdx.y + blockDim.y * blockIdx.y;
    IType k = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= H || y >= W)
        return;

    // (KH, KW, KD)
    // k =  kx * KH  ky kz
    IType k_H = k / (KW * KD);
    IType k_W = (k / KD) % KW;
    IType k_D = k % KD;

    IType oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH)
        return;
    IType oY = OutSpatial(k_W, y, sW, dW, padW);
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
            IType z = zIndices[pos];
            IType oZ = OutSpatial(k_D, z, sD, dD, padD);
            if (oZ < 0 || oZ >= oD)
                continue;

            grid[b][oX][oY][oZ] = IType(1);
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
    const GpuTensor<IType, 4> grid,
    GpuTensor<IType, 4> localRules,  // [NTile, KKK, 2, DMax]
    GpuTensor<IType, 2> ruleSize,    // number active index, [NTile, KKK]
    GpuTensor<IType, 3> globalRules, // [NTile, 2, MAX]
    int B, int H, int W, int D,  // TODO, cleanup unnaccesary
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD,
    int inTileH, int inTileW,
    int outTileH, int outTileW)
{
    IType oH = grid.size(1);
    IType oW = grid.size(2);
    IType oD = grid.size(3);

    // IType x = threadIdx.x + blockDim.x * blockIdx.x;
    IType y = threadIdx.y + blockDim.y * blockIdx.y; // TODO
    IType k = threadIdx.z + blockDim.z * blockIdx.z;

    if (y >= W)
        return;

    IType k_H = k / (KW * KD);
    IType k_W = (k / KD) % KW;
    IType k_D = k % KD;

    IType oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    int TileGridW = divUp(oW, outTileW);

    int baseIn = 0;
    int baseOut = 0;
    for (int b = 0; b < B; b++)
    {
        int oldOutX = 0; // refresh

        for (int x = 0; x < H; x++)
        {
            IType oX = OutSpatial(k_H, x, sH, dH, padH);
            int zEnd = zPtr[b][x][y];
            int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];
            IType nTile = getLinearTileIdx(outTileH, outTileW, oX, oY, TileGridW);

            if (x != 0)
                baseIn += updateBaseIn(zPtr, H, W, b, x - 1, y, inTileH, inTileW, padH, padW, oY, outTileW, sW);
            while (oldOutX < oX) {
                baseOut += updateBase(ozPtr, oH, oW, b, oldOutX, oY, outTileH, outTileW, 0, 0);
                oldOutX++;
            }

            if (oX < 0)
                continue;
            if (oX >= oH)
                break; // next batch ?

            for (int globalInIdx = zStart; globalInIdx < zEnd; globalInIdx++)
            {
                IType z = zIndices[globalInIdx];
                IType oZ = OutSpatial(k_D, z, sD, dD, padD);
                if (oZ < 0 || oZ >= oD)
                    continue;

                IType globalOutIdx = grid[b][oX][oY][oZ] - 1;
                IType counter = atomicAdd(&ruleSize[nTile][k], IType(1));

                int localInIdx = globalInIdx + getLocalInShift(zPtr, inTileH, inTileW,
                                                             H, W, baseIn, b, x, y, padH, padW, oY, outTileW, sW);
                int localOutIdx = globalOutIdx + getLocalShift(ozPtr, outTileH, outTileW,
                                                               oH, oW, baseOut, b, oX, oY, 0, 0);
                localRules[nTile][k][0][counter] = globalInIdx;
                localRules[nTile][k][1][counter] = globalOutIdx;

                // these assignment to globalRules executes many times, with the same value
                if (localInIdx < TILE_N_MAX)
                    globalRules[nTile][0][localInIdx] = globalInIdx;
                else
                    printf("overflow, input Idx:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), inShape(%d,%d,%d), std\n",
                           localInIdx, TILE_N_MAX, nTile, x, y, k, inTileH, inTileW, H, W, D);

                if (localOutIdx < TILE_N_MAX)
                    globalRules[nTile][1][localOutIdx] = globalOutIdx;
                else
                    printf("overflow, output Idx:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), outShape(%d,%d,%d), std\n",
                           localOutIdx, TILE_N_MAX, nTile, x, y, k, outTileH, outTileW, oH, oW, oD);

                ozIndices[globalOutIdx] = oZ;
            }
        } // x

        baseIn += updateBaseIn(zPtr, H, W, b, H - 1, y, inTileH, inTileW, padH, padW, oY, outTileW, sW);
        while (oldOutX < oH) {
            baseOut += updateBase(ozPtr, oH, oW, b, oldOutX, oY, outTileH, outTileW, 0, 0);
            oldOutX++;
        }
    } // b
}

/***
 *  fill rules,
        rules: [NTile, K*K*K, 4, DMax]
 */
template <typename IType>
__global__ void getSubMRulesKernel(
    const GpuTensor<IType, 1> zIndices,
    const GpuTensor<IType, 3> zPtr,
    const GpuTensor<IType, 4> grid,
    GpuTensor<IType, 4> localRules,
    GpuTensor<IType, 2> ruleSize, // number active index, [NTile, KKK]
    GpuTensor<IType, 3> globalRules,
    int B, int H, int W, int D,
    int KH, int KW, int KD,
    int sH, int sW, int sD,
    int padH, int padW, int padD,
    int dH, int dW, int dD,
    int inTileH, int inTileW,
    int outTileH, int outTileW)
{
    // extern __shared__ int *shared;
    // IType *inTileGrid = &shared[0];
    // IType *outTileGrid = &shared[inTileGridSize];

    IType oH = grid.size(1);
    IType oW = grid.size(2);
    IType oD = grid.size(3);

    // IType x = threadIdx.x + blockDim.x * blockIdx.x;
    IType y = threadIdx.y + blockDim.y * blockIdx.y;
    IType k = threadIdx.z + blockDim.z * blockIdx.z;

    if (y >= W)
        return;

    IType k_H = k / (KW * KD);
    IType k_W = (k / KD) % KW;
    IType k_D = k % KD;

    IType oY = OutSpatial(k_W, y, sW, dW, padW);
    if (oY < 0 || oY >= oW)
        return;

    // int TileGridH = divUp(oH, outTileH);
    int TileGridW = divUp(oW, outTileW);

    int baseIn = 0;
    int baseOut = 0;
    for (int b = 0; b < B; b++)
    {
        int oldOutX = 0;

        for (int x = 0; x < H; x++)
        {
            IType oX = OutSpatial(k_H, x, sH, dH, padH); // TODO, iterative
            int zEnd = zPtr[b][x][y];
            int zStart = (b == 0 && x == 0 && y == 0) ? 0 : zPtr[b][x][y - 1];
            IType nTile = getLinearTileIdx(outTileH, outTileW, oX, oY, TileGridW);

            if (x != 0)
                baseIn += updateBaseIn(zPtr, H, W, b, x - 1, y, inTileH, inTileW, padH, padW, oY, outTileW, sW);
            while (oldOutX < oX) {
                baseOut += updateBase(zPtr, oH, oW, b, oldOutX, oY, outTileH, outTileW, 0, 0);
                oldOutX++;
            }

            if (oX < 0)
                continue;
            if (oX >= oH)
                break; // next batch ?

            // diverge here
            for (int globalInIdx = zStart; globalInIdx < zEnd; globalInIdx++)
            {
                IType z = zIndices[globalInIdx];
                IType oZ = OutSpatial(k_D, z, sD, dD, padD);
                if (oZ < 0 || oZ >= oD)
                    continue;

                IType globalOutIdx = grid[b][oX][oY][oZ];
                if (globalOutIdx < 0)
                    continue;

                IType counter = atomicAdd(&ruleSize[nTile][k], IType(1));

                int localInIdx = globalInIdx + getLocalInShift(zPtr, inTileH, inTileW,
                                                             H, W, baseIn, b, x, y, padH, padW, oY, outTileW, sW);
                int localOutIdx = globalOutIdx + getLocalShift(zPtr, outTileH, outTileW,
                                                               oH, oW, baseOut, b, oX, oY, 0, 0);
                localRules[nTile][k][0][counter] = globalInIdx;
                localRules[nTile][k][1][counter] = globalOutIdx;

                // these assignment to globalRules executes many times, with the same value
                if (localInIdx < TILE_N_MAX)
                    globalRules[nTile][0][localInIdx] = globalInIdx;
                else
                    printf("overflow, input Idx:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), inShape(%d,%d,%d), subm\n",
                           localInIdx, TILE_N_MAX, nTile, x, y, k, inTileH, inTileW, H, W, D);

                if (localOutIdx < TILE_N_MAX)
                    globalRules[nTile][1][localOutIdx] = globalOutIdx;
                else
                    printf("overflow, output Idx:%d/%d, nTile:%d, x:%d, y:%d, k:%d, Tile(%d,%d), outShape(%d,%d,%d), subm\n",
                           localOutIdx, TILE_N_MAX, nTile, x, y, k, outTileH, outTileW, oH, oW, oD);
            }
        } // x
        baseIn += updateBaseIn(zPtr, H, W, b, H - 1, y, inTileH, inTileW, padH, padW, oY, outTileW, sW);
        while (oldOutX < oH) {
            baseOut += updateBase(zPtr, oH, oW, b, oldOutX, oY, outTileH, outTileW, 0, 0);
            oldOutX++;
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
get_rules_subm(torch::Tensor zIndices, //  [NNZ]
               torch::Tensor zPtr,     // [B, H, W]
               int batchSize,
               std::vector<int64_t> spatialShape,    // H, W, D
               std::vector<int64_t> outSpatialShape, // H, W, D
               std::vector<int64_t> kernelSize,
               std::vector<int64_t> stride,
               std::vector<int64_t> padding,
               std::vector<int64_t> dilation)
{
    int H_BLOCK = 4;
    int W_BLOCK = 32;

    dim3 gridSize = dim3(divUp(spatialShape[0], H_BLOCK), divUp(spatialShape[1], W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, 1);

    torch::Tensor grid = torch::full({batchSize, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]},
                                     /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

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

    int outTileH = 8; // TODO
    int outTileW = 8;
    if (outSpatialShape[0] <= 8) {
        outTileH = 4;
    }

    int NTile = divUp(outSpatialShape[0], outTileH) * divUp(outSpatialShape[1], outTileW);

    // allocate rules and indice Num
    torch::Tensor localRules =
        torch::full({NTile, kernelVolume, 2, TILE_N_MAX}, // TODO: TILE_N_MAX is fixed, not elegent
                    /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));
    // TODO: rules is allocated larger, to be trimed lalter

    torch::Tensor globalRules =
        torch::full({NTile, 2, TILE_N_MAX}, /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

    torch::Tensor ruleSize =
        torch::zeros({NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));

    int inTileH = getInTileSize(outTileH, stride[0], kernelSize[0]);
    int inTileW = getInTileSize(outTileW, stride[1], kernelSize[1]);

    W_BLOCK = 16;
    gridSize = dim3(1, divUp(spatialShape[1], W_BLOCK), 1);
    blockSize = dim3(1, W_BLOCK, kernelVolume);
    // int inTileGridSize = batchSize * inTileH * inTileW * spatialShape[2];
    // int outTileGridSize = batchSize * outTileH * outTileW * outSpatialShape[2];
    // auto sharedMemorySize =  sizeof(int32_t) * (inTileGridSize +
    // outTileGridSize);
    getSubMRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        globalRules.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        batchSize,
        spatialShape[0], spatialShape[1], spatialShape[2],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        inTileH, inTileW,
        outTileH, outTileW);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // loadingRule, global indices for each tile
    // shape [NTile, NMax]
    // Note Rule have indiex of '-1'
    // uniq loadingRule  =  tensor([[-1,  0,  1,  2,  3],
    //     [-1, -1, -1, -1, -1],
    //     [-1, -1, -1, -1, -1],
    //     [-1, -1, -1, -1, -1]],
    // torch::Tensor loadingRule =
    //     std::get<0>(
    //         torch::unique_dim(rules.index({Slice(), Slice(), 0, Slice()}).reshape({NTile, -1}), /*dim=*/1));

    return {zIndices, zPtr, localRules, ruleSize, globalRules};
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
          int batchSize,
          std::vector<int64_t> spatialShape,    // H, W, D
          std::vector<int64_t> outSpatialShape, // oH, oW, oD
          std::vector<int64_t> kernelSize,
          std::vector<int64_t> stride,
          std::vector<int64_t> padding,
          std::vector<int64_t> dilation)
{
    int H_BLOCK = 2; // TODO
    int W_BLOCK = 16;

    int64_t kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int64_t>());
    dim3 gridSize = dim3(divUp(spatialShape[0], H_BLOCK), divUp(spatialShape[1], W_BLOCK), 1);
    dim3 blockSize = dim3(H_BLOCK, W_BLOCK, kernelVolume);

    // printf(" befaore preapre geridf kernel a\n");
    // printf("launch config : (%d,%d,%d),(%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

    torch::Tensor grid = torch::zeros({batchSize, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]},
                                      torch::dtype(torch::kInt32).device(zIndices.device()));

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
    // std::cout << "grid(1) = " << grid << std::endl;

    grid = torch::cumsum(grid, 3, torch::kInt32); // [B, oH, oW, oD]

    // std::cout << "grid(2) = " << grid << std::endl;
    // here we want non inclusive scan, but pytorch only provides this.
    torch::Tensor ozPtr = torch::cumsum(grid.index({Slice(), Slice(), Slice(), -1}).reshape({-1}), 0, torch::kInt32)
                              .reshape({batchSize, outSpatialShape[0], outSpatialShape[1]});
    // [B, oH, oW]
    // PRINT_SHAPE(ozPtr);
    // PRINT_SHAPE(grid);
    // std::cout << "ozPtr = " << ozPtr << std::endl;
    torch::Tensor exclusiveScan = ozPtr.roll(1);
    exclusiveScan.index_put_({0, 0, 0}, 0);
    grid += exclusiveScan.unsqueeze(-1); // now grid is filled with global output index
    // std::cout << "grid(3) = " << grid << std::endl;

    int outTileH = 8; // TODO
    int outTileW = 8;
    int inTileH = getInTileSize(outTileH, stride[0], kernelSize[0]);
    int inTileW = getInTileSize(outTileW, stride[1], kernelSize[1]);

    int NTile = divUp(outSpatialShape[0], outTileH) * divUp(outSpatialShape[1], outTileW);

    torch::Tensor localRules = torch::full({NTile, kernelVolume, 2, TILE_N_MAX}, // TODO: TILE_N_MAX is fixed
                                           /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));
    // TODO: rules is allocated larger, to be trimed lalter
    torch::Tensor globalRules = torch::full({NTile, 2, TILE_N_MAX},
                                            /*value=*/-1, torch::dtype(torch::kInt32).device(zIndices.device()));

    torch::Tensor ruleSize =
        torch::zeros({NTile, kernelVolume}, torch::dtype(torch::kInt32).device(zIndices.device()));
    // PRINT_SHAPE(ruleSize);

    int outNNZ = ozPtr.view({-1}).index({-1}).item<int>();
    torch::Tensor ozIndices = torch::empty({outNNZ}, torch::dtype(torch::kInt32).device(zIndices.device()));
    // PRINT_SHAPE(ozIndices);

    W_BLOCK = 16; // TODO
    gridSize = dim3(1, divUp(spatialShape[1], W_BLOCK), 1);
    blockSize = dim3(1, W_BLOCK, kernelVolume);

    getOzIndicesAndRulesKernel<int32_t><<<gridSize, blockSize>>>(
        zIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        ozIndices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        zPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        ozPtr.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        globalRules.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        batchSize, spatialShape[0], spatialShape[1], spatialShape[2],
        kernelSize[0], kernelSize[1], kernelSize[2],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        inTileH, inTileW,
        outTileH, outTileW);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // outZPtr
    // outIndices ?

    return {ozIndices, ozPtr, localRules, ruleSize, globalRules};
}

} // namespace sphconv
