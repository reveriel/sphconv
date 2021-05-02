#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "timer.h"
#include "debug_utils.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

template <typename IType, typename DType,
          int V_BLOCK,
          int IC_BLOCK,
          int OC_BLOCK>
__global__ void ruleConvKernel(
    const GpuTensor<DType, 2> feature,
    const GpuTensor<DType, 3> weight, // [kernelVolume, oC, iC]
    const GpuTensor<IType, 4> localRules,  // [ NTile, kernelVolume, 4, NNZ']
    const GpuTensor<IType, 2> ruleSize,  // [NTile, kernelVolume]
    const GpuTensor<IType, 3> globalRules, // [NTile, 2, TILE_N_MAX]
    GpuTensor<DType, 2> outFeature,
    int kernelVolume,
    int iC, int oC)
{

    __shared__ DType input[V_BLOCK][IC_BLOCK];
    __shared__ DType output[V_BLOCK][OC_BLOCK];
    __shared__ DType subKernel[IC_BLOCK][OC_BLOCK];

    // set output to zeros
    for (int v = threadIdx.x; v < V_BLOCK; v += blockDim.x) {
        for (int oc = threadIdx.y; oc < oC; oc += blockDim.y) {
            output[v][oc] = DType(0);
        }
    }

    // for each NTile
    int tile = blockIdx.x;

    // load tile data to shared memory
    for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
        int global_in_idx = globalRules[tile][0][v];
        if (global_in_idx == -1)
            break;

        for (int ic = threadIdx.y; ic < iC; ic += blockDim.y) {
            input[v][ic] = feature[global_in_idx][ic];
        }

        if (threadIdx.y == 0) {
            printf("input[%d][0] = %f, feature[global_in:%d][0] = %f\n",
                v, input[v][0],  global_in_idx, feature[global_in_idx][0]);
        }
    }

    // for each kernelVolume * NNZ'
    for (int k = 0; k < kernelVolume; k++)
    {
        // load input data to shared memory
        int kRuleSize = ruleSize[tile][k];

        // load kernel to shared memory
        for (int ic = threadIdx.x;  ic < iC; ic += blockDim.x) {
            for (int oc = threadIdx.y; oc < oC; oc += blockDim.y) {
                subKernel[ic][oc] = weight[k][oc][ic]; // TODO : swap kernel oC iC order
            }
        }
        __syncthreads();

        // matrix multiply
        for (int v = threadIdx.x; v < kRuleSize; v += blockDim.x) {
            int local_in_idx = localRules[tile][k][2][v];
            int local_out_idx = localRules[tile][k][3][v];
            if (threadIdx.y == 0)
                printf("lcal in/ out idx = %d / %d \n", local_in_idx, local_out_idx);
            for (int oc = threadIdx.y; oc < oC; oc += blockDim.y) {
                DType sum = DType(0);

                if (threadIdx.y == 0)
                    printf("input[%d][0] = %f, subKernel[0][0] = %f, k = %d\n",
                           local_in_idx, input[local_in_idx][0], subKernel[0][0], k);
                for (int ic = 0; ic < iC; ic++) {
                    sum += input[local_in_idx][ic] * subKernel[ic][oc];
                }

                if (threadIdx.y == 0)
                    printf("sum = %f, in  v(%d)\n", sum, v);
                output[local_out_idx][oc] += sum;
            }
        }
        __syncthreads();
    } // k

    __syncthreads();
    // fill back to global output
    for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
        int global_out_idx = globalRules[tile][1][v];
        if (global_out_idx == -1)
            break;
        for (int oc = threadIdx.y; oc < oC; oc += blockDim.y) {
            outFeature[global_out_idx][oc] = output[v][oc];
        }
    }
}

/**
 * @brief
 *
 *  TODO: localRules : size
 *
 * @param feature
 * @param weight
 * @param localRules
 * @param ruleSize
 * @return std::vector<torch::Tensor>
 */
torch::Tensor
rule_conv(torch::Tensor feature,  //  [NNZ, C]
          torch::Tensor weight,   // [kernelVolume, Co, Ci]
          torch::Tensor localRules,    //  [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize, // [Ntile, kernelVolume]
          torch::Tensor globalRules, // [NTile, 2, TILE_N_MAX]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape,
          int outNNZ)
{

    int C = feature.size(1);
    int kernelVolume = weight.size(0);
    int oC = weight.size(1);
    int iC = weight.size(2);

    int IC_BLOCK = 16;
    // TODO: define shared memory size
    if (C <= 16)
    {
        IC_BLOCK = 16;
    }
    else
    {
        IC_BLOCK = 32;
    }
    const int OC_BLOCK = 32;
    // const int OC_ILP = 4;
    const int VOX_BLOCK = 32;

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    // dim3 gridSize = dim3(NTile, divUp(iC, IC_BLOCK), divUp(oC, OC_BLOCK));
    dim3 gridSize = dim3(NTile);
    dim3 blockSize = dim3(VOX_BLOCK, OC_BLOCK, 1);

    // global version, with no shared memory
    switch (IC_BLOCK)
    {
    case 16:
        ruleConvKernel<int32_t, float, VOX_BLOCK, 16, OC_BLOCK>
            <<<gridSize, blockSize>>>(
                feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                globalRules.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                kernelVolume, iC, oC);
        break;
    case 32:
        ruleConvKernel<int32_t, float, VOX_BLOCK, 32, OC_BLOCK>
            <<<gridSize, blockSize>>>(
                feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                globalRules.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                kernelVolume, iC, oC);
        break;
    }
    // ruleConvKernel<int32_t, float, VOX_BLOCK, IC_BLOCK, OC_BLOCK>
    //     <<<gridSize, blockSize>>>(
    //         feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //         weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
    //         localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
    //         ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
    //         outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //         kernelVolume, iC, oC);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return outFeature;
}

} // namespace sphconv
