#include <debug_utils.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "timer.h"

#include <vector>
#include <torch/extension.h>

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

template <typename IType, typename DType,
          int V_STEP,
          int IC_BLOCK,
          int OC_BLOCK>
__global__ void ruleConvKernel(
    const GpuTensor<DType, 2> feature,
    const GpuTensor<DType, 3> weight, // [kernelVolume, oC, iC]
    const GpuTensor<IType, 4> rules,
    const GpuTensor<IType, 2> ruleSize,
    GpuTensor<DType, 2> outFeature,
    int kernelVolume,
    int iC, int oC)
{
    // parallel by rules
    // rule [NTile, kernelVolume, 2/4, NNZ']
    // rule Size [NTile, kernelVolume]

    // for each NTile
    int tile = blockIdx.x;
    int ic_block = blockIdx.y; // channel block
    int oc = threadIdx.z + blockIdx.z * blockDim.z; //
    if (oc >= oC)
        return ;
    // printf("thread((%d,%d,%d), (%d,%d,%d))\n",
    //        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

    // for each kernelVolume * NNZ'
    for (int k = 0; k < kernelVolume; k++)
    {
        // for each voxel
        int kRuleSize = ruleSize[tile][k];
        // divergence occurs here, so we divie them into blocks,
        // not serious .. I think
        for (int v = threadIdx.x; v < kRuleSize; v += V_STEP)
        {
            int global_in_idx = rules[tile][k][0][v];
            int global_out_idx = rules[tile][k][1][v];
            // for each outChannel
            // printf("oc = %d\n", oc);
            // for each inChannel segment
            DType sum = 0;
#pragma unroll
            for (int ic = threadIdx.y + ic_block * IC_BLOCK;
                 ic < iC && ic < (ic_block + 1) * IC_BLOCK;
                 ic += 1)
            {
                // printf("thread((%d,%d,%d), (%d,%d,%d)), feature[%d][%d] = %f\n" ,
                //     blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
                //     global_in_idx, ic, feature[global_in_idx][ic]);
                sum += feature[global_in_idx][ic] * weight[k][oc][ic];
                // printf("weight[%d][%d][%d] = %f\n", k, oc, ic, weight[k][oc][ic]);
            }
            // printf("thread((%d,%d,%d), (%d,%d,%d)), outFreature[%d][%d] = %f\n" ,
            // blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
            // global_out_idx, oc, sum);
            atomicAdd(&outFeature[global_out_idx][oc], sum);
        }
    }
}

/**
 * @brief
 *
 *  TODO: rules : size
 *
 * @param feature
 * @param weight
 * @param rules
 * @param ruleSize
 * @return std::vector<torch::Tensor>
 */
torch::Tensor
rule_conv(torch::Tensor feature,  //  [NNZ, C]
          torch::Tensor weight,   // [kernelVolume, Co, Ci]
          torch::Tensor rules,    //  [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize, // [Ntile, kernelVolume]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape
)
{

    // TODO: define shared memory size
    const int IC_BLOCK = 16;
    const int OC_BLOCK = 16;
    // const int OC_ILP = 4;
    const int VOX_BLOCK = 32;

    int kernelVolume = weight.size(0);
    int oC = weight.size(1);
    int iC = weight.size(2);

    const int NTile = 1; // TODO

    // allocate outFeature ?
    // TODO: non subm
    torch::Tensor outFeature =
        torch::zeros({feature.size(0), oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    dim3 gridSize = dim3(NTile, divUp(iC, IC_BLOCK), divUp(oC, OC_BLOCK));
    dim3 blockSize = dim3(VOX_BLOCK, 1, OC_BLOCK);

    // global version, with no shared memory
    ruleConvKernel<int32_t, float, VOX_BLOCK, IC_BLOCK, OC_BLOCK >
    <<<gridSize, blockSize>>>(
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        kernelVolume, iC, oC);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return outFeature;
}

} // namespace sphconv
