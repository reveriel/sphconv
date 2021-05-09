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
          int V_MAX,
          int IC_BLOCK,
          int OC_BLOCK>
__global__ void ruleConvKernel(
    const GpuTensor<DType, 2> feature,
    const GpuTensor<DType, 3> weight, // [kernelVolume, iC, oC]
    const GpuTensor<IType, 4> localRules,  // [ NTile, kernelVolume, 2, NNZ']
    const GpuTensor<IType, 2> ruleSize,  // [NTile, kernelVolume]
    const GpuTensor<IType, 3> globalRules, // [NTile, 2, TILE_N_MAX]
    GpuTensor<DType, 2> outFeature,
    int kernelVolume,
    int iC, int oC)
{
    __shared__ DType input[V_MAX][IC_BLOCK];
    __shared__ DType output[V_MAX][OC_BLOCK];
    __shared__ DType subKernel[IC_BLOCK][OC_BLOCK];

    // for each NTile
    int tile = blockIdx.x;

    // load tile data to shared memory
    for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
        int global_in_idx = globalRules[tile][0][v];
        if (global_in_idx == -1)
            continue;

        for (int ic = threadIdx.y; ic < iC; ic += blockDim.y) {
            input[v][ic] = feature[global_in_idx][ic];
        }
    }

    for (int oc = threadIdx.y; oc < oC; oc += OC_BLOCK) {

        // set output to zeros
        for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
            output[v][oc % OC_BLOCK] = DType(0);
        }

        // for each kernelVolume * NNZ'
        for (int k = 0; k < kernelVolume; k++) {
            int kRuleSize = ruleSize[tile][k];

            // load kernel to shared memory
            for (int ic = threadIdx.x; ic < iC; ic += blockDim.x) {
                subKernel[ic][oc % OC_BLOCK] = weight[k][ic][oc]; // TODO : swap kernel oC iC order
            }
            __syncthreads();

            // matrix multiply
            for (int v = threadIdx.x; v < kRuleSize; v += blockDim.x) {
                int local_in_idx = localRules[tile][k][0][v];
                int local_out_idx = localRules[tile][k][1][v];

                DType sum = DType(0);

                for (int ic = 0; ic < iC; ic++) {
                    sum += input[local_in_idx][ic] * subKernel[ic][oc % OC_BLOCK];
                }

                output[local_out_idx][oc % OC_BLOCK] += sum;
            }
        } // k

        __syncthreads();
        // fill back to global output
        for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
            int global_out_idx = globalRules[tile][1][v];
            if (global_out_idx == -1)
                continue;
            outFeature[global_out_idx][oc] = output[v][oc % OC_BLOCK];
        }
        __syncthreads();
    } // oc
}


template <typename IType, typename DType,
          int V_MAX,
          int IC_BLOCK,
          int OC_BLOCK>
__global__ void ruleConvKernelDynamic(
    const GpuTensor<DType, 2> feature,
    const GpuTensor<DType, 3> weight, // [kernelVolume, iC, oC]
    const GpuTensor<IType, 4> localRules,  // [ NTile, kernelVolume, 2, NNZ']
    const GpuTensor<IType, 2> ruleSize,  // [NTile, kernelVolume]
    const GpuTensor<IType, 3> globalRules, // [NTile, 2, TILE_N_MAX]
    GpuTensor<DType, 2> outFeature,
    int kernelVolume,
    int iC, int oC)
{
    extern __shared__ DType shared[];

    DType (*input)[IC_BLOCK] = (DType (*) [IC_BLOCK]) &shared[0];
    DType (*output)[OC_BLOCK] = (DType (*) [OC_BLOCK])&shared[V_MAX * IC_BLOCK];
    DType (*subKernel)[OC_BLOCK] = (DType (*) [OC_BLOCK]) &shared[V_MAX * IC_BLOCK + V_MAX * OC_BLOCK];

    // for each NTile
    int tile = blockIdx.x;

    // load tile data to shared memory
    for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
        int global_in_idx = globalRules[tile][0][v];
        if (global_in_idx == -1)
            continue;

        for (int ic = threadIdx.y; ic < iC; ic += blockDim.y) {
            input[v][ic] = feature[global_in_idx][ic];
        }
    }

    for (int oc = threadIdx.y; oc < oC; oc += OC_BLOCK) {

        // set output to zeros
        for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
            output[v][oc % OC_BLOCK] = DType(0);
        }

        // for each kernelVolume * NNZ'
        for (int k = 0; k < kernelVolume; k++) {
            int kRuleSize = ruleSize[tile][k];

            // load kernel to shared memory
            for (int ic = threadIdx.x; ic < iC; ic += blockDim.x) {
                subKernel[ic][oc % OC_BLOCK] = weight[k][ic][oc]; // TODO : swap kernel oC iC order
            }
            __syncthreads();

            // matrix multiply
            for (int v = threadIdx.x; v < kRuleSize; v += blockDim.x) {
                int local_in_idx = localRules[tile][k][0][v];
                int local_out_idx = localRules[tile][k][1][v];

                DType sum = DType(0);

                for (int ic = 0; ic < iC; ic++) {
                    sum += input[local_in_idx][ic] * subKernel[ic][oc % OC_BLOCK];
                }

                output[local_out_idx][oc % OC_BLOCK] += sum;
            }
        } // k

        __syncthreads();
        // fill back to global output
        for (int v = threadIdx.x; v < globalRules.size(2); v += blockDim.x) {
            int global_out_idx = globalRules[tile][1][v];
            if (global_out_idx == -1)
                continue;
            outFeature[global_out_idx][oc] = output[v][oc % OC_BLOCK];
        }
        __syncthreads();
    } // oc
}



int near2power(int num) {
    if (num <= 8) return 8;
    if (num <= 16) return 16;
    if (num <= 32) return 32;
    if (num <= 64) return 64;
    if (num <= 128) return 128;
    printf(" channel size of %d is too big\n", num);
    exit(-1);
    return 0;
}

torch::Tensor
rule_conv(torch::Tensor feature,  //  [NNZ, C]
          torch::Tensor weight,   // [kernelVolume, iC, oC]
          torch::Tensor localRules,    //  [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize, // [Ntile, kernelVolume]
          torch::Tensor globalRules, // [NTile, 2, TILE_N_MAX]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape,
          int outNNZ)
{
    // cudaSharedMemConfig *cfg;
    // cudaDeviceGetSharedMemConfig()

    int kernelVolume = weight.size(0);
    int iC = weight.size(1);
    int oC = weight.size(2);

    int IC_BLOCK = near2power(iC);
    int OC_BLOCK = near2power(oC);

    const int VOX_BLOCK = 32;
    // const int OC_ILP = 4;

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    // dim3 gridSize = dim3(NTile, divUp(iC, IC_BLOCK), divUp(oC, OC_BLOCK));
    dim3 gridSize = dim3(NTile);
    dim3 blockSize = dim3(VOX_BLOCK, OC_BLOCK, 1);


    // kernel function parameters, all the same
#define PARAMS feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), \
               weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), \
               localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
               ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
               globalRules.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),\
               outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),\
               kernelVolume, iC, oC


    int carveout = 70;
    int maxbytes = 56 * 1024;
    switch (IC_BLOCK)
    {
    case 8:
        switch (OC_BLOCK)
        {
            // NOTE:  OC_BLOCK,   blockDim.y,  must match now
        case 8:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 8, 8><<<gridSize, dim3(32, 8, 1)>>>(PARAMS);
            break;
        case 16:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 8, 16><<<gridSize, dim3(32, 16, 1)>>>(PARAMS);
            break;
        default:
            printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
        }
        break;
    case 16:
        switch (OC_BLOCK)
        {
        case 16:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 16, 16><<<gridSize, dim3(32, 16, 1)>>>(PARAMS);
            break;
        case 32:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 16, 16><<<gridSize, dim3(32, 16, 1)>>>(PARAMS);
            break;
        default:
            printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
        }
        break;
    case 32:
        switch (OC_BLOCK)
        {
        case 32:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 32, 8><<<gridSize, dim3(32, 8, 1)>>>(PARAMS);
            break;
        case 64:
            ruleConvKernel<int32_t, float, TILE_N_MAX, 32, 8><<<gridSize, dim3(32, 8, 1)>>>(PARAMS);
            break;
        default:
            printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
        }
        break;
    case 64:
        switch (OC_BLOCK)
        {
        case 64:
#define myKernel  ruleConvKernelDynamic<int32_t, float, TILE_N_MAX, 64, 8>
            cudaFuncSetAttribute(myKernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
            cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
            myKernel<<<gridSize, dim3(32, 8, 1), maxbytes>>>(PARAMS);
#undef myKernel
            break;
        default:
            printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
        }
        break;
    default:
        printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return outFeature;
}

} // namespace sphconv
