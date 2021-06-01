#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "default_mma.cu.h"
#include "default_epilogue.cu.h"
#include "rule_conv_kernel.cu.h"


#include "timer.h"
#include "debug_utils.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

const int VBLOCK = 16;


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

namespace device {



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

    // const int VOX_BLOCK = 32;
    // const int OC_ILP = 4;

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    // dim3 gridSize = dim3(NTile, divUp(iC, IC_BLOCK), divUp(oC, OC_BLOCK));

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = ElementC;
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    // v,  oC, iC
    using ThreadblockShape = cutlass::gemm::GemmShape<VBLOCK, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static int const kStages = 2;

    // using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    //     ElementC,
    //     128 / cutlass::sizeof_bits<ElementC>::value,
    //     ElementAccumulator,
    //     ElementAccumulator
    // >;
    // using EpilogueOutputOp = cutlass::epilogue::thread::Convert<
    //     ElementC, 1, ElementAccumulator>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,  1, ElementAccumulator>;

    using Mma = typename sphconv::threadblock::DefaultMma<
        ElementA,
        cutlass::layout::RowMajor,
        kAlignmentA,
        ElementB,
        cutlass::layout::RowMajor,
        kAlignmentB,
        ElementC,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50, ThreadblockShape, WarpShape, InstructionShape, kStages,
        Operator
        >::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

    // using Epilogue = typename sphconv::threadblock::DefaultEpilogueVoltaTensorOp<
    //     ThreadblockShape,
    //     typename Mma::Operator,
    //     EpilogueOutputOp,
    //     EpilogueOutputOp::kCount>::Epilogue;
    using Epilogue = typename sphconv::threadblock::DefaultEpilogueSimt<
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess
        >::Epilogue;

    using ConvKernel = kernel::Conv<Mma, Epilogue>;

    dim3 gridSize(NTile);
    dim3 blockSize(ConvKernel::kThreadCount, 1, 1);
    // dim3 blockSize(1, 1, 1);

    cudaError_t result;
    int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
    // printf("smem_size = %d\n", smem_size);

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<ConvKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
          printf(" error, cudaFuncSetAttribute, dynam"); exit(-1);
        // return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          cutlass::Kernel<ConvKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
          printf(" error, cudaFuncSetAttribute, carveout"); exit(-1);
        // return Status::kErrorInternal;
      }
    }

    typename ConvKernel::Params params_(
        feature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        kernelVolume);

    cutlass::Kernel<ConvKernel><<<gridSize, blockSize, smem_size, nullptr>>>(params_);

    // dim3(VOX_BLOCK, OC_BLOCK, 1);

    // kernel function parameters, all the same

//     int carveout = 70;
//     int maxbytes = 56 * 1024;
//     switch (IC_BLOCK)
//     {
//     case 8:
//         switch (OC_BLOCK)
//         {
//             // NOTE:  OC_BLOCK,   blockDim.y,  must match now
//         case 8:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 8, 8><<<gridSize, dim3(64, 8, 1)>>>(PARAMS);
//             break;
//         case 16:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 8, 8><<<gridSize, dim3(64, 8, 1)>>>(PARAMS);
//             break;
//         default:
//             printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
//         }
//         break;
//     case 16:
//         switch (OC_BLOCK)
//         {
//         case 16:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 16, 16><<<gridSize, dim3(16, 16, 1)>>>(PARAMS);
//             break;
//         case 32:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 16, 32><<<gridSize, dim3(16, 32, 1)>>>(PARAMS);
//             break;
//         default:
//             printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
//         }
//         break;
//     case 32:
//         switch (OC_BLOCK)
//         {
//         case 32:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 32, 32><<<gridSize, dim3(32, 32, 1)>>>(PARAMS);
//             break;
//         case 64:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 32, 32><<<gridSize, dim3(32, 32, 1)>>>(PARAMS);
//             break;
//         default:
//             printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
//         }
//         break;
//     case 64:
//         switch (OC_BLOCK)
//         {
//         case 64:
//             ruleConvKernel<int32_t, float, TILE_N_MAX, 32, 32><<<gridSize, dim3(32, 32, 1)>>>(PARAMS);
//             break;
//         default:
//             printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
//         }
//         break;
//     default:
//         printf("not support ic ocblock %d, %d\n", IC_BLOCK, OC_BLOCK);
//     }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return outFeature;
}

} // namespace device

} // namespace sphconv
