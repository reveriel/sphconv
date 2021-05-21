#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <mma.h>

#include "iterator.cu.h"

#include "cutlass/cutlass.h"
#include "cutlass/wmma_array.h"
#include "cutlass/device_kernel.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_wmma.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/warp/tile_iterator_wmma_tensor_op.h"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"


#include "timer.h"
#include "debug_utils.h"

namespace sphconv
{

using cutlass::AlignedBuffer;
using cutlass::MatrixShape;

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;


namespace threadblock
{


template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
    >
struct DefaultMma;




/// Specialization for Wmma TensorOp operator with 2 staged pipeline
template <
    ///< Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  cutlass::arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassSimt, 2, Operator>;

  static int const kThreads = MmaCore::kThreads;
  using Shape = typename MmaCore::Shape; // gemmshape

  using IteratorThreadMapA =
        sphconv::threadblock::InThreadMap<
            cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
            kThreads,
            1>;
  // Define iterators over tiles from the A operand
  using IteratorA =
      sphconv::threadblock::InputTileIterator<
        ElementA, IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//       ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
//       kAlignmentB>;


    using IteratorThreadMapB =
            sphconv::threadblock::InThreadMap<
                cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
                kThreads,
                1>;

    // we customized the threadMap, of defaultMmaCore
    using IteratorB =
        sphconv::threadblock::KernelTileIterator<
            ElementB, IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped singlestage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

// template <
//     typename Shape_,
//     typename WarpMmaTensorOp_,
//     typename OutputOp_,
//     int ElementsPerAccess>
// struct DefaultEpilogueVoltaTensorOp
// {
//     using Shape = Shape_;
//     using WarpMmaTensorOp = WarpMmaTensorOp_;
//     using OutputOp = OutputOp_;
//     static int const kElementsPerAccess = ElementsPerAccess;

//     using ElementOutput = typename OutputOp::ElementOutput;
//     using LayoutC = typename WarpMmaTensorOp::LayoutC;
//     using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

//     using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
//         Shape,
//         typename WarpMmaTensorOp::Shape,
//         1,
//         ElementOutput,
//         kElementsPerAccess,
//         ElementAccumulator>::Type;

//     // using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
//     //     OutputTileThreadMap,
//     //     ElementOutput>;
//     using OutputTileIterator = sphconv::threadblock::OutTileIterator<
//             OutputTileThreadMap,
//             ElementOutput>;

//     using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
//         typename WarpMmaTensorOp::Shape,
//         cutlass::gemm::GemmShape<32, 32, 4>,
//         ElementAccumulator,
//         LayoutC>;

//     using WarpTileIterator = cutlass::epilogue::warp::TileIteratorVoltaTensorOp<
//         typename WarpMmaTensorOp::Shape,
//         cutlass::gemm::GemmShape<32, 32, 4>,
//         ElementAccumulator,
//         LayoutC>;

//     static int const kSharedMemAlignment = cutlass::sizeof_bits<ElementAccumulator>::value * WarpTileIterator::kElementsPerAccess / 8;

//     static_assert(kSharedMemAlignment == 8, "Shared memory alignment must be 8B");

//     using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
//         typename OutputTileThreadMap::CompactedThreadMap,
//         ElementAccumulator,
//         kSharedMemAlignment>;

//     /// Hard-coded padding elements added
//     using Padding = typename WarpTileIterator::Padding;

//     //
//     // Define the epilogue
//     //
//     using Epilogue = cutlass::epilogue::threadblock::Epilogue<
//         Shape,
//         WarpMmaTensorOp,
//         1,
//         OutputTileIterator,
//         AccumulatorFragmentIterator,
//         WarpTileIterator,
//         SharedLoadIterator,
//         OutputOp,
//         Padding>;
// };


template <
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueSimt {
    using Shape = Shape_;
    using WarpMmaSimt = WarpMmaSimt_;
    using OutputOp = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

    using ElementOutput = typename OutputOp::ElementOutput;
    using LayoutC = typename WarpMmaSimt::LayoutC;
    using ElementAccumulator = typename WarpMmaSimt::ElementC;

    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
        Shape,
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::Policy,
        kPartitionsK,
        ElementOutput,
        kElementsPerAccess
    >::Type;

    // using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //     OutputTileThreadMap,
    //     ElementOutput
    // >;

    using OutputTileIterator = sphconv::threadblock::OutTileIterator<
        OutputTileThreadMap,
        ElementOutput
    >;

    using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy
    >;

    using WarpTileIterator = cutlass::epilogue::warp::TileIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        ElementAccumulator,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy
    >;

    using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
        typename OutputTileThreadMap::CompactedThreadMap,
        ElementAccumulator
    >;

    /// Hard-coded padding elements added
    using Padding = typename WarpTileIterator::Padding;

    //
    // Define the epilogue
    //
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        Shape,
        WarpMmaSimt,
        kPartitionsK,
        OutputTileIterator,
        AccumulatorFragmentIterator,
        WarpTileIterator,
        SharedLoadIterator,
        OutputOp,
        Padding
    >;

};


} // namespace threadblock

namespace kernel
{

template <
    typename Mma_,
    typename Epilogue_>
struct Conv
{

    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using WarpCount = typename Mma::WarpCount;
    using ElementA = typename Mma::IteratorA::Element;
    using ElementB = typename Mma::IteratorB::Element;
    using ElementD = ElementB;
    using Index = int;
    static int const kThreadCount = 32 * WarpCount::kCount;

    struct Params
    {
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorB::Params params_B;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename OutputOp::Params output_op;
        int kernel_volume_;
        int in_channel_;
        const GpuTensor<Index, 2> &ruleSize_;

        // CUTLASS_HOST_DEVICE
        Params(torch::Tensor feature,
               torch::Tensor weight,
               torch::Tensor localRules,
               torch::Tensor ruleSize,
               torch::Tensor outFeature,
               int kernel_volume,
               typename OutputOp::Params output_op = typename OutputOp::Params())
            : params_A(feature, localRules, ruleSize),
                params_B(weight),
                params_D(outFeature, localRules, ruleSize),
                output_op(output_op),
                in_channel_(weight.size(1)),
                kernel_volume_(kernel_volume),
                ruleSize_(ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()) {}
    };

    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_HOST_DEVICE
    Conv() {}

    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage)
    {
        const int VBLOCK = 32;

        int tile = blockIdx.x;

        // Problem size is a function of threadblock index in the K dimension
        // int problem_size_k = min(
        // params.problem_size.k(),
        // (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

        // Compute threadblock-scoped matrix multiply-add
        // int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
        int gemm_k_iterations = params.in_channel_ / Mma::Shape::kK;


        for (int k = 0; k < params.kernel_volume_; k++)
        {
            int kRuleSize = params.ruleSize_[tile][k];
            if (kRuleSize == 0)
                continue;

            for (int vbegin = 0; vbegin < kRuleSize; vbegin += VBLOCK)
            {

                int thread_idx = threadIdx.x;
                // Construct iterators
                typename Mma::IteratorA iterator_A(params.params_A, thread_idx, tile, vbegin, k);

                typename Mma::IteratorB iterator_B(params.params_B, thread_idx, k);

                // Broadcast the warp_id computed by lane 0 to ensure dependent code
                // is compiled as warp-uniform.
                int warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
                int lane_id = threadIdx.x % 32;

                Mma mma(shared_storage.main_loop, thread_idx, warp_id, lane_id);

                typename Mma::FragmentC accumulators;
                accumulators.clear();

                // threadblock-scoped
                mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

                OutputOp output_op(params.output_op);

                typename Epilogue::OutputTileIterator iterator_D(params.params_D, tile, vbegin, thread_idx, k);

                Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_id, lane_id);

                epilogue(output_op, iterator_D, accumulators, iterator_D);

            } // v block
        }     // k
    }
};

} // namespace kernel



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



struct Arguments {

};

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

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = ElementC;
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static int const kStages = 2;

    // using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    //     ElementC,
    //     128 / cutlass::sizeof_bits<ElementC>::value,
    //     ElementAccumulator,
    //     ElementAccumulator
    // >;
    using EpilogueOutputOp = cutlass::epilogue::thread::Convert<
        ElementC, 1, ElementAccumulator>;

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

    cudaError_t result;
    int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
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
        feature,
        weight,
        localRules,
        ruleSize,
        outFeature,
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
// // #define myKernel  ruleConvKernelDynamic<int32_t, float, TILE_N_MAX, 32, 8>
// //             cudaFuncSetAttribute(myKernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
// //             cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
// //             myKernel<<<gridSize, dim3(32, 8, 1), maxbytes>>>(PARAMS);
// // #undef myKernel
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
