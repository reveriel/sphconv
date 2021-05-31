#include <torch/extension.h>

#include "cutlass/aligned_buffer.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/wmma_array.h"

#include "debug_utils.h"
#include "timer.h"

namespace sphconv
{

namespace kernel
{

template <typename Mma_,      // threadblock level MMA
          typename Epilogue_> // threadblock level epilogue
struct Conv {

    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using WarpCount = typename Mma::WarpCount;
    using ElementA = typename Mma::IteratorA::Element;
    using ElementB = typename Mma::IteratorB::Element;
    using ElementD = ElementB;

    // using Index = int;
    static int const kThreadCount = 32 * WarpCount::kCount;
    static int const VBLOCK = 16;

    struct Params {
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorB::Params params_B;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename OutputOp::Params output_op;
        int kernel_volume_;
        int in_channel_;
        const GpuTensor<int32_t, 2> ruleSize_;

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<float, 2>& feature, const GpuTensor<float, 3>& weight,
               const GpuTensor<int32_t, 4>& localRules, const GpuTensor<int32_t, 2>& ruleSize,
               const GpuTensor<float, 2>& outFeature, int kernel_volume,
               typename OutputOp::Params output_op = typename OutputOp::Params(1, 1)) // acumulate on result, beta = 1
            : params_A(feature, localRules, ruleSize), params_B(weight), params_D(outFeature, localRules, ruleSize),
              output_op(output_op), in_channel_(weight.size(1)), kernel_volume_(kernel_volume), ruleSize_(ruleSize)
        {
        }
    };

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_HOST_DEVICE
    Conv() {}

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // const int VBLOCK = 32;

        int tile = blockIdx.x;

        // Problem size is a function of threadblock index in the K dimension
        // int problem_size_k = min(
        // params.problem_size.k(),
        // (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

        // Compute threadblock-scoped matrix multiply-add
        // int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
        int gemm_k_iterations = params.in_channel_ / Mma::Shape::kK;

        for (int k = 0; k < params.kernel_volume_; k++) {
            // printf(" rulesize [tile: %d] [k: %d] = ?\n", tile, k);
            int kRuleSize = params.ruleSize_[tile][k];

            // printf(" rulesize [tile: %d] [k: %d] = %d\n",  tile, k, kRuleSize );

            if (kRuleSize == 0)
                continue;

            for (int vbegin = 0; vbegin < kRuleSize; vbegin += VBLOCK) {

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

                // printf("&params.params_D = %p\n", &params.params_D);
                typename Epilogue::OutputTileIterator iterator_D(params.params_D, tile, vbegin, thread_idx, k);

                typename Epilogue::OutputTileIterator iterator_C = iterator_D;

                Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_id, lane_id);

                epilogue(output_op, iterator_D, accumulators, iterator_C);

            } // v block
        }     // k
    }
};

} // namespace kernel
} // namespace sphconv
