#pragma once

#include "sphconv/indice_conv.h"
#include "sphconv/sphconv.h"

namespace sphconv {

namespace device {

using cutlass::Status;

// datatype
// X, RuleMap
template <
/// Element type for features
typename ElementFeature_,
/// Element type for RuleMaps
typename ElementRuleMap_,
/// Element type for Kernel
typename ElementKernel_,
/// Element type for NumIn,
typename ElementNumIn_
>
class IndiceConv {
    public:
    using ElementFeature = ElementFeature_;
    using ElementRuleMap = ElementRuleMap_;
    using ElementKernel = ElementKernel_;
    using ElementNumIn = ElementNumIn_;

    // tile
    using ThreadblockShape = TileShape<8, 8>; // ???
    /// TODO: how to decide this value?
    ///  what does this mean ?
    /// dynamic ? or choosing a better one based on depth of layer ?
    // gemm
    using WarpShape = GemmShape<32, 64, 8>; ///
    // gemm
    using InstructionShape = GemmShape<1, 1, 1>;

    using ThreadblockSwizzle = threadblock::BatchedIdentityThreadblockSwizzle;

    using ConvKernel = typename kernel::IndiceConv<
      ElementFeature,
      ElementRuleMap,
      ElementKernel,
      ElementNumIn,
      ThreadblockShape,
      WarpShape,
      InstructionShape>ConvKernel;

    // IndiceConv () {}

    struct Arguments
    {
      // N, T, H, W, C_in, C_out, K_a, K_b, K_c

      // Indice problem_size[];


      // TODO: substitue torch's tensor accessor (nessisarry?)
      const TorchTensor<ElementFeature, 5> feature;
            TorchTensor<ElementFeature, 5> new_feature;
      const TorchTensor<ElementRuleMap, 5> InRuleMap;
      const TorchTensor<ElementRuleMap, 5> OutRuleMap;
      const TorchTensor<ElementNumIn, 4> NumIn;
      const TorchTensor<ElementKernel, 5> weight;
      /// batchsize
      int N;
      int in_channels;
      int out_channels;
      int KD, KH, KW;
      int kernel_volume;
      int sH, sW;
      int padH, padW;
      int dH, dW;
      int oH, oW;
      int H, W;
      int oT;

      CUTLASS_HOST_DEVICE
      Arguments() { }

      CUTLASS_HOST_DEVICE
      Arguments(
        torch::Tensor feature_,
        torch::Tensor weight_,
        torch::Tensor InRuleMap_,
        torch::Tensor OutRuleMap_,
        torch::Tensor NumIn_,
        //  torch::Tensor bias,
        int oT_,
        int sD_, int sH_, int sW_,
        int padD_, int padH_, int padW_,
        int dD_, int dH_, int dW_,
        int groups_)
      ):
      feature(feature_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      new_feature(new_feature_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      InRuleMap(InRuleMap_.packed_accessor32<int32_t, 5, RestrictPtrTraits>()),
      OutRuleMap(OutRuleMap_.packed_accessor32<int32_t, 5, RestrictPtrTraits>()),
      NumIn(NumIn_.packed_accessor32<int32_t, 4, RestrictPtrTraits>()),
      weight(weight_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      N(feature.size(0)),
      in_channels(weight.size(1)),
      out_channels(weight.size(0)),
      KD(weight.size(2)),
      KH(weight.size(3)),
      KW(weight.size(4)),
      kernel_volume(KD * KH * KW),
      oT(oT_), sH(sH_), sW(sW_), padH(padH_), padW(padW_), dH(dH_), dW(dW_),
      H(feature.size(3)), W(feature.size(4)) { }

    };

  private:

  typename Kernel::IndiceConv::Params params_;

  public:

    void initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr)
    {
      // Determin grid shape
      ThreadblockSwizzle threadblock_swizzle;

      TileCoord grid_shape = threadblock_swizzle.get_tiled_shape(

      );

        params_ = typename ConvKernel::Params{


        };

    }

    Status run(cudaStream_t stream = nullptr)
    {
      ThreadblockSwizzle threadblock_swizzle;

      dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
      dim3 block(ConvKernel::kThreadCount, 1, 1);

      cudaError_t result;

      int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
      if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<ConvKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          Kernel<ConvKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        printf("erro\n");
        // return;
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
    }
  };


} // namespace device

} // namespace sphconv