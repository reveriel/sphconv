#include "cutlass/gemm/gemm.h"
#include "debug_utils.h"
#include "default_conv.cu.h"
#include "rule_conv_kernel.cu.h"
#include "threadblock_swizzle.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;
using cutlass::gemm::GemmShape;

int near2power(int num)
{
    if (num <= 8)
        return 8;
    if (num <= 16)
        return 16;
    if (num <= 32)
        return 32;
    if (num <= 64)
        return 64;
    if (num <= 128)
        return 128;
    printf(" channel size of %d is too big\n", num);
    exit(-1);
    return 0;
}

namespace device
{

using cutlass::Status;

struct ConvBase {

    struct Arguments {
        torch::Tensor feature;    //  [NNZ, iC]
        torch::Tensor weight;     // [kernelVolume, iC, oC]
        torch::Tensor rules;      //  [NTile, kernelVolume, 2, NNZ ],
        torch::Tensor ruleSize;   // [Ntile, kernelVolume]
        torch::Tensor outFeature; // [outNNZ, oC]
        std::vector<int64_t> tile_grid_shape;

        Arguments() {}

        Arguments(
            const torch::Tensor& feature_,
            const torch::Tensor& weight_,
            const torch::Tensor& rules_,
            const torch::Tensor& ruleSize_,
            const torch::Tensor& outFeature_,
            const std::vector<int64_t>& tile_grid_shape)
            : feature(feature_),
              weight(weight_),
              rules(rules_),
              ruleSize(ruleSize_),
              outFeature(outFeature_),
              tile_grid_shape(tile_grid_shape)
        {
        }
    };

    virtual Status initialize(Arguments const& args) { return Status::kSuccess; };

    virtual Status run(cudaStream_t stream = nullptr) { return Status::kSuccess; };

    virtual ~ConvBase() = default;
};

template <
    /// GemmShape, V, oC, iC
    typename ThreadBlockShape_,
    /// GemmShape, V, oC, iC
    typename WarpShape_,
    int VBLOCK,
    ///
    typename ThreadblockSwizzle_ =
        typename threadblock::IdentityThreadblockSwizzle>
struct Conv : public ConvBase {

    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using ConvKernel = typename kernel::DefaultConv<
        ThreadBlockShape_, WarpShape_, VBLOCK, ThreadblockSwizzle_>::ConvKernel;

    static size_t get_workspace_size()
    {
        // TODO: use cudaMemsetAsync(workspace, 0, bytes, stream)
        // might be useful when used in multiple streams
        return 0;
    }

private:
    typename ConvKernel::Params params_;
    int NTile_;

public:
    /// Constructs the Conv
    Conv()
    {
        // printf(" kWarpGemmIterations = %d \n", ConvKernel::Mma::kWarpGemmIterations);
    }

    Status initialize(Arguments const& args) override
    {
        NTile_ = args.ruleSize.size(0);
        int kernelVolume = args.weight.size(0);

        params_ = typename ConvKernel::Params(
            args.feature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            args.weight.template packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            args.rules.template packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            args.ruleSize.template packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            args.outFeature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            kernelVolume);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr) override
    {
        ThreadblockSwizzle ts;

        dim3 grid = ts.get_grid_shape(NTile_);
        dim3 block(ConvKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
        // printf("smem_size = %d\n", smem_size);

        if (smem_size >= (48 << 10)) {
            printf("info: use 48KB more SMEM\n");
            result = cudaFuncSetAttribute(cutlass::Kernel<ConvKernel>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size);

            if (result != cudaSuccess) {
                printf(" error, cudaFuncSetAttribute, dynam");
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(cutlass::Kernel<ConvKernel>,
                                          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                printf(" error, cudaFuncSetAttribute, carveout");
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // result = cudaGetLastError();

        // return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
        return Status::kSuccess;
    }
};

/**
* partial specilization for d_feature
*/
template <
    /// GemmShape, V, oC, iC
    typename ThreadBlockShape_,
    /// GemmShape, V, oC, iC
    typename WarpShape_,
    int VBLOCK>
struct Conv<ThreadBlockShape_, WarpShape_, VBLOCK, threadblock::InterleavedThreadblockSwizzle>
    : public ConvBase {

    using ThreadblockSwizzle = threadblock::InterleavedThreadblockSwizzle;
    using ConvKernel = typename kernel::DefaultConv<
        ThreadBlockShape_, WarpShape_, VBLOCK, ThreadblockSwizzle>::ConvKernel;

    static size_t get_workspace_size()
    {
        // TODO: use cudaMemsetAsync(workspace, 0, bytes, stream)
        // might be useful when used in multiple streams
        return 0;
    }

private:
    typename ConvKernel::Params params_;
    int NTile_;
    int tile_grid_h_;
    int tile_grid_w_;

public:
    /// Constructs the Conv
    Conv()
    {
        // printf(" kWarpGemmIterations = %d \n", ConvKernel::Mma::kWarpGemmIterations);
    }

    Status initialize(Arguments const& args) override
    {
        NTile_ = args.ruleSize.size(0);
        tile_grid_h_ = args.tile_grid_shape[0];
        tile_grid_w_ = args.tile_grid_shape[1];
        int kernelVolume = args.weight.size(0);

        params_ = typename ConvKernel::Params(
            args.feature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            args.weight.template packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            args.rules.template packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            args.ruleSize.template packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            args.outFeature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            kernelVolume, tile_grid_h_, tile_grid_w_);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr) override
    {
        ThreadblockSwizzle ts;

        dim3 grid = ts.get_grid_shape(NTile_, tile_grid_h_, tile_grid_w_);
        dim3 block(ConvKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
        // printf("smem_size = %d\n", smem_size);

        if (smem_size >= (48 << 10)) {
            printf("info: use 48KB more SMEM\n");
            result = cudaFuncSetAttribute(cutlass::Kernel<ConvKernel>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size);

            if (result != cudaSuccess) {
                printf(" error, cudaFuncSetAttribute, dynam");
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(cutlass::Kernel<ConvKernel>,
                                          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                printf(" error, cudaFuncSetAttribute, carveout");
                return Status::kErrorInternal;
            }
        }

        for (int i = 0; i < 9; i++) {
            params_.update_tile_idx(i);
            cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
        // result = cudaGetLastError();
        // return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
        return Status::kSuccess;
    }
};



torch::Tensor
rule_conv(const torch::Tensor feature,  // [NNZ, C]
          const torch::Tensor weight,   // [kernelVolume, iC, oC]
          const torch::Tensor rules,    // [NTile, kernelVolume, 2, nnz_max],
          const torch::Tensor ruleSize, // [Ntile, kernelVolume]
          int outNNZ)
{
    int iC = weight.size(1);
    int oC = weight.size(2);

    int IC_BLOCK = near2power(iC);
    int OC_BLOCK = near2power(oC);

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    std::shared_ptr<ConvBase> conv;

    switch (OC_BLOCK) {
    case 8:
        // if oc = 8
        // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8>>();
        break;
    case 16:
        // if oc = 16
        // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8>>();
        break;
    case 32:
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8>>();
        break;
    case 64:
        conv = std::make_shared<Conv<GemmShape<8, 64, 8>, GemmShape<8, 32, 8>, 8>>();
        break;
    default:
        printf("unsupported oC = %d\n", oC);
    }

    ConvBase::Arguments args(feature, weight, rules, ruleSize, outFeature, {0,0});

    conv->initialize(args);

    conv->run();

    return outFeature;
}

torch::Tensor
rule_conv_d_feature(const torch::Tensor feature,  // [NNZ, C]
                    const torch::Tensor weight,   // [kernelVolume, iC, oC]
                    const torch::Tensor rules,    // [NTile, kernelVolume, 2, nnz_max],
                    const torch::Tensor ruleSize, // [Ntile, kernelVolume]
                    std::vector<int64_t> tile_grid_shape,
                    int outNNZ)
{
    CHECK_INPUT(feature);
    CHECK_INPUT(weight);
    CHECK_INPUT(rules);
    CHECK_INPUT(ruleSize);

    int iC = weight.size(1);
    int oC = weight.size(2);

    int IC_BLOCK = near2power(iC);
    int OC_BLOCK = near2power(oC);

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));

    std::shared_ptr<ConvBase> conv;

    switch (OC_BLOCK) {
    case 8:
        // if oc = 8
        // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8, threadblock::InterleavedThreadblockSwizzle>>();
        break;
    case 16:
        // if oc = 16
        // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8, threadblock::InterleavedThreadblockSwizzle>>();
        break;
    case 32:
        conv = std::make_shared<Conv<GemmShape<8, 32, 8>, GemmShape<8, 32, 8>, 8, threadblock::InterleavedThreadblockSwizzle>>();
        break;
    case 64:
        conv = std::make_shared<Conv<GemmShape<8, 64, 8>, GemmShape<8, 32, 8>, 8, threadblock::InterleavedThreadblockSwizzle>>();
        break;
    default:
        printf("unsupported oC = %d\n", oC);
    }

    ConvBase::Arguments args(feature, weight, rules, ruleSize, outFeature, tile_grid_shape);

    conv->initialize(args);

    conv->run();

    return outFeature;
}


std::vector<torch::Tensor>
rule_conv_backward(const torch::Tensor d_featureOut, // [outNNZ, oC]
                   const torch::Tensor feature,      // [NNZ, iC]
                   const torch::Tensor weight,       // [kernelVolume, iC, oC]
                   const torch::Tensor rules,        // [NTile, kernelVolume, 2, nnz_max],
                   const torch::Tensor ruleSize,     // [Ntile, kernelVolume]
                   std::vector<int64_t> tile_grid_shape)
{
    int kernelVolume = weight.size(0);
    int iC = weight.size(1);
    int oC = weight.size(2);

    int IC_BLOCK = near2power(iC);
    int OC_BLOCK = near2power(oC);

    int NNZ = feature.size(0);
    int NTile = ruleSize.size(0);

    // allocate d_feature
    // d_feature = d_featureOut * weight
    // TODO: weight [ic oc] to  [oc ic]
    torch::Tensor d_feature = rule_conv_d_feature(
        d_featureOut, weight, rules, ruleSize, tile_grid_shape, NNZ);

    torch::Tensor d_weight = torch::zeros(
        {kernelVolume, iC, oC}, torch::dtype(feature.dtype()).device(feature.device()));

    return {d_feature, d_weight};
}

} // namespace device

} // namespace sphconv
