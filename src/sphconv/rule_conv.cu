#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "default_conv.cu.h"
#include "rule_conv_kernel.cu.h"

#include "debug_utils.h"
#include "timer.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

const int VBLOCK = 16;

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

// struct ConvBase {


//     virtual Status initialize(Arguments const& args){};

//     virtual Status run(cudaStream_t stream=nullptr){};

//     virtual ~ConvBase() = default;
// };

template <
    int VBLOCK,
    // output channel size
    int Co_BLOCK,
    // input channel size
    int Ci_BLOCK = 8>
struct Conv  {

    using ConvKernel = typename kernel::DefaultConv<VBLOCK, Co_BLOCK, Ci_BLOCK>::ConvKernel;

    static size_t get_workspace_size()
    {
        // TODO: use cudaMemsetAsync(workspace, 0, bytes, stream)
        return 0;
    }

    struct Arguments {
        torch::Tensor feature;    //  [NNZ, iC]
        torch::Tensor weight;     // [kernelVolume, iC, oC]
        torch::Tensor localRules; //  [NTile, kernelVolume, 2, NNZ ],
        torch::Tensor ruleSize;   // [Ntile, kernelVolume]
        torch::Tensor outFeature; // [outNNZ, oC]

        Arguments() {}

        Arguments(
            const torch::Tensor& feature_,
            const torch::Tensor& weight_,
            const torch::Tensor& localRules_,
            const torch::Tensor& ruleSize_,
            const torch::Tensor& outFeature_)
            : feature(feature_),
              weight(weight_),
              localRules(localRules_),
              ruleSize(ruleSize_),
              outFeature(outFeature_)
        {
        }
    };

private:
    typename ConvKernel::Params params_;
    int NTile_;

public:
    /// Constructs the Conv
    Conv() {
        printf("how do you do ?\n");
    }

    Status initialize(Arguments const& args)
    {
        NTile_ = args.ruleSize.size(0);
        int kernelVolume = args.weight.size(0);

        params_ = typename ConvKernel::Params(
            args.feature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            args.weight.template packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            args.localRules.template packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            args.ruleSize.template packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            args.outFeature.template packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            kernelVolume);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr)
    {
        dim3 grid(NTile_);
        dim3 block(ConvKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
        printf("smem_size = %d\n", smem_size);

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
        return Status::kSuccess ;
    }
};

torch::Tensor
rule_conv(torch::Tensor feature,     //  [NNZ, C]
          torch::Tensor weight,      // [kernelVolume, iC, oC]
          torch::Tensor localRules,  //  [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize,    // [Ntile, kernelVolume]
          torch::Tensor globalRules, // [NTile, 2, TILE_N_MAX]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape,
          int outNNZ)
{
    printf(" conv begin\n");

    int iC = weight.size(1);
    int oC = weight.size(2);

    int IC_BLOCK = near2power(iC);
    int OC_BLOCK = near2power(oC);

    int NTile = ruleSize.size(0);

    // allocate outFeature ?
    torch::Tensor outFeature =
        torch::zeros({outNNZ, oC},
                     torch::dtype(feature.dtype()).device(feature.device()));


    using ConvKernel = Conv<32, 32>;
    // auto conv = Conv<16, 32>();

    printf(" conv constrcut\n");
    ConvKernel conv;

    // std::unique_ptr<ConvBase> conv;

    // switch (OC_BLOCK) {
    // case 8:
    //     // if oc = 8
    //     // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
    //     conv.reset(new Conv<16, 32>());
    //     break;
    // case 16:
    //     // if oc = 16
    //     // error: static assertion failed with "ThreadMap::Iterations::kColumn must be > 0"
    //     conv.reset(new Conv<16, 32>());
    //     break;
    // case 32:
    //     conv.reset(new Conv<16, 32>());
    //     break;
    // case 64:
    //     conv.reset(new Conv<16, 64>());
    //     break;
    // default:
    //     printf("unsupported oC = %d\n", oC);
    // }

    printf(" args init\n ");
    typename ConvKernel::Arguments args(
        feature,
        weight,
        localRules,
        ruleSize,
        outFeature);

    printf(" conv init\n");
    conv.initialize(args);

    printf(" conv run\n");
    conv.run();

    return outFeature;
}

} // namespace device

} // namespace sphconv
