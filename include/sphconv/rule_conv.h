#pragma once

#include <torch/extension.h>
#include <vector>

namespace sphconv
{

namespace device
{

torch::Tensor
rule_conv(const torch::Tensor feature,  // [NNZ, C]
          const torch::Tensor weight,   // [kernelVolume, iC, oC]
          const torch::Tensor rules,    // [NTile, kernelVolume, 2, NNZ ],
          const torch::Tensor ruleSize, // [Ntile, kernelVolume]
          int outNNZ);

std::vector<torch::Tensor>
rule_conv_backward(const torch::Tensor d_featureOut, // [outtNNZ, oC]
                   const torch::Tensor feature,      // [NNZ, iC]
                   const torch::Tensor weight,       // [kernelVolume, iC, oC]
                   const torch::Tensor rules,        // [NTile, kernelVolume, 2, NNZ ],
                   const torch::Tensor ruleSize,     // [Ntile, kernelVolume]
                   std::vector<int64_t> tile_grid_shape);

torch::Tensor
rule_conv_d_feature(const torch::Tensor feature,  // [NNZ, C]
                    const torch::Tensor weight,   // [kernelVolume, iC, oC]
                    const torch::Tensor rules,    // [NTile, kernelVolume, 2, nnz_max],
                    const torch::Tensor ruleSize, // [Ntile, kernelVolume]
                    std::vector<int64_t> tile_grid_shape,
                    int outNNZ);

} // namespace device

} // namespace sphconv
