#pragma once

#include <torch/extension.h>
#include <vector>

namespace sphconv
{

namespace device
{

torch::Tensor
rule_conv(torch::Tensor feature,  // [NNZ, C]
          torch::Tensor weight,   // [kernelVolume, iC, oC]
          torch::Tensor rules,    // [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize, // [Ntile, kernelVolume]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape,
          int outNNZ);

std::vector<torch::Tensor>
rule_conv_backward(torch::Tensor d_featureOut, // [outtNNZ, oC]
                   torch::Tensor feature,      // [NNZ, iC]
                   torch::Tensor weight,       // [kernelVolume, iC, oC]
                   torch::Tensor rules,        // [NTile, kernelVolume, 2, NNZ ],
                   torch::Tensor ruleSize,     // [Ntile, kernelVolume]
                   int batchSize,
                   std::vector<int64_t> spatialShape, // H, W, D
                   std::vector<int64_t> outSpatialShape);

} // namespace device

} // namespace sphconv
