#pragma once

#include <vector>
#include <torch/extension.h>

namespace sphconv {

torch::Tensor
rule_conv(torch::Tensor feature,  //  [NNZ, C]
          torch::Tensor weight,   // [kernelVolume, iC, oC]
          torch::Tensor localRules,    //  [NTile, kernelVolume, 2, NNZ ],
          torch::Tensor ruleSize, // [Ntile, kernelVolume]
          torch::Tensor globalRules, // [NTile, NMax]
          int batchSize,
          std::vector<int64_t> spatialShape, // H, W, D
          std::vector<int64_t> outSpatialShape,
          int outNNZ
);

}
