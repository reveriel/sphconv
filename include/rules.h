#pragma once

#include <vector>
#include <torch/extension.h>

namespace sphconv {

// std::vector<torch::Tensor>
// get_rules(torch::Tensor depth,
//                  torch::Tensor thick,
//                  int D,
//                  int KD, int KH, int KW,
//                  int sD, int sH, int sW,
//                  int padD, int padH, int padW,
//                  int dD, int dH, int dW,
//                  int groups);


std::vector<torch::Tensor>
get_rules_subm(torch::Tensor zIndices,               //  [NNZ]
                torch::Tensor zPtr,                   // [B, H, W]
                torch::Tensor grid,                   // [B, H, W, D]
                int batchSize,
                std::vector<int64_t> spatialShape,    // H, W, D
                std::vector<int64_t> outSpatialShape, // H, W, D
                std::vector<int64_t> kernelSize,
                std::vector<int64_t> stride,
                std::vector<int64_t> padding,
                std::vector<int64_t> dilation);


} // namespace sphconv
