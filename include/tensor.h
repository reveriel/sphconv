#pragma once

#include <torch/extension.h>
#include <vector>

namespace sphconv
{

std::vector<torch::Tensor> init_tensor(
    const torch::Tensor feature,        // [NNZ, C]
    const torch::Tensor indicesBZYX,    // [NNZ, 4]
    int batchSize,                      //
    std::vector<int64_t> spatialShape); // H, W, D

torch::Tensor init_tensor_backward(
    const torch::Tensor d_featureOut, // [NNZ, C]
    const torch::Tensor permutation); // [NNZ]

torch::Tensor to_dense(
    const torch::Tensor feature,  // [NNZ, C]
    const torch::Tensor zIndices, // [NNZ]
    const torch::Tensor zPtr,     // [B, H, W]
    int D,
    torch::Tensor out);

torch::Tensor to_dense_backward(
    const torch::Tensor d_featureOut,
    const torch::Tensor zIndices,
    const torch::Tensor zPtr);

} // namespace sphconv
