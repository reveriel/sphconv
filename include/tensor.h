#pragma once

#include <vector>
#include <torch/extension.h>

namespace sphconv
{

    torch::Tensor init_tensor(
        const torch::Tensor feature,       // [NNZ, C]
        const torch::Tensor indicesZYX,    // [NNZ, 4]
        int batchSize,                     //
        std::vector<int64_t> spatialShape, // H, W, D
        torch::Tensor outFeature,          // [NNZ, C]
        torch::Tensor zIndices);           //

    torch::Tensor to_dense(
        const torch::Tensor feature,  // [NNZ, C]
        const torch::Tensor zIndices, // [NNZ]
        const torch::Tensor zPtr,     // [B, H, W]
        int D,
        torch::Tensor out);

}