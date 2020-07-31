#pragma once
#include <torch/extension.h>

namespace sphconv {
torch::Tensor
to_dense(torch::Tensor feature,
         torch::Tensor depth,
         torch::Tensor thick,
         int D);

torch::Tensor
to_dense_backward(torch::Tensor d_featureOut,
                  torch::Tensor depth,
                  torch::Tensor thick,
                  int T);

} // namespace sphconv