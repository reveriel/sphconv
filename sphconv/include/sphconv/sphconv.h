#pragma once

#include "torch/extension.h"
// #include "indice_conv/conv.h"

namespace sphconv {

// for GPU access, with stride information
template <typename T, int DIM>
using TorchTensor = torch::PackedTensorAccessor32<T, DIM, torch::RestrictPtrTraits>;

} // namespace sphconv