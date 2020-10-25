#pragma once

#include "cutlass/cutlass.h"
// #include "indice_conv/conv.h"

template<typename T, int DIM>
using TorchTensor = torch::PackedTensorAccessor32<T, DIM, RestrictPtrTraits>;







