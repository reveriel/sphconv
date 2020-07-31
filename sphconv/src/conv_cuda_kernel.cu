
#include "timer.h"
#include "debug_utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

using torch::RestrictPtrTraits;
// using namespace torch::indexing;

// input tile size = TILE + K
//
// __shared__ feature_tile[N][C][T][H_TILE][W_TILE];
// TODO: make T varadic length

// constexpr int T = 8;
// const int INPUT_TILE_H = 4;
// const int INPUT_TILE_W = 16;
// __shared__ scalar_t depth_tile[T][INPUT_TILE_H][INPUT_TILE_W];
// TODO
// __shared__ Index thickMap[INPUT_TILE_H][INPUT_TILE_W];
// __shared__ Index Num[3x3x3][INPUT_TILE_H][INPUT_TILE_W];

namespace sphconv {

const int H_BLOCK = 4, W_BLOCK = 8;





} // namespace sphconv
