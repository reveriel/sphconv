
#pragma once

#include "cutlass/cutlass.h"
#include "sphconv/sphconv.h"

using cutlass::CUTLASS_DEVICE;
namespace sphconv
{

namespace threadblock
{

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX() {
  return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY() {
  return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
  return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX() {
  return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY() {
  return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
  return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY() {
  return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ() {
  return blockDim.z;
}

struct BatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  BatchedTileCoord get_tiled_Shape(
    BatchedTileCoord problem_size,
    BatchedTileCoord tile_size,
    int batch_count) const {
      return BatchedTileCoord(
        (problem_size.h() + tile_size.h() - 1) / tile_size.h(),
        (problem_size.w() + tile_size.w() - 1) / tile_size.w(),
        (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
    }



  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(BatchedTileCoord tiled_shape) const {
    return dim3(tiled_shape.h(), tiled_shape.w(), tiled_shape.batch());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  BatchedTileCoord get_tile_offset() const {
    return BatchedTileCoord{
      RematerializeBlockIdxX(),
      RematerializeBlockIdxY(),
      RematerializeBlockIdxZ()
    };
  }

  /// Gets the batch tile index
  CUTLASS_DEVICE
  int get_batch_tile_idx() const {
    return RematerializeBlockIdxZ();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockDimZ()*RematerializeBlockIdxZ() + RematerializeThreadIdxZ();
  }

  CUTLASS_DEVICE
  int get_batch_tile_idx() const {
    return RematerializeBlockIdxY();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockDimY()*RematerializeBlockIdxY() + RematerializeThreadIdxY();
  }

};

} // namespace threadblock


} // namespace sphconv