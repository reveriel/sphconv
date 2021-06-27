
#pragma once

#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
// #include "cutlass/layout/matrix.h"
// #include "cutlass/platform/platform.h"
// #include "cutlass/gemm/gemm.h"

namespace sphconv
{
namespace threadblock
{

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX()
{
    return blockIdx.x;
}
/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

struct IdentityThreadblockSwizzle {
    CUTLASS_HOST_DEVICE
    IdentityThreadblockSwizzle() {}

    CUTLASS_DEVICE
    int get_tile_offset() const
    {
        int block_idx_x = RematerializeBlockIdxX();
        return block_idx_x;
    }

    CUTLASS_HOST_DEVICE
    dim3 get_grid_shape(int NTile) {
        return dim3(NTile, 1, 1);
    }

    CUTLASS_DEVICE
    int end() {
        return RematerializeBlockIdxX() + 1;
    }

    CUTLASS_DEVICE
    int next(int tile) {
        return tile + 1;
    }

    CUTLASS_DEVICE
    void sync() const
    {
        // empty
    }
};

// every threadblock do two tile, for testing
struct Identity2ThreadblockSwizzle {
    CUTLASS_HOST_DEVICE
    Identity2ThreadblockSwizzle() {}

    CUTLASS_DEVICE
    int get_tile_offset() const
    {
        int block_idx_x = RematerializeBlockIdxX() * 2;
        return block_idx_x;
    }

    CUTLASS_HOST_DEVICE
    dim3 get_grid_shape(int NTile) {
        return dim3((NTile + 1) / 2, 1, 1);
    }

    CUTLASS_DEVICE
    int end() {
        return RematerializeBlockIdxX() * 2 + 2;
    }

    CUTLASS_DEVICE
    int next(int tile) {
        return tile + 1;
    }

    CUTLASS_DEVICE
    void sync() const
    {
        // empty
    }
};


struct InterleavedThreadblockSwizzle{
    CUTLASS_HOST_DEVICE
    InterleavedThreadblockSwizzle() {}

    CUTLASS_DEVICE
    int get_tile_offset() const
    {

    }

    private:

};

} // namespace threadblock

} // namespace sphconv
