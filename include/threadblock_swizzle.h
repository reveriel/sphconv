#pragma once

#include "cutlass/cutlass.h"
#include "debug_utils.h"
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

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY()
{
    return blockIdx.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX()
{
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
    dim3 get_grid_shape(int NTile)
    {
        return dim3(NTile, 1, 1);
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
    dim3 get_grid_shape(int NTile)
    {
        return dim3((NTile + 1) / 2, 1, 1);
    }
};

// struct InterleavedThreadblockSwizzle {
//     CUTLASS_HOST_DEVICE
//     InterleavedThreadblockSwizzle() {}

//     CUTLASS_DEVICE
//     int get_tile_offset(int i, int NTile, int tile_grid_h, int tile_grid_w) const
//     {
//         int x, y;
//         switch (i) {
//         case 0:
//             x = RematerializeBlockIdxX() * 2;
//             y = RematerializeBlockIdxY() * 2;
//             break;
//         case 1:
//             x = RematerializeBlockIdxX() * 2;
//             y = RematerializeBlockIdxY() * 2 + 1;
//             break;
//         case 2:
//             x = RematerializeBlockIdxX() * 2 + 1;
//             y = RematerializeBlockIdxY() * 2;
//             break;
//         case 3:
//             x = RematerializeBlockIdxX() * 2 + 1;
//             y = RematerializeBlockIdxY() * 2 + 1;
//             break;
//         default:
//             break;
//         }
//         if (x < tile_grid_h && y < tile_grid_w) {
//             return linearIdx(x, y, tile_grid_w);
//         } else {
//             return NTile;
//         }
//     }

//     CUTLASS_HOST_DEVICE
//     dim3 get_grid_shape(int NTile, int tile_grid_h, int tile_grid_w)
//     {
//         return dim3(((tile_grid_h + 1) / 2), (tile_grid_w + 1) / 2, 1);
//     }
// };


struct InterleavedThreadblockSwizzle {
    CUTLASS_HOST_DEVICE
    InterleavedThreadblockSwizzle() {}

    CUTLASS_DEVICE
    int get_tile_offset(int i, int NTile, int tile_grid_h, int tile_grid_w) const
    {
        return i;
        int x, y;
        switch (i) {
        case 0:
            x = RematerializeBlockIdxX() * 3;
            y = RematerializeBlockIdxY() * 3;
            break;
        case 1:
            x = RematerializeBlockIdxX() * 3;
            y = RematerializeBlockIdxY() * 3 + 1;
            break;
        case 2:
            x = RematerializeBlockIdxX() * 3;
            y = RematerializeBlockIdxY() * 3 + 2;
            break;
        case 3:
            x = RematerializeBlockIdxX() * 3 + 1;
            y = RematerializeBlockIdxY() * 3;
            break;
        case 4:
            x = RematerializeBlockIdxX() * 3 + 1;
            y = RematerializeBlockIdxY() * 3 + 1;
            break;
        case 5:
            x = RematerializeBlockIdxX() * 3 + 1;
            y = RematerializeBlockIdxY() * 3 + 2;
            break;
        case 6:
            x = RematerializeBlockIdxX() * 3 + 2;
            y = RematerializeBlockIdxY() * 3;
            break;
        case 7:
            x = RematerializeBlockIdxX() * 3 + 2;
            y = RematerializeBlockIdxY() * 3 + 1;
            break;
        case 8:
            x = RematerializeBlockIdxX() * 3 + 2;
            y = RematerializeBlockIdxY() * 3 + 2;
            break;
        default:
            break;
        }
        if (x < tile_grid_h && y < tile_grid_w) {
            return linearIdx(x, y, tile_grid_w);
        } else {
            return NTile;
        }
    }

    CUTLASS_HOST_DEVICE
    dim3 get_grid_shape(int NTile, int tile_grid_h, int tile_grid_w)
    {
        return dim3(1, 1, 1);
        return dim3(((tile_grid_h + 2) / 3), (tile_grid_w + 2) / 3, 1);
    }
};




} // namespace threadblock

} // namespace sphconv
