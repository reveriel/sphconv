#pragma once
#include "cutlass/cutlass.h"
#include "sphconv/sphconv.h"

// how threads are mapped to a given tile
namespace sphconv {
namespace threadblock {


template<
  typename Shape_,
  int Threads
>
struct TensorNTHWCThreadMap {
    // Tensor coordinate
    using TensorCoord = layout::TensorNTHWCCoord;
    // tile shape on H and W
    using Shape = Shape_;
    /// Number of threads total
    static int const kThreads = Threads;
    static int const kElementsPerAccess = ElementsPerAccess;
    /// shape of access per warp
    //  B,T, H,W,C,
    using ThreadAccessShape =layout::TensorNTHWCShape<>; //
    struct Detials {
        /// Number of threads per warp
        static int const kWarpSize = 32; //WarpzThread ?;
        i/// Number of participating warps
        static int const kWarpCount =  ?;

    };
    // Iterations along each dimmension ?
    using Iterations = layout ..;
    // delta between accesses
    using Delta =  ?;
    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space
    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        // int warp_id = (thread_id / Detail::kWarpSize);
        // int lane_id = (thread_id % Detail::kWarpSize);
        // // compute warp-level offset
        // warp_footprint {};
        // warp_offset
        // thread_offset_in_warp
        return  {thread_id  };
    }

};


} // namespace threadblock
} // namespace sphconv
