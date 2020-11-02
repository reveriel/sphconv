#pragma once

#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/layout/pitch_linear.h"

#include "sphconv/sphconv.h"
#include "sphconv/indice_conv/layout.cu.h"
#include "sphconv/indice_conv/coord.cu.h"

// how threads are mapped to a given tile
namespace sphconv {
namespace threadblock {


template<
  typename Shape_,
  int Threads,
  int ElementsPerAccess = 1
>
struct TensorNTHWCThreadMap {
  // Tensor coordinate
  using TensorCoord = typename TensorNTHWCCoord;
  // tile shape on H and W, and C
  using Shape = Shape_;
  /// Number of threads total
  static int const kThreads = Threads;
  static int const kElementsPerAccess = ElementsPerAccess;
  /// shape of access per warp
  //  B,T, H,W,C,
  using ThreadAccessShape = cutlass::layout::PitchLinearShape<kElementsPerAccess, 1>; //
  struct Detail {
    static_assert(!(Shape::kW * Shape::kC % kElementsPerAccess), "");

    static_assert(!((Shape::kW * Shape::kC * Shape::kH) %
      (kThreads * kElementsPerAccess)), "Shape must be divisible thread count.");

      /// Shape
    /// Number of threads per warp
    // static int const kWarpSize = 32; //WarpsThread ?;
    /// Number of participating warps
    // static int const kWarpCount =  ?;

    /// Shape of the tile in units of vectors
    using ShapeVec = typename cutlass::layout::PitchLinearShape<
      Shape::kW * Shape::kC / kElementsPerAccess,
      Shape::kH >;

  };
  // Number of iterations by each thread
  // Iterations along each dimmension ?
  using Iterations = typename std::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    cutlass::layout::PitchLinearShape<
      1,
      (Threads >= Detail::ShapeVec::KContiguous ?
        Detail::ShapeVec::kStrided / (kThreads / Detail::ShapeVec::kContiguous)
        : 0)>,
    cutlass::layout::PitchLinearShape<
      Detail::ShapeVec::kContiguous / kThreads,
      Detail::ShapeVec::kStrided>
    >::type;

  // delta between accesses along each dimension of the tensor's logical
  // coordinate
  using Delta = typename std::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    cutlass::layout::PitchLinearShape<
      1,
      kThreads / Detail::ShapeVec::kContiguous
    >,
    cutlass::layout::PitchLinearShape<
      kThreads * kElementsPerAccess,
      1
    >
  >::type;
  /// Maps thread ID to a coordinate offset within the tensor's logical
  /// coordinate space (in units of Elements)
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

      return TensorCoord(
        (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess,
        thread_id / Detail::ShapeVec::kContiguous);
  }
};


} // namespace threadblock
} // namespace sphconv
