#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

#include "sphconv/indice_conv/coord.cu.h"

namespace sphconv {
namespace layout {

class TensorNTHWC {

  static int const kRank = 5;
  static int const kStrideRank = 3;

  using Index = int32_t;
  using LongIndex = int64_t;
  using TensorCoord = TensorNTHWCCoord;

  using Stride = cutlass::Coord<kStrideRank>;


};


/// only H W C are const values
template <
  int H,
  int W,
  int C>
class TensorNTHWCShape {
public:
  static int const kH = H;
  static int const kW = W;
  static int const kC = C;
  // dynamic
  int n;
  int t;
  TensorNTHWCShape(): t(0), n(0) {}
  TensorNTHWCShape(int n_, int t_) : n(n_), t(t_) {}
};



} // namespace layout
} // namespace sphconv