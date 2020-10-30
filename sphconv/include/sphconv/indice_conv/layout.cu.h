#pragma once

#include "sphconv/sphconv.h"
#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace sphconv {

namespace layout {

class TensorNTHWC {
    static int const kRank = 5;
    static int const kStrideRank = 4;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorCoord = cutlass::Coord<5>;
    using Stride = cutlass::Coord<kStrideRank>;
};

template <int N,
          int H,
          int W,
          int C>
class TensorNTHWCShape
{
public:
    static int const kN = N;
    static int const kH = H;
    static int const kW = W;
    static int const kC = C;
    // dynamic
    int t;
    TensorNTHWCShape(int t_) : t(t_)  {}
};



} // namespace layout
} // namespace sphconv