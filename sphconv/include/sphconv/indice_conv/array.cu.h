#pragma once

#include "sphconv/sphconv.h"
#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace sphconv {

template<
    typename Element,
    int N,
    int H,
    int W,
    int C
>
class ArrayNTHWC {
    static int const kRank = 5;
    static int const kStrideRank = 4;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorCoord = cutlass::Coord<5>;
    using Stride = cutlass::Coord<kStrideRank>;
};


}