
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace sphconv
{

struct Tensor5DCoord : public cutlass::Coord<5> {
    using Base = Coord<5>;
    using Index = typename Base::Index;
    using LongIndex = typename Base::LongIndex;
    static int const kN = 0;
    static int const KT = 1;
    static int const kH = 2;
    static int const kW = 3;
    static int const kC = 4;
}

} // namespace sphconv
