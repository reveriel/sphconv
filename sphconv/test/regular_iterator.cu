#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"


// #include "sphconv/sphconv.h"
// #include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/layout.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"

#include "torch/extension.h"

#include "catch2/catch_test_macros.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

using sphconv::layout::TensorNTHWCShape;
using sphconv::threadblock::TensorNTHWCThreadMap;

template <typename Iterator>
__global__ void kernel_store(
    typename Iterator::TensorRef ref_output,
    typename Iterator::Element *input) {

    typename Iterator::Fragment frag;
    frag.clear();

    using AccessType = cutlass::Array<typename Iterator::Element, Iterator::ThreadMap::kElementsPerAccess>;

    int const kElementsPerAccess = Iterator::ThreadMap::kElementsPerAccess;
    int stride = Iterator::Shape::kContiguous;

    int warp_id = (threadIdx.x / 32);
    int lane_id = (threadIdx.x % 32);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Iterator::ThreadMap::Iterations::kStrided; ++s)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < Iterator::ThreadMap::Iterations::kContiguous; ++c)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < Iterator::ThreadMap::kElementsPerAccess; ++v)
            {
                frag[v + Iterator::ThreadMap::kElementsPerAccess *
                    (c + s * Iterator::ThreadMap::Iterations::kContiguous)] =
                    input[v + c * 64 + s * Iterator::ThreadMap::Delta::kStrided * stride];
            }
        }
    }
    Iterator iter(ref_output, threadIdx.x);
    iter.store(frag);
}

template <
    typename Shape_,
    int WarpCount
>
class MultiplicandTileIteratorTestbed {
public:
    using Shape = Shape_;
    using Element = float;
    using Layout = sphconv::layout::TensorNTHWC<Element, 4, 4, 16>;
    static int const kAdvanceRank = 1;
    static int const kThreads = 32 * WarpCount;
    using ThreadMap = TensorNTHWCThreadMap<
        Shape,
        kThreads,
        cutlass::layout::PitchLinearShape<16, 4>,
        1>;

    using Iterator = sphconv::threadblock::RegularTileIterator<
        Shape, Element, Layout, kAdvanceRank, ThreadMap>;
        // TODO
};


TEST_CASE("TileAccesssIteratorTest") {
    using Shape = TensorNTHWCShape<8, 8, 16>;
    using Layout = sphconv::layout::TensorNTHWC;
    using Element = int;
    int const kThreads = 32;
    using ThreadMap = TensorNTHWCThreadMap<Shape, 32>;

    // PredicatedTileAccessIterator<Shape, Element, 0, ThreadMap, >

}

