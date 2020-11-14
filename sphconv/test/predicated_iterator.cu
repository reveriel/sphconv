#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"


// #include "sphconv/sphconv.h"
#include "sphconv/indice_conv/iterator.cu.h"
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
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<5> extent) {

    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    int iterations = (extent[2] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment frag;

    for (int i = 0; i < frag.size(); i++)
        frag[i] = 0;

    src_iterator.load(frag);
    dst_iterator.store(frag);

    ++dst_iterator;
    ++src_iterator;
    for (; iterations > 1; --iterations) {
        src_iterator.load(frag);
        dst_iterator.store(frag);

        ++dst_iterator;
        ++src_iterator;
    }

}



TEST_CASE("TileAccesssIteratorTest")
{
    int const N = 3, T = 3, H = 8, W = 8, C = 16;

    using Shape = TensorNTHWCShape<H, W, C>;
    using Layout = sphconv::layout::TensorNTHWC;
    using Element = int;
    int const kThreads = 32;
    using ThreadMap = TensorNTHWCThreadMap<Shape, kThreads>;

    using Iterator = sphconv::threadblock::PredicatedTileIterator<
        Shape, Element, Layout, 1, ThreadMap>;

    cutlass::Coord<2> copy_extent = cutlass::make_Coord(H, W);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(H, W);

    // configration

    auto options = torch::TensorOptions().dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, 0)
        .requires_grad(false);

    // allocate source and destination tensors using pytorch
    torch::Tensor src_tensor = torch::rand({N, T, H, W, C}, options);
    torch::Tensor dst_tensor = torch::zeros({N, T, H, W, C}, options);
    Layout src_layout = Layout()


    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_parasm(src_tesnor.layout());


}
