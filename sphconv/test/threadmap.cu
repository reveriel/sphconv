
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

#include "sphconv/sphconv.h"
// #include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/layout.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"

#include "torch/extension.h"

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

using sphconv::layout::TensorNTHWCShape;
using sphconv::threadblock::TensorNTHWCThreadMap;

TEST(TensorShape, test1) {
    using Shape = TensorNTHWCShape<8, 8, 16>;

    Shape shape;
    EXPECT_EQ(Shape::kH, 8);
    EXPECT_EQ(Shape::kW, 8);
    EXPECT_EQ(Shape::kC, 16);
    EXPECT_EQ(shape.n, 0);
    EXPECT_EQ(shape.t, 0);
}

/*
TEST(ThreadMap, ref1) {
    using Shape = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    cout << "ShapeVec::kContiguous =  " << ThreadMap::Detail::ShapeVec::kContiguous << endl;
    cout << "ShapeVec::kStrided  = " << ThreadMap::Detail::ShapeVec::kStrided << endl;
    // iterations,  delta, initial_offset

    cout << "Iterations::kContiguous =  " << ThreadMap::Iterations::kContiguous << endl;
    cout << "Iterations::kStrided =  " << ThreadMap::Iterations::kStrided << endl;

    // Delta : Interval between accesses along each dimension
    cout << "Delta::kContiguous =  " << ThreadMap::Delta::kContiguous << endl;
    cout << "Delta::kStrided =  " << ThreadMap::Delta::kStrided << endl;

}

TEST(ThreadMap, test1) {
    using Shape = TensorNTHWCShape<8, 8, 16>;

    using Shape_ref =

    using ThreadMap = TensorNTHWCThreadMap<Shape, 32>;
    // cout << ThreadMap::Delta
}


*/
