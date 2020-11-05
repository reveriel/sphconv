
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

#include "sphconv/sphconv.h"
// #include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/layout.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"

#include "torch/extension.h"

// #include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

using sphconv::layout::TensorNTHWCShape;
using sphconv::threadblock::TensorNTHWCThreadMap;

/*
TEST(TensorShapeTest, Constructor) {
    using Shape = TensorNTHWCShape<8, 8, 16>;

    Shape shape;
    EXPECT_TRUE(Shape::kH == 8);
    EXPECT_TRUE(Shape::kW == 8);
    EXPECT_TRUE(Shape::kC == 16);
    EXPECT_EQ(shape.n, 0);
    EXPECT_EQ(shape.t, 0);
}
*/

int main () {
    using Shape = TensorNTHWCShape<8, 8, 16>;

    Shape shape;
    // EXPECT_TRUE(Shape::kH == 8);
    // EXPECT_TRUE(Shape::kW == 8);
    // EXPECT_TRUE(Shape::kC == 16);
    // EXPECT_EQ(shape.n, 0);
    // EXPECT_EQ(shape.t, 0);
    cout << Shape::kH << endl;
    cout << shape.n << endl;
    return 0;
}


/*
// compiler complains "error: expression must have a constant value"
// I don't konw why
TEST(ThreadMapTest, Inspect) {

    int const W = 64;
    int const H = 32;
    using Shape = cutlass::layout::PitchLinearShape<W, H>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    EXPECT_TRUE(ThreadMap::Detail::ShapeVec::kContiguous == W);
    EXPECT_TRUE(ThreadMap::Detail::ShapeVec::kStrided == H);

    // cout << "ShapeVec::kContiguous =  " << ThreadMap::Detail::ShapeVec::kContiguous << endl;
    // cout << "ShapeVec::kStrided  = " << ThreadMap::Detail::ShapeVec::kStrided << endl;
    // iterations,  delta, initial_offset

    EXPECT_TRUE(ThreadMap::Iterations::kContiguous == (W / kThreads));
    EXPECT_TRUE(ThreadMap::Iterations::kStrided == H);
    // cout << "Iterations::kContiguous =  " << ThreadMap::Iterations::kContiguous << endl;
    // cout << "Iterations::kStrided =  " << ThreadMap::Iterations::kStrided << endl;

    // Delta : Interval between accesses along each dimension
    EXPECT_TRUE(ThreadMap::Delta::kContiguous == (kThreads));
    EXPECT_TRUE(ThreadMap::Delta::kStrided == 1);
    // cout << "Delta::kContiguous =  " << ThreadMap::Delta::kContiguous << endl;
    // cout << "Delta::kStrided =  " << ThreadMap::Delta::kStrided << endl;

    EXPECT_TRUE(true);
}
*/

/*

TEST(TensorShapeTest, test1) {
    using Shape = TensorNTHWCShape<8, 8, 16>;
    using Layout = sphconv::layout::TensorNTHWC;
    using Element = int;
    int const kThreads = 32;

    using ThreadMap = TensorNTHWCThreadMap<Shape, 32>;

    EXPECT_TRUE(ThreadMap::Detail::ShapeVec::kContiguous == 128);
    EXPECT_TRUE(ThreadMap::Detail::ShapeVec::kStrided == 8);

    EXPECT_TRUE(ThreadMap::Iterations::kContiguous == 4);
    EXPECT_TRUE(ThreadMap::Iterations::kStrided == 8);

    EXPECT_TRUE(ThreadMap::Delta::kContiguous == 32);
    EXPECT_TRUE(ThreadMap::Delta::kStrided == 1);

    // cout << ThreadMap::Delta
}


TEST(TensorShapeTest, fail) {
    EXPECT_TRUE(false);
}
*/