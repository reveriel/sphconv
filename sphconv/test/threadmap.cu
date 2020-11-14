
#include <iostream>


#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

// #include "sphconv/sphconv.h"
// #include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/layout.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"


#define CATCH_CONFIG_MAIN
#include "catch2/catch_test_macros.hpp"

using std::cout;
using std::endl;

using sphconv::layout::TensorNTHWCShape;
using sphconv::threadblock::TensorNTHWCThreadMap;

TEST_CASE("TensorShapeConstructorTest", ) {
    using Shape = TensorNTHWCShape<8, 8, 16>;

    Shape shape;
    REQUIRE((Shape::kH == 8));
    REQUIRE((Shape::kW == 8));
    REQUIRE((Shape::kC == 16));
    REQUIRE(shape.n == 0);
    REQUIRE(shape.t == 0);
}


TEST_CASE("ThreadMapInspect") {

    int constexpr W = 64;
    int constexpr H = 32;
    using Shape = cutlass::layout::PitchLinearShape<W, H>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int constexpr kThreads = 32;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    REQUIRE((ThreadMap::Detail::ShapeVec::kContiguous == W));
    REQUIRE((ThreadMap::Detail::ShapeVec::kStrided == H));

    cout << "ShapeVec::kContiguous =  " << ThreadMap::Detail::ShapeVec::kContiguous << endl;
    cout << "ShapeVec::kStrided  = " << ThreadMap::Detail::ShapeVec::kStrided << endl;
    // iterations,  delta, initial_offset

    REQUIRE((ThreadMap::Iterations::kContiguous == (W / kThreads)));
    REQUIRE((ThreadMap::Iterations::kStrided == H));
    cout << "Iterations::kContiguous =  " << ThreadMap::Iterations::kContiguous << endl;
    cout << "Iterations::kStrided =  " << ThreadMap::Iterations::kStrided << endl;

    // Delta : Interval between accesses along each dimension
    REQUIRE((ThreadMap::Delta::kContiguous == (kThreads)));
    REQUIRE((ThreadMap::Delta::kStrided == 1));
    cout << "Delta::kContiguous =  " << ThreadMap::Delta::kContiguous << endl;
    cout << "Delta::kStrided =  " << ThreadMap::Delta::kStrided << endl;

    // 3
    auto coord = ThreadMap::initial_offset(3);
    cout << "initial_offset" << coord.at(0) << "," << coord.at(1) << endl;

    coord = ThreadMap::initial_offset(65);
    cout << "initial_offset" << coord.at(0) << "," << coord.at(1) << endl;
}

TEST_CASE("ThreadMapTest") {
    const int H = 8, W = 8, C = 16;
    using Shape = TensorNTHWCShape<H, W, C>;
    using Layout = sphconv::layout::TensorNTHWC;
    using Element = int;
    int const kThreads = 32;

    using ThreadMap = TensorNTHWCThreadMap<Shape, kThreads>;

    REQUIRE((ThreadMap::Detail::ShapeVec::kContiguous == W * C));
    REQUIRE((ThreadMap::Detail::ShapeVec::kStrided == H));

    REQUIRE((ThreadMap::Iterations::kContiguous == (W * C)/ kThreads ));
    REQUIRE((ThreadMap::Iterations::kStrided == H));

    REQUIRE((ThreadMap::Delta::kContiguous == kThreads));
    REQUIRE((ThreadMap::Delta::kStrided == 1));

}


// TEST(TensorShapeTest, fail) {
//     EXPECT_TRUE(false);
// }