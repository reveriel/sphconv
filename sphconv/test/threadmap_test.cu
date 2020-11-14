#include <iostream>
using std::cout;
using std::endl;

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

int main() {
    using Shape = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    static int const kThreads = 32;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    cout << "ShapeVec::kContiguous =  " << ThreadMap::Detail::ShapeVec::kContiguous << endl;
    cout << "ShapeVec::kStrided  = " << ThreadMap::Detail::ShapeVec::kStrided << endl;
    // iterations,  delta, initial_offset

    cout << "Iterations::kContiguous =  " << ThreadMap::Iterations::kContiguous << endl;
    cout << "Iterations::kStrided =  " << ThreadMap::Iterations::kStrided << endl;

    // Delta : Interval between accesses along each dimension
    cout << "Delta::kContiguous =  " << ThreadMap::Delta::kContiguous << endl;
    cout << "Delta::kStrided =  " << ThreadMap::Delta::kStrided << endl;

    return 0;

}