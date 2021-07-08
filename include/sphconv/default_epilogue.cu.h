#pragma once

#include "sphconv/iterator.cu.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/wmma_array.h"

#include "cutlass/arch/arch.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "debug_utils.h"
#include "timer.h"

namespace sphconv
{

namespace threadblock
{

template <
    typename Shape_,
    typename WarpMmaSimt_,
    typename OutputOp_,
    int ElementsPerAccess,
    int reverseRule>
struct DefaultEpilogueSimt {
    using Shape = Shape_;
    using WarpMmaSimt = WarpMmaSimt_;
    using OutputOp = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

    using ElementOutput = typename OutputOp::ElementOutput;
    using LayoutC = typename WarpMmaSimt::LayoutC;
    using ElementAccumulator = typename WarpMmaSimt::ElementC;

    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
        Shape,
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::Policy,
        kPartitionsK,
        ElementOutput,
        kElementsPerAccess>::Type;

    // using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //     OutputTileThreadMap,
    //     ElementOutput
    // >;

    using OutputTileIterator = sphconv::threadblock::OutTileIterator<
        ElementOutput,
        OutputTileThreadMap, 1 - reverseRule>;

    using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy>;

    using WarpTileIterator = cutlass::epilogue::warp::TileIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        ElementAccumulator,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy>;

    using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
        typename OutputTileThreadMap::CompactedThreadMap,
        ElementAccumulator>;

    /// Hard-coded padding elements added
    using Padding = typename WarpTileIterator::Padding;

    //
    // Define the epilogue
    //
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        Shape,
        WarpMmaSimt,
        kPartitionsK,
        OutputTileIterator,
        AccumulatorFragmentIterator,
        WarpTileIterator,
        SharedLoadIterator,
        OutputOp,
        Padding>;
};







template <
    typename Shape_,
    typename WarpMmaSimt_,
    typename OutputOp_,
    int ElementsPerAccess>
struct DefaultEpilogueSimt_DW {
    using Shape = Shape_;
    using WarpMmaSimt = WarpMmaSimt_;
    using OutputOp = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

    using ElementOutput = typename OutputOp::ElementOutput;
    using LayoutC = typename WarpMmaSimt::LayoutC;
    using ElementAccumulator = typename WarpMmaSimt::ElementC;

    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
        Shape,
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::Policy,
        kPartitionsK,
        ElementOutput,
        kElementsPerAccess>::Type;

    // using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //     OutputTileThreadMap,
    //     ElementOutput
    // >;

    using OutputTileIterator = sphconv::threadblock::NKernelTileIterator<
        ElementOutput,
        OutputTileThreadMap, 1>;

    using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy>;

    using WarpTileIterator = cutlass::epilogue::warp::TileIteratorSimt<
        typename WarpMmaSimt::Shape,
        typename WarpMmaSimt::ThreadMma,
        ElementAccumulator,
        cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy>;

    using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
        typename OutputTileThreadMap::CompactedThreadMap,
        ElementAccumulator>;

    /// Hard-coded padding elements added
    using Padding = typename WarpTileIterator::Padding;

    //
    // Define the epilogue
    //
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        Shape,
        WarpMmaSimt,
        kPartitionsK,
        OutputTileIterator,
        AccumulatorFragmentIterator,
        WarpTileIterator,
        SharedLoadIterator,
        OutputOp,
        Padding>;
};








































} // namespace threadblock
} // namespace sphconv
