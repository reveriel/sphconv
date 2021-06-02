#pragma once

#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

#include "iterator.cu.h"
#include "debug_utils.h"
#include "timer.h"

namespace sphconv
{

namespace threadblock
{

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false>
struct DefaultMma;

/// Specialization for simt operator with 2 staged pipeline
// row-major x row-major
template <
    ///< Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  cutlass::arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator> {
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
        ElementB, LayoutB, ElementAccumulator, LayoutC,
        cutlass::arch::OpClassSimt, 2, Operator>;

    static int const kThreads = MmaCore::kThreads;
    static int const kElementsPerAccess = 1;
    using Shape = typename MmaCore::Shape; // gemmshape

    using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
        kThreads,
        kElementsPerAccess>;

    // Define iterators over tiles from the A operand
    using IteratorA =
        sphconv::threadblock::InputTileIterator<
            ElementA, IteratorThreadMapA, kAlignmentA>;

    using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
        kThreads,
        kElementsPerAccess>;

    // we customized the threadMap, of defaultMmaCore
    using IteratorB =
        sphconv::threadblock::KernelTileIterator<
            ElementB, IteratorThreadMapB, kAlignmentB>;

    // Define the threadblock-scoped singlestage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
        IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
        LayoutC, typename MmaCore::MmaPolicy>;
};

} // namespace threadblock
} // namespace sphconv
