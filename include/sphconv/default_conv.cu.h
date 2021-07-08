#pragma once

#include "sphconv/default_mma.cu.h"
#include "sphconv/default_epilogue.cu.h"
#include "sphconv/rule_conv_kernel.cu.h"

namespace sphconv
{
namespace kernel
{

template <
    /// GemmShape, V, oC, iC
    typename ThreadBlockShape_,
    /// GemmShape, V, oC, iC
    typename WarpShape_,
    int VBLOCK,
    typename ThreadblockSwizzle,
    int reverseRule>
struct DefaultConv {

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = ElementC;
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    // v,  oC, iC
    using ThreadblockShape = ThreadBlockShape_; // cutlass::gemm::GemmShape<VBLOCK, Co_BLOCK, Ci_BLOCK>;
    using WarpShape = WarpShape_;               // cutlass::gemm::GemmShape<16, Co_, 8>;

    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static int const kStages = 2;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator>;

    using Mma = typename sphconv::threadblock::DefaultMma<
        ElementA,
        cutlass::layout::RowMajor,
        kAlignmentA,
        ElementB,
        cutlass::layout::RowMajor,
        kAlignmentB,
        ElementC,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50, ThreadblockShape, WarpShape, InstructionShape, kStages,
        Operator,
        false,
        reverseRule>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

    using Epilogue = typename sphconv::threadblock::DefaultEpilogueSimt<
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess,
        reverseRule>::Epilogue;

    using ConvKernel = kernel::Conv<Mma, Epilogue, VBLOCK, ThreadblockSwizzle>;
};

template <
    /// GemmShape, V, oC, iC
    typename ThreadblockShape_,
    /// GemmShape, V, oC, iC
    typename WarpShape_,
    int VBLOCK,
    typename ThreadblockSwizzle>
struct DefaultConvDW {

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = ElementC;
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    // v,  oC, iC
    using ThreadblockShape = ThreadblockShape_; // cutlass::gemm::GemmShape<VBLOCK, Co_BLOCK, Ci_BLOCK>;
    using WarpShape = WarpShape_;               // cutlass::gemm::GemmShape<16, Co_, 8>;

    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static int const kStages = 2;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator>;

    using Mma = typename sphconv::threadblock::DefaultMmaDW<
        ElementA,
        cutlass::layout::RowMajor,
        kAlignmentA,
        ElementB,
        cutlass::layout::RowMajor,
        kAlignmentB,
        ElementC,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50, ThreadblockShape, WarpShape, InstructionShape, kStages,
        Operator>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

    using Epilogue = typename sphconv::threadblock::DefaultEpilogueSimt_DW<
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess>::Epilogue;

    using ConvKernel = kernel::ConvDW<Mma, Epilogue, VBLOCK, ThreadblockSwizzle>;
};

} // namespace kernel

} // namespace sphconv
