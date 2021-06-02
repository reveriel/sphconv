#pragma once

#include "default_mma.cu.h"
#include "default_epilogue.cu.h"
#include "rule_conv_kernel.cu.h"

namespace sphconv
{
namespace kernel
{

template <
    int VBLOCK,
    // output channel size
    int Co_BLOCK,
    // input channel size
    int Ci_BLOCK = 8
    >
struct DefaultConv {

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = ElementC;
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    // v,  oC, iC
    using ThreadblockShape = cutlass::gemm::GemmShape<VBLOCK, Co_BLOCK, Ci_BLOCK>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static int const kStages = 2;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,  1, ElementAccumulator>;

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
        Operator
        >::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

    using Epilogue = typename sphconv::threadblock::DefaultEpilogueSimt<
        ThreadblockShape,
        typename Mma::Operator,
        EpilogueOutputOp,
        kEpilogueElementsPerAccess
        >::Epilogue;

    using ConvKernel = kernel::Conv<Mma, Epilogue>;

};

} // namespace kernel

} // namespace sphconv