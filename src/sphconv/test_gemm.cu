#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"

#include "cutlass/util/host_tensor.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


inline char const* to_string(cutlass::Status status)
{

    switch (status) {
    case cutlass::Status::kSuccess:
        return "kSuccess";
    case cutlass::Status::kErrorMisalignedOperand:
        return "kErrorMisalignedOperand";
    case cutlass::Status::kErrorInvalidLayout:
        return "kErrorInvalidLayout";
    case cutlass::Status::kErrorInvalidProblem:
        return "kErrorInvalidProblem";
    case cutlass::Status::kErrorNotSupported:
        return "kErrorNotSupported";
    case cutlass::Status::kErrorWorkspaceNull:
        return "kErrorWorkspaceNull";
    case cutlass::Status::kErrorInternal:
        return "kErrorInternal";
    case cutlass::Status::kInvalid:
        return "kInvalid";
    default:
        break;
    }
    return "invalid";
}

template <typename Gemm>
class Test
{
    using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

public:
    typename Gemm::TensorRefA ref_A;
    typename Gemm::TensorRefB ref_B;
    typename Gemm::TensorRefC ref_C;
    typename Gemm::TensorRefD ref_D;

    Test(torch::Tensor A,
         torch::Tensor B,
         torch::Tensor C)
    {
        auto a = A.packed_accessor64<float, 2, torch::RestrictPtrTraits>();
        auto b = B.packed_accessor64<float, 2, torch::RestrictPtrTraits>();
        auto c = C.packed_accessor64<float, 2, torch::RestrictPtrTraits>();

        ref_A = typename Gemm::TensorRefA((float*)A.data_ptr(), cutlass::layout::RowMajor(a.stride(0)));
        ref_B = typename Gemm::TensorRefB((float*)B.data_ptr(), cutlass::layout::RowMajor(b.stride(0)));
        ref_C = typename Gemm::TensorRefC((float*)C.data_ptr(), cutlass::layout::RowMajor(c.stride(0)));
        ref_D = typename Gemm::TensorRefD((float*)C.data_ptr(), cutlass::layout::RowMajor(c.stride(0)));
    }

    bool run(
        cutlass::gemm::GemmCoord problem_size,
        int split_k_slices = 1,
        ElementCompute alpha = ElementCompute(1),
        ElementCompute beta = ElementCompute(0))
    {

        // D = A * B + C
        typename Gemm::Arguments arguments{
            problem_size,
            ref_A,
            ref_B,
            ref_C,
            ref_D,
            {alpha, beta},
            split_k_slices};

        Gemm gemm_op;

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

        if (status != cutlass::Status::kSuccess) {
            cudaError_t error = cudaGetLastError();
            std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
            return true;
        }

        //
        // Run the GEMM
        //

        status = gemm_op();

        std::cout << to_string(status) << std::endl;
        // EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        return status == cutlass::Status::kSuccess;
    }
};

template <int M, int N, int K>
void test_cutlass(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    using precision = float;
    using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::RowMajor,
        precision, cutlass::layout::RowMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
        >;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    Test<Gemm> test(A, B, C);
    double alpha = 1;
    double beta = 1;
    int split_k = 1;

    using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

    //  D = alpha * A * B + beta * C
    test.run(problem_size, split_k,
             cutlass::from_real<ElementCompute>(alpha),
             cutlass::from_real<ElementCompute>(beta));
}

int main()
{
    const int M = 8;
    const int N = 32;
    const int K = 64;

    torch::manual_seed(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0).requires_grad(false);
    // torch::Tensor A = torch::randint(40, {M, K}, options);
    // torch::Tensor B = torch::randint(40, {K, N}, options);
    torch::Tensor A = torch::randn({M, K}, options);
    torch::Tensor B = torch::randn({K, N}, options);
    torch::Tensor C_torch = torch::zeros({M, N}, options);
    torch::mm_out(C_torch, A, B);

    torch::Tensor C_cutlass = torch::zeros({M, N}, options);
    test_cutlass<M, N, K>(A, B, C_cutlass);


    std::cout << " C_torch = \n" << C_torch << std::endl;
    std::cout << " cutlass result C2 = \n" << C_cutlass << std::endl;

    std::cout << "distance = " << (C_cutlass - C_torch).abs().sum() << std::endl;

    return 0;
}