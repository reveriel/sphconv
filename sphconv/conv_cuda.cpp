
#include <torch/extension.h>

#include <iostream>
#include <vector>

#include "conv_cuda.h"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>  conv_forward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor weights,
    // torch::Tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups,
    int64_t D,
    int64_t subm)
{
    CHECK_INPUT(feature);
    CHECK_INPUT(depth);
    CHECK_INPUT(thick);
    CHECK_INPUT(weights);
    // CHECK_INPUT(bias);

    return conv_cuda_forward(feature, depth, thick, weights,
                                sD, sH, sW,
                                padD, padH, padW,
                                dD, dH, dW,
                                groups,
                                D, subm);
}

std::vector<torch::Tensor> conv_backward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor gradOutput,
    torch::Tensor weights,
    // torch::Tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups,
    int64_t subm)
{

    CHECK_INPUT(gradOutput);

    return conv_cuda_backward(feature,
                                 depth,
                                 thick,
                                 gradOutput,
                                 weights,
                                //  bias,
                                 sD, sH, sW,
                                 padD, padH, padW,
                                 dD, dH, dW,
                                 groups,
                                 subm);
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv_forward", &conv_forward, "conv forward (CUDA)");
    m.def("conv_backward", &conv_backward, "conv backward (CUDA)");
}

