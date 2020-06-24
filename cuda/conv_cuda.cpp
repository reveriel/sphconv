
#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> sphconv_cuda_forward(
    torch::tensor feature,
    torch::tensor depth,
    torch::tensor thick,
    torch::tensor weight,
    torch::tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW);

std::vector<torch::Tensor> sphconv_cuda_backward(
    torch::Tensor grad_feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor grad_weight,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding,
    torch::Tensor dilation,
    torch::Tensor groups);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sphconv_forward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor weights,
    torch::Tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW)
{
    CHECK_INPUT(feature);
    CHECK_INPUT(depth);
    CHECK_INPUT(thick);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);

    return sphconv_cuda_forward(feature, depth, thick, weight, bias, stride,
                                padding, dilation, groups);
}

std::vector<torch::Tensor> sphconv_backward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor gradOutput,
    torch::Tensor weights,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW)
{

    CHECK_INPUT(gradOutput);

    return sphconv_cuda_backward(feature, depth, thick, weight, bias, stride,
                                padding, dilation, groups);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &sphconv_forward, "LLTM forward (CUDA)");
    m.def("backward", &sphconv_backward, "LLTM backward (CUDA)");
    m.def("test", &test, "test func");
}
