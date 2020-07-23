#pragma once

#include <vector>
#include <torch/extension.h>

std::vector<torch::Tensor> conv_cuda_forward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor weight,
    // torch::Tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups,
    int64_t D,
    int64_t subm);

std::vector<torch::Tensor> conv_cuda_backward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor gradOutput,
    torch::Tensor weights,
    // torch::Tensor bias,
    torch::Tensor InRuleMap,
    torch::Tensor OutRuleMap,
    torch::Tensor NumIn,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW,
    int64_t groups,
    int64_t sumb);
