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

std::vector<torch::Tensor>
indice_conv(torch::Tensor feature,
            torch::Tensor weight,
            torch::Tensor InRuleMap,
            torch::Tensor OutRuleMap,
            torch::Tensor NumIn,
                    //  torch::Tensor bias,
                    int64_t D, int64_t oT,
                     int64_t sD, int64_t sH, int64_t sW,
                     int64_t padD, int64_t padH, int64_t padW,
                     int64_t dD, int64_t dH, int64_t dW,
                     int64_t groups);

std::vector<torch::Tensor>
get_indice_pairs(torch::Tensor depth,
                 torch::Tensor thick,
                 int64_t N, int64_t T,
                 int64_t D, int64_t H, int64_t W,
                 int64_t KD, int64_t KH, int64_t KW,
                 int64_t sD, int64_t sH, int64_t sW,
                 int64_t padD, int64_t padH, int64_t padW,
                 int64_t dD, int64_t dH, int64_t dW,
                 int64_t groups);

std::vector<torch::Tensor>
get_indice_pairs_subm(torch::Tensor depth,
                      torch::Tensor thick,
                      int64_t N, int64_t T,
                      int64_t D, int64_t H, int64_t W,
                      int64_t KD, int64_t KH, int64_t KW,
                      int64_t sD, int64_t sH, int64_t sW,
                      int64_t padD, int64_t padH, int64_t padW,
                      int64_t dD, int64_t dH, int64_t dW,
                      int64_t groups);