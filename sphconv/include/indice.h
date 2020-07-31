#pragma once

#include <vector>
#include <torch/extension.h>

namespace sphconv {

std::vector<torch::Tensor>
get_indice_pairs(torch::Tensor depth,
                 torch::Tensor thick,
                 int D,
                 int KD, int KH, int KW,
                 int sD, int sH, int sW,
                 int padD, int padH, int padW,
                 int dD, int dH, int dW,
                 int groups);

std::vector<torch::Tensor>
get_indice_pairs_subm(torch::Tensor depth,
                        torch::Tensor thick,
                        int D,
                        int KD, int KH, int KW,
                        int sD, int sH, int sW,
                        int padD, int padH, int padW,
                        int dD, int dH, int dW,
                        int groups);

} // namespace sphconv
