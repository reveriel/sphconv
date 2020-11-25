#pragma once
#include <torch/extension.h>


void init() {
    return;

}

void forward(torch::Tensor output,
             torch::Tensor points,
             torch::Tensor weights)
{
    return;
}

void backward(torch::Tensor grad)
{
    return;
}