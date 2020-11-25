#pragma once
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <cstdlib>


void init() {
    return;

}

void forward(torch::Tensor output,
             torch::Tensor points,
             py::tuple weights)
{
    printf("number of weights = %d\n", (int)weights.size());

    return;
}

void backward(torch::Tensor grad)
{
    return;
}