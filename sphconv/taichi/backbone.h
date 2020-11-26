#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct Backbone;
namespace taichi{
namespace Tlang{
struct Program;
}
}

///
// global taichi program
extern std::unique_ptr<taichi::Tlang::Program> prog;
// the backbone
extern std::unique_ptr<Backbone> backbone;


/**
 *  init the backbone taichi program
 */
void init();

/**gg
 * forward
 */
void forward(torch::Tensor output,
             torch::Tensor points,
             py::tuple weights);

void backward(torch::Tensor grad);
