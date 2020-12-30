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
void init(bool debug=false);

/**
 * forward
 */
void forward(torch::Tensor output,
             torch::Tensor points,
             std::vector<torch::Tensor> weights);

void backward(torch::Tensor grad);

void print_last_layer();
void print_first_layer();
void print_first_layer_nz();
void print_last_layer_nz();
void print_layer(int i);

void profiler_print();
