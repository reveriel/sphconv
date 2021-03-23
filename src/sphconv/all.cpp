
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <string>

#include "indice.h"
#include "rules.h"
#include "rule_conv.h"
#include "indice_conv.h"
#include "to_dense.h"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(sphconv_cuda, m)
{
  // m.def("get_indice_pairs", &sphconv::get_indice_pairs, "");
  // m.def("get_indice_pairs_subm", &sphconv::get_indice_pairs_subm, "");
  m.def("get_rules_subm", &sphconv::get_rules_subm, "");
  m.def("rule_conv", &sphconv::rule_conv, "");
  // m.def("indice_conv", &sphconv::indice_conv, "");
  // m.def("indice_conv_gemm", &sphconv::indice_conv_gemm, "");
  // m.def("conv_backward", &sphconv::indice_conv_backward, "");
  // m.def("conv_backward_gemm", &sphconv::indice_conv_backward_gemm, "");
  // m.def("to_dense", &sphconv::to_dense, "");
  // m.def("to_dense_backward", &sphconv::to_dense_backward, "");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
