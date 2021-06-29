#include <iostream>
#include <vector>
#include <string>
#include <torch/extension.h>

#include "rules.h"
#include "rule_conv.h"
#include "tensor.h"

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
  m.def("get_rules_subm", &sphconv::get_rules_subm, "");
  m.def("get_rules", &sphconv::get_rules, "");

  m.def("rule_conv", &sphconv::device::rule_conv, "");
  m.def("rule_conv_backward", &sphconv::device::rule_conv_backward, "");
  m.def("rule_conv_d_feature", &sphconv::device::rule_conv_d_feature, "");

  m.def("init_tensor", &sphconv::init_tensor, "");
  m.def("init_tensor_backward", &sphconv::init_tensor_backward, "");

  m.def("to_dense", &sphconv::to_dense, "");
  m.def("to_dense_backward", &sphconv::to_dense_backward, "");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
