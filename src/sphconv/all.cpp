#include <iostream>
#include <vector>
#include <string>
#include <torch/extension.h>

#include "sphconv/tensor.h"
#include "sphconv/rules.h"
#include "sphconv/rule_conv.h"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(sphconv_cuda, m)
{

  // tensor.h
  m.def("init_tensor", &sphconv::init_tensor, "");
  m.def("init_tensor_backward", &sphconv::init_tensor_backward, "");

  m.def("to_dense", &sphconv::to_dense, "");
  m.def("to_dense_backward", &sphconv::to_dense_backward, "");

  // rules.h
  m.def("get_rules_subm", &sphconv::get_rules_subm, "");
  m.def("get_rules", &sphconv::get_rules, "");

  // rule_conv.h
  m.def("rule_conv", &sphconv::device::rule_conv, "");
  m.def("rule_conv_backward", &sphconv::device::rule_conv_backward, "");
  m.def("rule_conv_d_feature", &sphconv::device::rule_conv_d_feature, "");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
