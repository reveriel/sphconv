#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "program.h"

// #define TORCH_EXTENSION_NAME conv_taichi

PYBIND11_MODULE(conv_taichi, m)
{
    m.def("init", &init, "");
    m.def("forward", &forward,"" );
    m.def("backward", &backward,"");
}