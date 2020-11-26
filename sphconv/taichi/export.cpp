#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "backbone.h"

// #define TORCH_EXTENSION_NAME conv_taichi

PYBIND11_MODULE(conv_taichi, m)
{
    m.def("init", &init, "");
    m.def("forward", &forward,"" );
    m.def("backward", &backward,"");
    m.def("profiler_print", &profiler_print, "");
}