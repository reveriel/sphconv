
#include <torch/extension.h>

#include <iostream>

void test()
{
    std::cout << "test " << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    //   m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
    //   m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
    m.def("test", &test, "test func");
}
