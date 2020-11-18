#include <torch/extension.h>
#include <taichi/lang.h>

using taichi::Tlang::Expr;

// cuda
__global__
void to_torch(Expr arr, torch::Tensor) {

}


__global__
void from_torch(Expr arr, torch::Tensor) {

}

