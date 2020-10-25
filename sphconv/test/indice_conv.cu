
#include "sphconv/sphconv.h"

#include "sphconv/indice_conv/device.cu.h"

int main() {
    // define the operation
    using IndiceConv = sphconv::device::IndiceConv<
        float, int, float, int>;

    IndiceConv conv_op;

    // construct arguments to the operation
    IndiceConv::Arguments args(

    );


    // lanuch the kernel
    cutlass::Status status = conv_op(args);



}