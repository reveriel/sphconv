
#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

#include "sphconv/sphconv.h"
#include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"
#include "sphconv/indice_conv/layout.cu.h"

#include "torch/torch.h"

#include <gtest/gtest.h>
#include <cuda.h>


TEST(testMath, myTest) {
    EXPECT_EQ(100, 100);
}

