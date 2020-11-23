#pragma once

#include <vector>

// data read from numpy file
struct Points {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;
};
