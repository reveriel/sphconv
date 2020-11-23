#include "utils.h"
#include "npy.hpp"

void init_taichi_program() {
    taichi::CoreState::set_trigger_gdb_when_crash(false);
}

// create points data,  on GPU, of shape (N, 4)
Points gen_points_data(int num_points, int num_channel ) {

    std::string points_file_name = "points0.npy";
    // numpy array

    Points points;
    npy::LoadArrayFromNumpy(points_file_name, points.shape, points.fortran_order, points.data);
    std::cout << "shape: ";
    for (size_t i = 0; i < points.shape.size(); i++)
        std::cout << points.shape[i] << ", ";
    std::cout << std::endl;
    // shape: shape: 20285, 4,

    return points;
}