#include <point2voxel.h>
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(sphconv_utils, m)
{
    m.doc() = "util pybind11 functions for sphconv";
    m.def("points_to_voxel_3d_np", &sphconv::points_to_voxel_3d_np<float, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
          "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
          "max_points"_a = 9, "max_voxels"_a = 10);
    m.def("points_to_voxel_3d_np", &sphconv::points_to_voxel_3d_np<double, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
          "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
          "max_points"_a = 9, "max_voxels"_a = 10);
    m.def("points_to_voxel_3d_sphere_np",
          &sphconv::points_to_voxel_3d_sphere_np<double, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
          "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
          "max_points"_a = 9, "max_voxels"_a = 10);
    m.def("points_to_voxel_3d_sphere_np",
          &sphconv::points_to_voxel_3d_sphere_np<float, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
          "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
          "max_points"_a = 9, "max_voxels"_a = 10);
    m.def("points_to_voxel_3d",
          &sphconv::points_to_voxel_3d<float, 3>,
          "matrix tensor_square", "points"_a = 1, "system_points"_a = 2,
          "voxels"_a = 3,
          "voxel_point_mask"_a = 4, "coors"_a = 5, "num_points_per_voxel"_a = 6,
          "coor_to_voxelidx"_a = 7, "coors_range"_a = 8,
          "grid_size"_a = 9,
          "max_points"_a = 10, "max_voxels"_a = 11, "append"_a = 12);
    m.def("points_to_voxel_3d",
          &sphconv::points_to_voxel_3d<double, 3>,
          "matrix tensor_square", "points"_a = 1, "system_points"_a = 2,
          "voxels"_a = 3,
          "voxel_point_mask"_a = 4, "coors"_a = 5, "num_points_per_voxel"_a = 6,
          "coor_to_voxelidx"_a = 7, "coors_range"_a = 8,
          "grid_size"_a = 9,
          "max_points"_a = 10, "max_voxels"_a = 11, "append"_a = 12);
    m.def("points_to_voxel_3d_np_mean",
          &sphconv::points_to_voxel_3d_np_mean<float, 3>, "matrix tensor_square",
          "points"_a = 1, "voxels"_a = 2, "voxel_point_mask"_a = 3, "means"_a = 4,
          "coors"_a = 5, "num_points_per_voxel"_a = 6, "coor_to_voxelidx"_a = 7,
          "voxel_size"_a = 8, "coors_range"_a = 9, "max_points"_a = 10,
          "max_voxels"_a = 11);
    m.def("points_to_voxel_3d_np_mean",
          &sphconv::points_to_voxel_3d_np_mean<double, 3>, "matrix tensor_square",
          "points"_a = 1, "voxels"_a = 2, "voxel_point_mask"_a = 3, "means"_a = 4,
          "coors"_a = 5, "num_points_per_voxel"_a = 6, "coor_to_voxelidx"_a = 7,
          "voxel_size"_a = 8, "coors_range"_a = 9, "max_points"_a = 10,
          "max_voxels"_a = 11);
    m.def("points_to_voxel_3d_with_filtering",
          &sphconv::points_to_voxel_3d_with_filtering<float, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "voxel_mask"_a = 4, "mins"_a = 5,
          "maxs"_a = 6, "coors"_a = 7, "num_points_per_voxel"_a = 8,
          "coor_to_voxelidx"_a = 9, "voxel_size"_a = 10, "coors_range"_a = 11,
          "max_points"_a = 12, "max_voxels"_a = 13, "block_factor"_a = 14,
          "block_size"_a = 15, "height_threshold"_a = 16,
          "height_high_threshold"_a = 17);
    m.def("points_to_voxel_3d_with_filtering",
          &sphconv::points_to_voxel_3d_with_filtering<float, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
          "voxel_point_mask"_a = 3, "voxel_mask"_a = 4, "mins"_a = 5,
          "maxs"_a = 6, "coors"_a = 7, "num_points_per_voxel"_a = 8,
          "coor_to_voxelidx"_a = 9, "voxel_size"_a = 10, "coors_range"_a = 11,
          "max_points"_a = 12, "max_voxels"_a = 13, "block_factor"_a = 14,
          "block_size"_a = 15, "height_threshold"_a = 16,
          "height_high_threshold"_a = 17);
}
