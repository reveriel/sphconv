// Copyright 2019 Yan Yan,
//           2021 Guo Xing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>

namespace sphconv
{

    namespace py = pybind11;
    using namespace pybind11::literals;

    template <typename DType, int NDim>
    int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                              py::array_t<DType> voxel_point_mask, py::array_t<int> coors,
                              py::array_t<int> num_points_per_voxel,
                              py::array_t<int> coor_to_voxelidx,
                              std::vector<DType> voxel_size,
                              std::vector<DType> coors_range, int max_points,
                              int max_voxels)
    {
        auto points_rw = points.template mutable_unchecked<2>();
        auto voxels_rw = voxels.template mutable_unchecked<3>();
        auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
        auto coors_rw = coors.mutable_unchecked<2>();
        auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
        auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
        auto N = points_rw.shape(0);
        auto num_features = points_rw.shape(1);
        // auto ndim = points_rw.shape(1) - 1;
        constexpr int ndim_minus_1 = NDim - 1;
        int voxel_num = 0;
        bool failed = false;
        int coor[NDim];
        int c;
        int grid_size[NDim];
        for (int i = 0; i < NDim; ++i)
        {
            grid_size[i] =
                round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
        }
        int voxelidx, num;
        for (int i = 0; i < N; ++i)
        {
            failed = false;
            for (int j = 0; j < NDim; ++j)
            {
                c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
                if ((c < 0 || c >= grid_size[j]))
                {
                    failed = true;
                    break;
                }
                coor[ndim_minus_1 - j] = c;
            }
            if (failed)
                continue;
            voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
            if (voxelidx == -1)
            {
                voxelidx = voxel_num;
                if (voxel_num >= max_voxels)
                    break;
                voxel_num += 1;
                coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
                for (int k = 0; k < NDim; ++k)
                {
                    coors_rw(voxelidx, k) = coor[k];
                }
            }
            num = num_points_per_voxel_rw(voxelidx);
            if (num < max_points)
            {
                voxel_point_mask_rw(voxelidx, num) = DType(1);
                for (int k = 0; k < num_features; ++k)
                {
                    voxels_rw(voxelidx, num, k) = points_rw(i, k);
                }
                num_points_per_voxel_rw(voxelidx) += 1;
            }
        }
        for (int i = 0; i < voxel_num; ++i)
        {
            coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
        }
        return voxel_num;
    }

    template <typename DType, int NDim>
    int points_to_voxel_3d_np_mean(py::array_t<DType> points,
                                   py::array_t<DType> voxel_point_mask, py::array_t<DType> voxels,
                                   py::array_t<DType> means, py::array_t<int> coors,
                                   py::array_t<int> num_points_per_voxel,
                                   py::array_t<int> coor_to_voxelidx,
                                   std::vector<DType> voxel_size,
                                   std::vector<DType> coors_range, int max_points,
                                   int max_voxels)
    {
        auto points_rw = points.template mutable_unchecked<2>();
        auto means_rw = means.template mutable_unchecked<2>();
        auto voxels_rw = voxels.template mutable_unchecked<3>();
        auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
        auto coors_rw = coors.mutable_unchecked<2>();
        auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
        auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
        auto N = points_rw.shape(0);
        auto num_features = points_rw.shape(1);
        // auto ndim = points_rw.shape(1) - 1;
        constexpr int ndim_minus_1 = NDim - 1;
        int voxel_num = 0;
        bool failed = false;
        int coor[NDim];
        int c;
        int grid_size[NDim];
        for (int i = 0; i < NDim; ++i)
        {
            grid_size[i] =
                round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
        }
        int voxelidx, num;
        for (int i = 0; i < N; ++i)
        {
            failed = false;
            for (int j = 0; j < NDim; ++j)
            {
                c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
                if ((c < 0 || c >= grid_size[j]))
                {
                    failed = true;
                    break;
                }
                coor[ndim_minus_1 - j] = c;
            }
            if (failed)
                continue;
            voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
            if (voxelidx == -1)
            {
                voxelidx = voxel_num;
                if (voxel_num >= max_voxels)
                    break;
                voxel_num += 1;
                coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
                for (int k = 0; k < NDim; ++k)
                {
                    coors_rw(voxelidx, k) = coor[k];
                }
            }
            num = num_points_per_voxel_rw(voxelidx);
            if (num < max_points)
            {
                voxel_point_mask_rw(voxelidx, num) = DType(1);
                for (int k = 0; k < num_features; ++k)
                {
                    voxels_rw(voxelidx, num, k) = points_rw(i, k);
                }
                num_points_per_voxel_rw(voxelidx) += 1;
                for (int k = 0; k < num_features; ++k)
                {
                    means_rw(voxelidx, k) +=
                        (points_rw(i, k) - means_rw(voxelidx, k)) / DType(num + 1);
                }
            }
        }
        for (int i = 0; i < voxel_num; ++i)
        {
            coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
            num = num_points_per_voxel_rw(i);
            for (int j = num; j < max_points; ++j)
            {
                for (int k = 0; k < num_features; ++k)
                {
                    voxels_rw(i, j, k) = means_rw(i, k);
                }
            }
        }
        return voxel_num;
    }

    template <typename DType, int NDim>
    int points_to_voxel_3d_with_filtering(
        py::array_t<DType> points, py::array_t<DType> voxels,
        py::array_t<DType> voxel_point_mask, py::array_t<int> voxel_mask, py::array_t<DType> mins,
        py::array_t<DType> maxs, py::array_t<int> coors,
        py::array_t<int> num_points_per_voxel, py::array_t<int> coor_to_voxelidx,
        std::vector<DType> voxel_size, std::vector<DType> coors_range,
        int max_points, int max_voxels, int block_factor, int block_size,
        DType height_threshold, DType height_high_threshold)
    {
        auto points_rw = points.template mutable_unchecked<2>();
        auto mins_rw = mins.template mutable_unchecked<2>();
        auto maxs_rw = maxs.template mutable_unchecked<2>();
        auto voxels_rw = voxels.template mutable_unchecked<3>();
        auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
        auto voxel_mask_rw = voxel_mask.template mutable_unchecked<1>();
        auto coors_rw = coors.mutable_unchecked<2>();
        auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
        auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
        auto N = points_rw.shape(0);
        auto num_features = points_rw.shape(1);
        // auto ndim = points_rw.shape(1) - 1;
        constexpr int ndim_minus_1 = NDim - 1;
        int voxel_num = 0;
        bool failed = false;
        int coor[NDim];
        int c;
        int grid_size[NDim];

        DType max_value, min_value;
        for (int i = 0; i < NDim; ++i)
        {
            grid_size[i] =
                round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
        }
        int block_shape_H = grid_size[1] / block_factor;
        int block_shape_W = grid_size[0] / block_factor;
        int voxelidx, num;
        int block_coor[2];
        int startx, stopx, starty, stopy;
        for (int i = 0; i < N; ++i)
        {
            failed = false;
            for (int j = 0; j < NDim; ++j)
            {
                c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
                if ((c < 0 || c >= grid_size[j]))
                {
                    failed = true;
                    break;
                }
                coor[ndim_minus_1 - j] = c;
            }
            if (failed)
                continue;
            voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
            if (voxelidx == -1)
            {
                voxelidx = voxel_num;
                if (voxel_num >= max_voxels)
                    break;
                voxel_num += 1;
                coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
                for (int k = 0; k < NDim; ++k)
                {
                    coors_rw(voxelidx, k) = coor[k];
                }
            }
            num = num_points_per_voxel_rw(voxelidx);
            if (num < max_points)
            {
                voxel_point_mask_rw(voxelidx, num) = DType(1);
                for (int k = 0; k < num_features; ++k)
                {
                    voxels_rw(voxelidx, num, k) = points_rw(i, k);
                }
                block_coor[0] = coor[1] / block_factor;
                block_coor[1] = coor[2] / block_factor;
                mins_rw(block_coor[0], block_coor[1]) =
                    std::min(points_rw(i, 2), mins_rw(block_coor[0], block_coor[1]));
                maxs_rw(block_coor[0], block_coor[1]) =
                    std::max(points_rw(i, 2), maxs_rw(block_coor[0], block_coor[1]));
                num_points_per_voxel_rw(voxelidx) += 1;
            }
        }
        for (int i = 0; i < voxel_num; ++i)
        {
            coor[1] = coors_rw(i, 1);
            coor[2] = coors_rw(i, 2);
            coor_to_voxelidx_rw(coors_rw(i, 0), coor[1], coor[2]) = -1;
            block_coor[0] = coor[1] / block_factor;
            block_coor[1] = coor[2] / block_factor;
            min_value = mins_rw(block_coor[0], block_coor[1]);
            max_value = maxs_rw(block_coor[0], block_coor[1]);
            startx = std::max(0, block_coor[0] - block_size / 2);
            stopx =
                std::min(block_shape_H, block_coor[0] + block_size - block_size / 2);
            starty = std::max(0, block_coor[1] - block_size / 2);
            stopy =
                std::min(block_shape_W, block_coor[1] + block_size - block_size / 2);

            for (int j = startx; j < stopx; ++j)
            {
                for (int k = starty; k < stopy; ++k)
                {
                    min_value = std::min(min_value, mins_rw(j, k));
                    max_value = std::max(max_value, maxs_rw(j, k));
                }
            }
            voxel_mask_rw(i) = ((max_value - min_value) > height_threshold) &&
                               ((max_value - min_value) < height_high_threshold);
        }
        return voxel_num;
    }

    template <typename DType, int NDim>
    int points_to_voxel_3d_sphere_np(py::array_t<DType> points, py::array_t<DType> voxels,
                                     py::array_t<DType> voxel_point_mask, py::array_t<int> coors,
                                     py::array_t<int> num_points_per_voxel,
                                     py::array_t<int> coor_to_voxelidx,
                                     std::vector<DType> voxel_size,
                                     std::vector<DType> coors_range, int max_points,
                                     int max_voxels)
    {
        auto points_rw = points.template mutable_unchecked<2>();
        auto voxels_rw = voxels.template mutable_unchecked<3>();
        auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
        auto coors_rw = coors.mutable_unchecked<2>();
        auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
        auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
        auto N = points_rw.shape(0);
        auto num_features = points_rw.shape(1);

        int voxel_num = 0;
        bool failed = false;
        int coor[3];
        int grid_size[NDim];
        for (int i = 0; i < NDim; ++i)
        {
            grid_size[i] = round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
        }
        // std::cout << "grid_size = " << grid_size[0] <<","<<grid_size[1] <<","<<grid_size[2]<<std::endl;

        int voxelidx, num;
        for (int i = 0; i < N; ++i)
        {
            // const DType delta_phi = 0.0030679615757712823; // np.radians(90./512.)
            // const DType delta_theta = 0.007308566242726255; //  np.radians(26.8/64.)
            // DType delta_r = (coors_range[1] - coors_range[0]) / voxel_size[0];
            DType x = points_rw(i, 0);
            DType y = points_rw(i, 1);
            DType z = points_rw(i, 2);

            DType x2y2 = x * x + y * y;
            DType r = std::sqrt(x2y2 + z * z);
            DType theta = std::acos(z / r);
            DType phi = std::asin(y / std::sqrt(x2y2));

            // theta, phi, r
            // r phi theta
            coor[0] = floor((theta - coors_range[2]) / voxel_size[2]);
            if (coor[0] < 0 || coor[0] >= grid_size[2])
                continue;
            coor[1] = floor((phi - coors_range[1]) / voxel_size[1]);
            if (coor[1] < 0 || coor[1] >= grid_size[1])
                continue;
            coor[2] = floor((std::log(r) - coors_range[0]) / voxel_size[0]);
            if (coor[2] < 0 || coor[2] >= grid_size[0])
                continue;

            voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
            if (voxelidx == -1)
            {
                voxelidx = voxel_num;
                if (voxel_num >= max_voxels)
                    break;
                voxel_num += 1;
                coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
                for (int k = 0; k < NDim; ++k)
                {
                    coors_rw(voxelidx, k) = coor[k];
                }
            }
            num = num_points_per_voxel_rw(voxelidx);
            if (num < max_points)
            {
                voxel_point_mask_rw(voxelidx, num) = DType(1);
                for (int k = 0; k < num_features; ++k)
                {
                    voxels_rw(voxelidx, num, k) = points_rw(i, k);
                }
                num_points_per_voxel_rw(voxelidx) += 1;
            }
        }
        for (int i = 0; i < voxel_num; ++i)
        {
            coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
        }
        return voxel_num;
    }

    /**
 * @arg points : xyzr
 * @arg system_points : points coords in its specific coordinate
 *      NDim == 3
 *
 * @arg append_mode:
 *        case 0: // only xyzr feature, default
 *        case 1: // xyzr plus new coordinates
 *        case 2: // new coordinates + r
 *
**/
    template <typename DType, int NDim>
    int points_to_voxel_3d(py::array_t<DType> points,
                           py::array_t<DType> system_points,
                           py::array_t<DType> voxels,
                           py::array_t<DType> voxel_point_mask, py::array_t<int> coors,
                           py::array_t<int> num_points_per_voxel,
                           py::array_t<int> coor_to_voxelidx,
                           std::vector<DType> coors_range,
                           std::vector<int> grid_size,
                           int max_points,
                           int max_voxels,
                           int append_mode)
    {
        auto points_rw = points.template mutable_unchecked<2>();
        auto system_points_rw = system_points.template mutable_unchecked<2>();
        auto voxels_rw = voxels.template mutable_unchecked<3>();
        auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
        auto coors_rw = coors.mutable_unchecked<2>();
        auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
        auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
        auto N = points_rw.shape(0);
        auto num_features = points_rw.shape(1);

        int voxel_num = 0;
        int coor[3];

        DType voxel_size[NDim];
        for (int i = 0; i < NDim; i++)
        {
            voxel_size[i] = (coors_range[NDim + i] - coors_range[i]) / grid_size[i];
        }

        int voxelidx, num;
        for (int i = 0; i < N; ++i)
        {
            DType x = system_points_rw(i, 0);
            DType y = system_points_rw(i, 1);
            DType z = system_points_rw(i, 2);

            // theta, phi, r
            // r phi theta
            coor[0] = floor((z - coors_range[2]) / voxel_size[2]);
            if (coor[0] < 0 || coor[0] >= grid_size[2])
                continue;
            coor[1] = floor((y - coors_range[1]) / voxel_size[1]);
            if (coor[1] < 0 || coor[1] >= grid_size[1])
                continue;
            coor[2] = floor((x - coors_range[0]) / voxel_size[0]);
            if (coor[2] < 0 || coor[2] >= grid_size[0])
                continue;

            voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
            if (voxelidx == -1)
            {
                voxelidx = voxel_num;
                if (voxel_num >= max_voxels)
                    break;
                voxel_num += 1;
                coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
                for (int k = 0; k < NDim; ++k)
                {
                    coors_rw(voxelidx, k) = coor[k];
                }
            }
            num = num_points_per_voxel_rw(voxelidx);
            if (num < max_points)
            {
                voxel_point_mask_rw(voxelidx, num) = DType(1);
                switch (append_mode)
                {
                case 0: // only xyzr feature
                    for (int k = 0; k < num_features; ++k)
                    {
                        voxels_rw(voxelidx, num, k) = points_rw(i, k);
                    }
                    break;
                case 1: // xyzr plus new coordinates
                    for (int k = 0; k < num_features; ++k)
                    {
                        voxels_rw(voxelidx, num, k) = points_rw(i, k);
                    }
                    for (int k = 0; k < NDim; ++k)
                    {
                        voxels_rw(voxelidx, num, num_features + k) = system_points_rw(i, k);
                    }
                    break;
                case 2: // new coordinates + r
                    for (int k = 0; k < NDim; ++k)
                    {
                        voxels_rw(voxelidx, num, k) = system_points_rw(i, k);
                    }
                    voxels_rw(voxelidx, num, NDim) = points_rw(i, NDim);
                }
                num_points_per_voxel_rw(voxelidx) += 1;
            }
        }
        for (int i = 0; i < voxel_num; ++i)
        {
            coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
        }
        return voxel_num;
    }

} // namespace spconv