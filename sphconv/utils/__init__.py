# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Tuple
import numpy as np

from sphconv.sphconv_utils import (points_to_voxel_3d_np,
                                   points_to_voxel_3d_sphere_np, points_to_voxel_3d,
                                   points_to_voxel_3d_np_mean, points_to_voxel_3d_with_filtering)

def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=35,
                    max_voxels=20000,
                    full_mean=False,
                    block_filtering=True,
                    block_factor=1,
                    block_size=8,
                    height_threshold=0.2,
                    height_high_threshold=3.0,
                    pad_output=False):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels)
    with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
        full_mean: bool. if true, all empty points in voxel will be filled with mean
            of exist points.
        block_filtering: filter voxels by height. used for lidar point cloud.
            use some visualization tool to see filtered result.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if full_mean:
        assert block_filtering is False
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    voxel_point_mask = np.zeros(
        shape=(max_voxels, max_points), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    res = {
        "voxels": voxels,
        "coordinates": coors,
        "num_points_per_voxel": num_points_per_voxel,
        "voxel_point_mask": voxel_point_mask,
    }
    if full_mean:
        means = np.zeros(
            shape=(max_voxels, points.shape[-1]), dtype=points.dtype)
        voxel_num = points_to_voxel_3d_np_mean(
            points, voxels, voxel_point_mask, means, coors,
            num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
            coors_range.tolist(), max_points, max_voxels)
    else:
        if block_filtering:
            block_shape = [*voxelmap_shape[1:]]
            block_shape = [b // block_factor for b in block_shape]
            mins = np.full(block_shape, 99999999, dtype=points.dtype)
            maxs = np.full(block_shape, -99999999, dtype=points.dtype)
            voxel_mask = np.zeros((max_voxels, ), dtype=np.int32)
            voxel_num = points_to_voxel_3d_with_filtering(
                points, voxels, voxel_point_mask, voxel_mask, mins, maxs,
                coors, num_points_per_voxel, coor_to_voxelidx,
                voxel_size.tolist(), coors_range.tolist(), max_points,
                max_voxels, block_factor, block_size, height_threshold,
                height_high_threshold)
            voxel_mask = voxel_mask.astype(np.bool_)
            coors_ = coors[voxel_mask]
            if pad_output:
                res["coordinates"][:voxel_num] = coors_
                res["voxels"][:voxel_num] = voxels[voxel_mask]
                res["voxel_point_mask"][:voxel_num] = voxel_point_mask[
                    voxel_mask]

                res["num_points_per_voxel"][:voxel_num] = num_points_per_voxel[
                    voxel_mask]
                res["coordinates"][voxel_num:] = 0
                res["voxels"][voxel_num:] = 0
                res["num_points_per_voxel"][voxel_num:] = 0
                res["voxel_point_mask"][voxel_num:] = 0
            else:
                res["coordinates"] = coors_
                res["voxels"] = voxels[voxel_mask]
                res["num_points_per_voxel"] = num_points_per_voxel[voxel_mask]
                res["voxel_point_mask"] = voxel_point_mask[voxel_mask]
            voxel_num = coors_.shape[0]
        else:
            voxel_num = points_to_voxel_3d_np(
                points, voxels, voxel_point_mask, coors,
                num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
                coors_range.tolist(), max_points, max_voxels)
    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
        -1, max_points, 1)
    return res


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=True):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean)
        voxels = res["voxels"]
        coors = res["coordinates"]
        num_points_per_voxel = res["num_points_per_voxel"]
        voxel_num = res["voxel_num"]
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return (voxels, coors, num_points_per_voxel)

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean)
        voxels = res["voxels"]
        coors = res["coordinates"]
        num_points_per_voxel = res["num_points_per_voxel"]
        voxel_num = res["voxel_num"]
        return (voxels, coors, num_points_per_voxel)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


class VoxelGeneratorV2:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=False,
                 block_filtering=False,
                 block_factor=8,
                 block_size=3,
                 height_threshold=0.1,
                 height_high_threshold=2.0):
        assert full_mean is False, "don't use this."
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        if block_filtering:
            assert block_size > 0
            assert grid_size[0] % block_factor == 0
            assert grid_size[1] % block_factor == 0

        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean
        self._block_filtering = block_filtering
        self._block_factor = block_factor
        self._height_threshold = height_threshold
        self._block_size = block_size
        self._height_high_threshold = height_high_threshold

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range,
            self._coor_to_voxelidx, self._max_num_points, max_voxels
            or self._max_voxels, self._full_mean, self._block_filtering,
            self._block_factor, self._block_size, self._height_threshold,
            self._height_high_threshold)
        for k, v in res.items():
            if k != "voxel_num":
                res[k] = v[:res["voxel_num"]]
        return res

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._coor_to_voxelidx,
            self._max_num_points,
            max_voxels or self._max_voxels,
            self._full_mean,
            self._block_filtering,
            self._block_factor,
            self._block_size,
            self._height_threshold,
            self._height_high_threshold,
            pad_output=True)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


class VoxelGeneratorV3:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 point_cloud_sphere_range,
                 max_num_points,
                 max_voxels=30000,
                 resolution: List[int] = None,
                 coord_system=None,
                 append_mode=0):
        """
        resolution, or grid_size
        """

        point_cloud_sphere_range = np.array(
            point_cloud_sphere_range, dtype=np.float32)
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)

        # compute grid_size
        if resolution is not None:
            grid_size = np.array(resolution, dtype=np.int64)
        else:
            grid_size = (
                point_cloud_sphere_range[3:] - point_cloud_sphere_range[:3]) / voxel_size
            grid_size = np.round(grid_size).astype(np.int64)

        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        print("voxelmap_shape = ", voxelmap_shape)

        self._max_voxels = max_voxels
        self._max_ponts = max_num_points

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._point_cloud_sphere_range = point_cloud_sphere_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        # self._full_mean = full_mean
        self._resolution = resolution
        self._coord_system = coord_system
        print("self._coord_system =", coord_system)
        self._append_mode = append_mode

    def generate(self, points, max_voxels=None):
        res = points_to_voxel_v2(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._point_cloud_sphere_range,
            self._coor_to_voxelidx,
            self._max_num_points,
            max_voxels or self._max_voxels,
            resolution=self._resolution,
            coord_system=self._coord_system,
            append_mode=self._append_mode)

        for k, v in res.items():
            if k != "voxel_num":
                res[k] = v[:res["voxel_num"]]
        return res

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel_v2(points,
                                 self._voxel_size,
                                 self._point_cloud_sphere_range,
                                 self._coor_to_voxelidx,
                                 self._max_num_points, max_voxels or self._max_voxels,
                                 pad_output=True
                                 )
        # print("res voels .shape  = ", res["voxels"].shape)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


# transform_func = transform_funcs[coord_system];
# system_points, system_range = transform_func(points, coors_range)

def spherical_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([sph_range[0],
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          sph_range[3],
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([r, phi, theta, intensity], 1).astype(points.dtype), sph_range


def lspherical_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)
    sph_range = np.array([np.log(sph_range[0]),
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          np.log(sph_range[3]),
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([np.log(r), phi, theta, intensity], 1).astype(points.dtype), sph_range


def cylinder_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    coor_range = np.array([sph_range[0],
                           np.radians(sph_range[1]),
                           v_range[2],
                           sph_range[3],
                           np.radians(sph_range[4]),
                           v_range[5]])
    return np.stack([d, phi, z, intensity], 1).astype(points.dtype), coor_range


def lcylinder_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    coor_range = np.array([np.log(sph_range[0]),
                           np.radians(sph_range[1]),
                           v_range[2],
                           np.log(sph_range[3]),
                           np.radians(sph_range[4]),
                           v_range[5]])
    return np.stack([np.log(d), phi, z, intensity], 1).astype(points.dtype), coor_range


def hybrid_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([sph_range[0],
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          sph_range[3],
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([d, phi, theta, intensity], 1).astype(points.dtype), sph_range


def lhybrid_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([np.log(sph_range[0]),
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          np.log(sph_range[3]),
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([np.log(d), phi, theta, intensity], 1).astype(dtype=points.dtype), sph_range


transform_funcs = {
    "spherical": spherical_transform,
    "l-spherical": lspherical_transform,
    "cylinder": cylinder_transform,
    "l-cylinder": lcylinder_transform,
    "hybrid": hybrid_transform,
    "l-hybrid": lhybrid_transform
}


def points_to_voxel_v2(points,
                       voxel_size,
                       coors_range,
                       sphere_coors_range,
                       coor_to_voxelidx,
                       max_points=1,
                       max_voxels=30000,
                       pad_output=False,
                       resolution: List[int] = None,
                       coord_system: str = None,
                       append_mode: int = 0
                       ):
    """
    convert 3d points(x,y,z), to spherical voxels,

        coors_range, in degree
        resolution: 3d resolution in (xyz), or (r, phi, theta) or (r, phi, h)
        coord_system: one of
            "spherical" or None : default
            "l-spherical"
            "cylinder"
            "l-cylinder"
            "hybrid"
            "l-hybrid"
    """

    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    voxels_feature_dim = points.shape[-1]
    if append_mode == 1:
        voxels_feature_dim += 3
    voxels = np.zeros(
        shape=(max_voxels, max_points, voxels_feature_dim), dtype=points.dtype)

    voxel_point_mask = np.zeros(
        shape=(max_voxels, max_points), dtype=points.dtype)

    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    res = {
        "voxels": voxels,
        "coordinates": coors,
        "num_points_per_voxel": num_points_per_voxel,
        "voxel_point_mask": voxel_point_mask,
    }

    if resolution is None:
        # "spherical"

        # convert degree to rad
        rad_voxel_size = np.array(
            [voxel_size[0], np.radians(voxel_size[1]), np.radians(voxel_size[2])])
        rad_coors_range = np.array([
            sphere_coors_range[0],
            np.radians(sphere_coors_range[1]),
            np.radians(sphere_coors_range[2]),
            sphere_coors_range[3],
            np.radians(sphere_coors_range[4]),
            np.radians(sphere_coors_range[5])
        ])
        # print("rad_voxel_size,=", rad_voxel_size)
        # print("radcoors_range=", rad_coors_range)
        voxel_num = points_to_voxel_3d_sphere_np(
            points, voxels, voxel_point_mask, coors,
            num_points_per_voxel, coor_to_voxelidx,
            rad_voxel_size.tolist(),
            rad_coors_range.tolist(),
            max_points, max_voxels)

        # print("voxel_num = ", voxel_num)
    else:
        if coord_system in transform_funcs:
            transform_func = transform_funcs[coord_system]
            system_points, system_range = transform_func(
                points, coors_range, sphere_coors_range)
            voxel_num = points_to_voxel_3d(
                points, system_points, voxels, voxel_point_mask, coors,
                num_points_per_voxel, coor_to_voxelidx,
                system_range,
                resolution,
                max_points, max_voxels, append_mode)
        else:
            print("unkown coord_system")
            exit(-1)

    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
        -1, max_points, 1)
    return res






from torch._six import container_abcs
from itertools import repeat, product


def _triple(x):
    """If x is a single number, repeat three times."""
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 3))


# copy form pytorch
def _calculate_fan_in_and_fan_out_hwio(tensor):
    """Init convolution weight. Copied from pytorch."""
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out