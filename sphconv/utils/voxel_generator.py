from typing import List

import numpy as np
from sphconv.utils.voxelize import points_to_voxel, points_to_voxel_v2

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
        resolution:, or grid_size, is the dresolution of generated voxels.
            if resolution is not provided, we use voxel_size to calcluate it.
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

