
from typing import List

import numpy as np
from torch import dtype
from sphconv.sphconv_utils import (points_to_voxel_3d, points_to_voxel_3d_np,
                                   points_to_voxel_3d_np_mean,
                                   points_to_voxel_3d_sphere_np,
                                   points_to_voxel_3d_with_filtering)
from sphconv.utils.coord_transform import transform_funcs


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


def points_to_voxel_v2(points,  # xyz
                       voxel_size,
                       coors_range,
                       sphere_coors_range,
                       coor_to_voxelidx,
                       max_points=1,
                       max_voxels=30000,
                       pad_output=False,
                       resolution_HWD: List[int] = None,
                       coord_system: str = None,
                       append_mode: int = 0
                       ):
    """
    convert 3d points(x,y,z), to voxels in.. any coordiante system,

        coors_range, in degree
        resolution_HWD: 3d resolution in (xyz), or (r, phi, theta) or (r, phi, h)
        coord_system: one of
            "spherical" or None : default
            "l-spherical"
            "cylinder"
            "l-cylinder"
            "hybrid"
            "l-hybrid"
            "cartesian"
    """

    # input are points
    #
    # points
    # we do avg pooling here.

    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)

    voxels_feature_dim = points.shape[-1]
    if append_mode == 1:
        voxels_feature_dim += 3

    voxels = np.zeros(
        shape=(max_voxels, max_points, voxels_feature_dim), dtype=points.dtype)

    val = np.zeros(shape=(max_voxels, voxels_feature_dim), dtype=points.dtype)

    voxel_point_mask = np.zeros(
        shape=(max_voxels, max_points), dtype=points.dtype)

    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    res = {
        "voxels": voxels,
        "coordinates": coors,
        "num_points_per_voxel": num_points_per_voxel,
        "voxel_point_mask": voxel_point_mask,
    }

    if resolution_HWD is None:
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
                resolution_HWD,
                max_points, max_voxels, append_mode)
        else:
            print("unkown coord_system")
            exit(-1)

    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
        -1, max_points, 1)
    return res
