# read kitti point cloud
# see
#
import torch
import numpy as np
import pathlib
from pathlib import Path
import spconv
from spconv.utils import VoxelGeneratorV3
from sphconv.data import xyz2RangeVoxel, merge_rangevoxel_batch

from collections import defaultdict
import yaml


KITTI_DATASET_ROOT = "dataset/"
VOXEL_CONFIG = 'voxel.yaml'
RANGE_VOXEL_CONFIG = 'range_voxel.yaml'


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_velodyne_path(idx, prefix, training=True, relative_path=False, exist_check=True):
    """Get kitti's point cloud  path. eg. /dataset/training/velodyne/"""
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check)

# read point cloud bin file


def read_pc_data(idx: int, channel:int):
    """Read the 'idx' th  point cloud file from kitti.

        append random data so that data has 'channel' channels

    """
    velo_path = Path(get_velodyne_path(idx, KITTI_DATASET_ROOT))
    # velo_path = Path(KITTI_DATASET_ROOT) / velo_path
    velo_reduced_path = velo_path.parent.parent / \
        (velo_path.parent.stem + '_reduced') / velo_path.name
    if velo_reduced_path.exists():
        velo_path = velo_reduced_path
    print("velo_path ", str(velo_path))
    points = np.fromfile(str(velo_path), dtype=np.float32,
                         count=-1).reshape([-1, 4])
    print(points.shape)
    if channel > 4:
        # append some random data
        N = points.shape[0]
        # for reproducible tests
        np.random.seed(0);
        rand_feature = np.random.randn(N, channel).astype('f')
        points = np.concatenate((points, rand_feature), axis=1)

    return points


def merge_second_batch(batch_list):
    """Merge multiple pointcloud(already processed by point2voxel) to a batch."""
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes', 'points'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        elif key in ['points']:
            continue
        else:
            # print("elems.shape = ", elems.shape)
            print("key = ", key)
            continue
            ret[key] = np.stack(elems, axis=0)
    return ret


def point2voxel(points, voxel_generator):
    """Convert points to voxels, for spconv.

    Args:
        points (np.tensor of shape [M, NDIM]),
            points from one frame, M is the number of points.

        voxel_generator: see spconv.VoxelGenerator

    Return voxels and its coordinate
        a dictionary, with "voxels" and "coordinates" etc.

    """
    res = voxel_generator.generate(points, 20000)
    # voxels = res["voxels"]
    # coordinates = res["coordinates"]
    # num_points = res["num_points_per_voxel"]
    # num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    # print(voxels)
    # print(coordinates[])
    # print(res)

    return res


def read_config_file(config_file_path: str):
    with open(config_file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print(data)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def create_voxel_generator(config_file_path: str):
    """Create a **spherical** voxel generator based on a config file.

    Args:
        config_file_path (str), a yaml file

    """
    voxel_config = read_config_file(config_file_path)
    voxel_config = voxel_config['voxel_generator']

    voxel_generator = VoxelGeneratorV3(
        voxel_size=list(voxel_config["voxel_size"]),
        point_cloud_range=list(voxel_config["point_cloud_range"]),
        point_cloud_sphere_range=list(
            voxel_config["point_cloud_sphere_range"]),
        max_num_points=voxel_config['max_number_of_points_per_voxel'],
        max_voxels=20000)

    return voxel_generator


def get_range_voxels(idx,
                     batch_size=1,
                     channel=4,
                     range_voxel_config_path=RANGE_VOXEL_CONFIG):
    """Read point cloud from KITTI, return batched RangeVoxels.

    Args:
    ----
        idx (int): the start idx to read from
        batch_size (int): the number of file to read. consecutive
        channel (int): the number of channel

    """
    range_config = read_config_file(range_voxel_config_path)

    rangeV_list = []
    for i in range(idx, idx + batch_size):
        points = read_pc_data(i, channel)

        rangeV = xyz2RangeVoxel(points, **range_config)
        rangeV = rangeV.cuda()
        rangeV_list.append(rangeV)

    batched = merge_rangevoxel_batch(rangeV_list)
    print(batched)
    return batched


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    """Convert data from numpy to torch, move to GPU.
    """
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance",
        "feature"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def get_voxels(idx,
               batch_size=1,
               channel=4,
               voxel_generator_config_path=VOXEL_CONFIG):
    """ Read point cloud from KITTI, return batched voxels, for spconv

    Args:
    ----
        idx (int): the start idx to read from
        batch_size (int): the number of file to read. consecutive

    """
    voxel_generator = create_voxel_generator(voxel_generator_config_path)
    voxels_list = []
    for i in range(idx, idx + batch_size):
        points = read_pc_data(i, channel)

        voxels = point2voxel(points, voxel_generator=voxel_generator)
        voxels_list.append(voxels)

    batched = merge_second_batch(voxels_list)
    batched = example_convert_to_torch(batched)

    spatial_shape = voxel_generator.grid_size
    print("grid_size = {}".format(spatial_shape))

    res = spconv.SparseConvTensor(batched['voxels'],
                                  batched['coordinates'],
                                  spatial_shape,
                                  batch_size)
    print(res)
    return res
