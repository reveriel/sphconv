# read kitti point cloud
# see
#
import numpy as np
import pathlib
from pathlib import Path
# from spconv import
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


def read_pc_data(idx: int):
    """Read the 'idx' th  point cloud file from kitti."""
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


def get_range_voxels(idx, batch_size=1, range_voxel_config_path=RANGE_VOXEL_CONFIG):
    """ Read point cloud from KITTI, return batched RangeVoxels

    Args:
    ----
        idx (int): the start idx to read from
        batch_size (int): the number of file to read. consecutive

    """
    range_config = read_config_file(range_voxel_config_path)

    rangeV_list = []
    for i in range(idx, idx + batch_size):
        points = read_pc_data(i)
        rangeV = xyz2RangeVoxel(points, **range_config)
        rangeV_list.append(rangeV)
    batched = merge_rangevoxel_batch(rangeV_list)
    print(batched)
    return batched


def get_voxels(idx, batch_size=1, voxel_generator_config_path=VOXEL_CONFIG):
    """ Read point cloud from KITTI, return batched voxels, for spconv

    Args:
    ----
        idx (int): the start idx to read from
        batch_size (int): the number of file to read. consecutive

    """
    voxel_generator = create_voxel_generator(voxel_generator_config_path)
    voxels_list = []
    for i in range(idx, idx + batch_size):
        points = read_pc_data(i)
        voxels = point2voxel(points, voxel_generator=voxel_generator)
        voxels_list.append(voxels)

    batched = merge_second_batch(voxels_list)
    print(batched)
    return batched
