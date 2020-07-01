# read kitti point cloud
# see
#
import numpy as np
import pathlib
from pathlib import Path
# from spconv import
from spconv.utils import VoxelGeneratorV3

import yaml


KITTI_DATASET_ROOT = "dataset/"
VOXEL_CONFIG = 'config.yaml'


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
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check)

# read point cloud bin file
def read_pc_data(idx: int):
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


points = read_pc_data(3)

# voxelize

# voxel_config =

with open(VOXEL_CONFIG, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
        print(data)
        voxel_config = data['voxel_generator']
    except yaml.YAMLError as exc:
        print(exc)

voxel_generator = VoxelGeneratorV3(
    voxel_size=list(voxel_config["voxel_size"]),
    point_cloud_range=list(voxel_config["point_cloud_range"]),
    point_cloud_sphere_range=list(voxel_config["point_cloud_sphere_range"]),
    max_num_points=voxel_config['max_number_of_points_per_voxel'],
    max_voxels=20000)


res = voxel_generator.generate(points, 20000)
voxels = res["voxels"]
coordinates = res["coordinates"]
num_points = res["num_points_per_voxel"]
num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

# print(voxels)
# print(coordinates[])
print(res)
