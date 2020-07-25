
import numpy as np
import torch
from sphconv import RangeVoxel
import spconv
from spconv.utils import VoxelGeneratorV3
import yaml

VOXEL_CONFIG = 'voxel.yaml'

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

def generate_test_image(B, C, D, H, W, T):
    """
        B: batchsize
        C, channel
        D, H, W : depth, height, width
        T: thickness, the max number of non-empty voxels on Depth
    """
    assert T <= D, "T must be less then D, got T={}, D={}".format(T, D)
    feature = torch.randn(B, C, T, H, W)
    depth = torch.randint(D, (B, T, H, W))
    thick = torch.randint(T, (B, H, W))

    # sort depth
    for b in range(B):
        for i in range(H):
            for j in range(W):
                values, _ = torch.sort(depth[b, :, i, j])
                depth[b, :, i, j] = values

    return RangeVoxel(feature, depth, thick, (B, C, D, H, W))


def test_dense():
    B, D, H, W, C, T = 2, 4, 3, 3, 1, 2
    img = generate_test_image(B, C, D, H, W, T)
    print(check(img))


def check(input: RangeVoxel):
    B, C, T, H, W = input.feature.shape
    D = input.shape[2]

    dense = input.dense()
    # print("dense .shape =", dense.shape)
    # print("dense", dense.reshape((2, 4, 3, 3)))
    # print("depth = ", input.depth)
    # if abs(dense.sum() - input.feature.sum()) > 1e7:
    #     return False
    for b in range(B):
        for x in range(H):
            for y in range(W):
                for t in range(input.thick[b, x, y]):
                    z = input.depth[b, t, x, y]
                    if sum(abs(dense[b, :, z, x, y] - input.feature[b, :, t, x, y])) > 1e-6:
                        print("dense[", b, ":", z, x, y, "]=",
                              dense[b, :, z, x, y])
                        print("feature[", b, ":", t, x, y, "]=",
                              input.feature[b, :, t, x, y])
                        return False
    return True

# class SparseConvTensor(object):
#     def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
#         """
#         Args:
#             grid: pre-allocated grid tensor. should be used when the volume of spatial shape
#                 is very large.
#         """
#         self.features = features
#         self.indices = indices
#         if self.indices.dtype != torch.int32:
#             self.indices.int()
#         self.spatial_shape = spatial_shape
#         self.batch_size = batch_size
#         self.indice_dict = {}
#         self.grid = grid

def RangeVoxel2SparseTensor(input) -> spconv.SparseConvTensor :
    """ convert RangeVoxel to sponcv.SparseConvTensor """

    feature = input.feature
    depth = input.depth
    thick = input.thick
    B, C, D, H, W = input.shape
    T = feature.shape[2]
    print("bcdhw", B, C, D, H, W)

    spconv_feature = []
    spconv_indices = []

    for b in range(B):
        for x in range(H):
            for y in range(W):
                for t in range(thick[b,x,y]):
                    z = depth[b, t, x, y]
                    # if z == 0:
                    #     continue
                    spconv_feature.append(feature[b, :, t, x, y])
                    indice = torch.tensor(
                        list([b, z, x, y]), dtype=torch.int32)
                    spconv_indices.append(indice)

    spatial_shape = (D, H, W)
    spconv_feature = torch.stack(spconv_feature, dim=0)
    spconv_indices = torch.stack(spconv_indices, dim=0)

    return spconv.SparseConvTensor(spconv_feature, spconv_indices, spatial_shape, B)




