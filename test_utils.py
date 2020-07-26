
import numpy as np
import torch
from sphconv import RangeVoxel
import spconv
# from spconv.utils import VoxelGeneratorV3
import yaml

VOXEL_CONFIG = 'voxel.yaml'


class VoxelGeneratorV3:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 point_cloud_sphere_range,
                 max_num_points,
                 max_voxels=30000):
        point_cloud_sphere_range = np.array(point_cloud_sphere_range, dtype=np.float32)
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
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

    def generate(self, points, max_voxels):
        res = points_to_voxel_v2(
            points,
            self._voxel_size,
            self._point_cloud_sphere_range,
            self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)

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


def points_to_voxel_v2(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=1,
                    max_voxels=30000,
                    pad_output=False):
    """
    convert 3d points(x,y,z), to spherical voxels,
    """

    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
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


    rad_voxel_size = np.array([voxel_size[0],
        np.radians(voxel_size[1]), np.radians(voxel_size[2])
        ])
    rad_coors_range = np.array([
        coors_range[0],
        np.radians(coors_range[1]),
        np.radians(coors_range[2]),
        coors_range[3],
        np.radians(coors_range[4]),
        np.radians(coors_range[5])
    ])
    # print("rad_voxel_size,=", rad_voxel_size)
    # print("radcoors_range=", rad_coors_range)

    voxel_num = points_to_voxel_3d_sphere_np(
        points, voxels, voxel_point_mask, coors,
        num_points_per_voxel, coor_to_voxelidx, rad_voxel_size.tolist(),
        rad_coors_range.tolist(), max_points, max_voxels)
    # print("voxel_num = ", voxel_num)

    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(
        -1, max_points, 1)
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




