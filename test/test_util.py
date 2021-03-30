# from sphconv.utils import *

# point to raw voxelFeature
# raw voxelFeature processed with VFE
# voxel feature convert to sphconv.Feature and spconv.SparseTensor

# to dense to check eqaulity

# run convolution on both.

from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch

POINTS_FILE = "000003.bin"


def batch_artifical_inputs(
    indices_zyx: torch.Tensor,
    channel: int,
    batch_size: int
):
    """
    create batched inputs from indices_zyx
    """
    features = torch.randn(
        (indices_zyx.shape[0], channel), dtype=torch.float, device=indices_zyx.device)

    one_example = {'voxel': features, 'coordinates': indices_zyx}
    example = merge_batch_torch([one_example] * batch_size)

    return example['voxel'], example['coordinates']


def assert_dense_eq(
        features: torch.Tensor,
        indices_zyx: torch.Tensor,
        batch_size: int = 1,
        spatial_shape_DWH: List[int] = []):
    """
    assert dense is coorect
    """

    assert features.dim() == 2
    assert indices_zyx.shape[1] == 4  # four coodinates
    assert indices_zyx.dim() == 2

    spconv_tensor = spconv.SparseConvTensor(
        features, indices_zyx, spatial_shape_DWH, batch_size)
    spconv_dense = spconv_tensor.dense()

    sphconv_tensor = sphconv.SparseConvTensor(
        features, spatial_shape_DWH, batch_size, indices=indices_zyx)

    sphconv_dense = sphconv_tensor.dense().cuda()
    # print("sphconv_tensor.z_ptr =", sphconv_tensor.z_ptr)
    # print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.all(torch.isclose(sphconv_dense, spconv_dense))


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_todense(self):
        spatial_shape_HWD = [2, 2, 2]
        vvfe = VoxelizationVFE(resolution_HWD=spatial_shape_HWD)
        features, coords = vvfe.generate(POINTS_FILE, torch.device('cuda:0'))
        assert features.shape[1] == 4
        assert coords.shape[1] == 3  # zyx
        example = merge_batch_torch(
            [{'voxels': features, 'coordinates': coords}])
        indices = example['coordinates']
        features = example['voxels']

        spconv_tensor = spconv.SparseConvTensor(
            features, indices, vvfe.resolution_HWD[::-1], 1)
        spconv_dense = spconv_tensor.dense()  # torch
        # N C D W H
        assert spconv_dense.shape[0] == 1
        assert spconv_dense.shape[1] == 4
        assert spconv_dense.shape[2] == vvfe.resolution_HWD[2]  # D
        assert spconv_dense.shape[3] == vvfe.resolution_HWD[1]  # W
        assert spconv_dense.shape[4] == vvfe.resolution_HWD[0]  # H

        # print("features = ", features)
        # print("feautures.dtype = ", features.dtype)
        # print("indices = ", indices)

        sphconv_tensor = sphconv.SparseConvTensor(
            features, vvfe.resolution_HWD[::-1], 1, indices=indices)
        # print("sphconv_tensor.feature = ", sphconv_tensor.feature)
        # print("sphconv_tensor.z_ptr = ", sphconv_tensor.z_ptr)

        sphconv_dense = sphconv_tensor.dense()
        # print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
        # print("sphconv = ", sphconv_dense)
        # print("spconv = ", spconv_dense)
        assert spconv_dense.shape == sphconv_dense.shape
        assert torch.all(torch.isclose(spconv_dense, sphconv_dense))

    def test_batch(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=torch.int).cuda()
        feature, indices_zyx = batch_artifical_inputs(
            indices_zyx, channel=4, batch_size=3)
        assert_dense_eq(
            feature, indices_zyx, batch_size=3, spatial_shape_DWH=[2, 2, 3])

    def test_init(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
        ], dtype=torch.int).cuda()

        feature, indices_zyx = batch_artifical_inputs(
            indices_zyx, channel=4, batch_size=2)
        assert_dense_eq(
            feature, indices_zyx, batch_size=2, spatial_shape_DWH=[1, 1, 2])

    def test_batch2(self):
        batch_size = 4
        spatial_shape_DWH = [64, 511, 512]
        vvfe = VoxelizationVFE(resolution_HWD=spatial_shape_DWH[::-1])

        example_list = []
        for i in range(4):
            voxels, coords = vvfe.generate(
                '{:06d}.bin'.format(i), torch.device('cuda:0'))
            # print("coords shape = ", coords.shape)
            one_example = {'voxels': voxels, 'coordinates': coords}
            example_list.append(one_example)

        example = merge_batch_torch(example_list)

        assert_dense_eq(
            example['voxels'], example['coordinates'], batch_size=batch_size,
            spatial_shape_DWH=spatial_shape_DWH)
