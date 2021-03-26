# from sphconv.utils import *

# point to raw voxelFeature
# raw voxelFeature processed with VFE
# voxel feature convert to sphconv.Feature and spconv.SparseTensor

# to dense to check eqaulity

# run convolution on both.

import spconv
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch
from typing import List

POINTS_FILE = "000003.bin"

def assert_dense_eq(features: torch.Tensor, indices: torch.Tensor,
                    batch_size, spatial_shape: List[int]):

    assert features.shape[1] == 4
    assert features.dim() == 2
    assert indices.shape[1] == 4
    assert indices.dim() == 2
    spconv_tensor = spconv.SparseConvTensor(
        features, indices, spatial_shape, batch_size)
    spconv_dense = spconv_tensor.dense()

    sphconv_tensor = sphconv.SparseConvTensor(
        features, spatial_shape, batch_size, indices=indices)
    sphconv_dense = sphconv_tensor.dense().cuda()

    assert torch.all(torch.eq(sphconv_dense, spconv_dense))


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_todense(self):
        vvfe = VoxelizationVFE(resolution=[512, 511, 64])
        voxels, coords = vvfe.generate(POINTS_FILE)
        assert voxels.shape[1] == 4
        assert coords.shape[1] == 3
        example = merge_batch_torch([{'voxels':voxels, 'coordinates':coords}])

        spconv_tensor = spconv.SparseConvTensor(
            voxels, example['coordinates'], vvfe.resolution[::-1], 1)
        spconv_dense = spconv_tensor.dense()  # torch
        # N C D W H
        assert spconv_dense.shape[0] == 1
        assert spconv_dense.shape[1] == 4
        assert spconv_dense.shape[2] == vvfe.resolution[2]
        assert spconv_dense.shape[3] == vvfe.resolution[1]
        assert spconv_dense.shape[4] == vvfe.resolution[0]

        sphconv_tensor = sphconv.SparseConvTensor(
            voxels, vvfe.resolution[::-1], 1, indices=example['coordinates'])
        sphconv_dense = sphconv_tensor.dense()

        assert spconv_dense.shape == sphconv_dense.shape
        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

    def test_batch(self):
        batch_size = 8
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=torch.int).cuda()
        inChannel = 4
        features = torch.randn(
            (indices.shape[0], inChannel), dtype=torch.float, device=indices.device)

        one_example = {'voxel': features, 'coordinates': indices}
        example = merge_batch_torch([one_example] * batch_size)

        spatial_shape = [2, 2, 2]  # D, W, H

        # print("example[coordinates] = ", example['coordinates'])
        assert_dense_eq(
            example['voxel'], example['coordinates'], batch_size, spatial_shape)

    def test_batch2(self):
        batch_size = 4
        vvfe = VoxelizationVFE(resolution=[512, 511, 64])

        example_list = []
        for i in range(4):
            voxels, coords = vvfe.generate('{:06d}.bin'.format(i), torch.device('cuda:0'))
            # print("coords shape = ", coords.shape)
            one_example = {'voxels': voxels, 'coordinates': coords}
            example_list.append(one_example)

        example = merge_batch_torch(example_list)
        spatial_shape = [64, 511, 512]

        assert_dense_eq(example['voxels'], example['coordinates'], batch_size, spatial_shape)

