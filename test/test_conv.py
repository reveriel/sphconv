
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules_subm, rule_conv
from sphconv.utils import voxel_generator


def assert_subm_conv_eq(
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: List[int],
    inChannel: int, outChannel: int,
    weight: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
):
    tensor = sphconv.SparseConvTensor(
        features, spatial_shape, batch_size, indices=indices)

    oz_idx, oz_ptr, rules, rule_size = get_rules_subm(
        tensor.z_idx,
        tensor.z_ptr,
        tensor.grid,
        batch_size,
        spatial_shape,
        spatial_shape,
        [kernel_size, kernel_size, kernel_size],
        [stride, stride, stride],
        [padding, padding, padding],
        [dilation, dilation, dilation]
    )

    print("rules = ", rules)
    print("rule_size = ", rule_size)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
        out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    # convolution
    out_features = spconv.ops.indice_conv(
        features, weight.permute(2,1,0,4,3).contiguous(), indice_pairs, indice_pair_num, outids.shape[0])

    spconv_dense = spconv.SparseConvTensor(
        out_features, indices, spatial_shape, batch_size).dense()

    sph_out_features = rule_conv(
        tensor.features, weight.reshape(-1, outChannel, inChannel),
        rules, rule_size, batch_size, spatial_shape, spatial_shape)

    sphconv_dense = sphconv.SparseConvTensor(
        sph_out_features, spatial_shape, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

    print("sphconv = " , sphconv_dense[:,0,:,:,:])
    print("spconv = " , spconv_dense[:,0,:,:,:])

    assert torch.all(torch.isclose(spconv_dense, sphconv_dense))


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_rule_conv(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            # [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        spatial_shape = [2, 2, 2]  # D, W, H
        inChannel = 4
        batch_size = 1
        features = torch.arange(
            (indices.shape[0] * inChannel),
            dtype=torch.float, device=indices.device).reshape(indices.shape[0], inChannel)

        assert_subm_conv_eq(features, indices, batch_size=1, spatial_shape=spatial_shape,
                            inChannel=4, outChannel=4,
                            weight=torch.arange(
                                (2 * 2 * 2),
                                dtype=features.dtype).cuda().reshape(2, 2, 2, 1, 1).expand(2, 2, 2, 4, inChannel),
                            kernel_size=2,
                            stride=1, padding=1, dilation=1)

    def test_batch_size(self):

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

        assert_subm_conv_eq(example['voxel'], example['coordinates'], batch_size=batch_size, spatial_shape=spatial_shape,
                            inChannel=4, outChannel=4,
                            weight=torch.randn(
                                (2, 2, 2, 4, 4), dtype=features.dtype).cuda(),
                            kernel_size=2,
                            stride=1, padding=1, dilation=1)

        # # batch size, out channel
        # assert_subm_conv_eq(features, indices, batch_size=8, spatial_shape=spatial_shape,
        #     inChannel=4, outChannel=128,
        #     weight=torch.ones((2,2,2,128,4),dtype=features.dtype).cuda(),
        #     kernel_size=2,
        #     stride=1, padding=1, dilation=1)
