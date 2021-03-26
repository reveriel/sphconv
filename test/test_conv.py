
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator


def assert_conv_eq(
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: List[int],
    inChannel: int, outChannel: int,
    weight: torch.Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    """
    Note: weight for spconv needs to be permuted
    """
    tensor = sphconv.SparseConvTensor(
        features, spatial_shape, batch_size, indices=indices)

    out_spatial_shape = spatial_shape
    if not subm:
        out_spatial_shape = out_spatial(
            spatial_shape, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape)

    get_rule_func = get_rules_subm if subm else get_rules
    oz_idx, oz_ptr, rules, rule_size = get_rule_func(
        tensor.z_idx, tensor.z_ptr, tensor.grid,
        batch_size, spatial_shape, out_spatial_shape,
        kernel_size, stride, padding, dilation)

    # print("rules = ", rules)
    # print("rule_size = ", rule_size)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
        out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    # convolution
    out_features = spconv.ops.indice_conv(
        features, weight.permute(2, 1, 0, 4, 3).contiguous(), indice_pairs, indice_pair_num, outids.shape[0])

    spconv_dense = spconv.SparseConvTensor(
        out_features, indices, out_spatial_shape, batch_size).dense()

    sph_out_features = rule_conv(
        tensor.features, weight.reshape(-1, outChannel, inChannel),
        rules, rule_size, batch_size, spatial_shape, out_spatial_shape)

    sphconv_dense = sphconv.SparseConvTensor(
        sph_out_features, out_spatial_shape, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

    # print("sphconv = ", sphconv_dense[:, 0, :, :, :])
    # print("spconv = ", spconv_dense[:, 0, :, :, :])
    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)
    print("sphconv shape ", sphconv_dense.shape)
    print("spconv shape ", spconv_dense.shape)

    print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.001))


def assert_correct_cmp_with_spconv(
    indices: torch.Tensor,
    batch_size: int,
    inChannel: int, outChannel: int,
    spatial_shape: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]

    features = torch.randn(
        (indices.shape[0], inChannel), dtype=torch.float, device=indices.device)

    weight = torch.randn((*kernel_size, outChannel, inChannel),
                         dtype=features.dtype, device=indices.device)

    one_example = {'voxel': features, 'coordinates': indices}
    example = merge_batch_torch([one_example] * batch_size)

    assert_conv_eq(
        example['voxel'], example['coordinates'],
        batch_size=batch_size, spatial_shape=spatial_shape,
        inChannel=inChannel, outChannel=outChannel,
        weight=weight,
        kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        subm=subm)


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_conv(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=4, spatial_shape=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=5, spatial_shape=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=128, spatial_shape=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=128, outChannel=128, spatial_shape=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=128, outChannel=128, spatial_shape=[3, 5, 8],
            kernel_size=[3, 3, 3], stride=[1, 2, 1], padding=[1, 1, 1], subm=True)

    def test_std_conv(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape=[2, 2, 2],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, inChannel=4, outChannel=4, spatial_shape=[2, 2, 2],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape=[2, 2, 2],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape=[3, 3, 3],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)



