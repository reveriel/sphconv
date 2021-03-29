
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator


def assert_conv_eq(
    features: torch.Tensor,
    indices_zyx: torch.Tensor,
    batch_size: int,
    spatial_shape_HWD: List[int],
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
        features, spatial_shape_HWD[::-1], batch_size, indices=indices_zyx)

    out_spatial_shape_HWD = spatial_shape_HWD
    if not subm:
        out_spatial_shape_HWD = out_spatial(
            spatial_shape_HWD, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape_HWD)

    # TODO: remove the grid
    if not subm:
        tensor.grid = torch.empty(
            (batch_size, *out_spatial_shape_HWD), dtype=indices_zyx.dtype, device=indices_zyx.device)

    get_rule_func = get_rules_subm if subm else get_rules
    oz_idx, oz_ptr, rules, rule_size = get_rule_func(
        tensor.z_idx, tensor.z_ptr, tensor.grid,
        batch_size, spatial_shape_HWD, out_spatial_shape_HWD,
        kernel_size, stride, padding, dilation)

    # print("rules = ", rules)
    # print("rule_size = ", rule_size)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices_zyx, batch_size, spatial_shape_HWD[::-1], kernel_size[::-1],
        stride[::-1], padding[::-1], dilation[::-1], out_padding=0, subm=subm,
        transpose=False, grid=None, use_hash=False)

    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    # convolution
    out_features = spconv.ops.indice_conv(
        features, weight.permute(2, 1, 0, 4, 3).contiguous(), indice_pairs,
        indice_pair_num, outids.shape[0], subm=subm)

    spconv_dense = spconv.SparseConvTensor(
        out_features, outids, out_spatial_shape_HWD[::-1], batch_size).dense()

    sph_out_features = rule_conv(
        tensor.feature, weight.reshape(-1, outChannel, inChannel),
        rules, rule_size, batch_size, spatial_shape_HWD, out_spatial_shape_HWD, oz_idx.shape[0])

    assert sph_out_features.dim() == 2
    assert sph_out_features.shape[0] == oz_idx.shape[0]

    sphconv_dense = sphconv.SparseConvTensor(
        sph_out_features, out_spatial_shape_HWD[::-1], batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

    # print("sphconv = ", sphconv_dense[:, 0, :, :, :])
    # print("spconv = ", spconv_dense[:, 0, :, :, :])
    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)
    print("sphconv shape ", sphconv_dense.shape)
    print("spconv shape ", spconv_dense.shape)

    print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.1))

def assert_correct_cmp_with_spconv(
    indices_zyx: torch.Tensor,
    batch_size: int,
    inChannel: int, outChannel: int,
    spatial_shape_HWD: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]

    features = torch.randn(
        (indices_zyx.shape[0], inChannel), dtype=torch.float, device=indices_zyx.device)

    weight = torch.randn((*kernel_size, outChannel, inChannel),
                         dtype=features.dtype, device=indices_zyx.device)

    one_example = {'voxel': features, 'coordinates': indices_zyx}
    example = merge_batch_torch([one_example] * batch_size)

    assert_conv_eq(
        example['voxel'], example['coordinates'],
        batch_size=batch_size, spatial_shape_HWD=spatial_shape_HWD,
        inChannel=inChannel, outChannel=outChannel,
        weight=weight,
        kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        subm=subm)


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_conv(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, inChannel=4, outChannel=4, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, inChannel=4, outChannel=4, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, inChannel=4, outChannel=5, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, inChannel=4, outChannel=8, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, inChannel=4, outChannel=128, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, inChannel=128, outChannel=128, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, inChannel=128, outChannel=128, spatial_shape_HWD=[3, 5, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_subm_conv2(self):
        indices = torch.tensor([
            [1, 0, 7],
            [1, 1, 6],
            [2, 1, 6],
            [3, 3, 6],
            [3, 3, 3],
            [0, 0, 1],
            [1, 3, 3],
            [1, 3, 4],
            [1, 3, 5],
            [2, 3, 5],
            [3, 3, 5],
            [4, 3, 5],
            [7, 3, 5],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=4, spatial_shape_HWD=[8, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=5, spatial_shape_HWD=[9, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=128, spatial_shape_HWD=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=128, outChannel=128, spatial_shape_HWD=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=128, outChannel=128, spatial_shape_HWD=[10, 9, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_std_conv(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=4, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)


    def test_std_conv2(self):
        indices = torch.tensor([
            [1, 0, 7],
            [1, 1, 6],
            [2, 1, 6],
            [3, 3, 6],
            [3, 3, 3],
            [0, 0, 0],
            [1, 3, 3],
            [1, 3, 4],
            [1, 3, 5],
            [2, 3, 5],
            [3, 3, 5],
            [4, 3, 5],
            [7, 3, 5],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=4, outChannel=4, spatial_shape_HWD=[8, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, inChannel=4, outChannel=8, spatial_shape_HWD=[9, 9, 11],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, inChannel=128, outChannel=128, spatial_shape_HWD=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)


