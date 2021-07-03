
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator


def assert_conv_eq(
    feature: torch.Tensor,
    indices_bzyx: torch.Tensor,
    batch_size: int,
    spatial_shape_DWH: List[int],
    in_channels: int, out_channels: int,
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
        feature, spatial_shape_DWH, batch_size, indices=indices_bzyx)

    out_spatial_shape_DWH = spatial_shape_DWH
    if not subm:
        out_spatial_shape_DWH = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape_DWH)

    get_rule_func = get_rules_subm if subm else get_rules
    oz_idx, oz_ptr, rules, rule_size = get_rule_func(
        tensor.z_idx, tensor.z_ptr,
        batch_size, spatial_shape_DWH, out_spatial_shape_DWH,
        kernel_size, stride, padding, dilation, [2, 2])

    # print("rules = ", rules)
    # print("rule_size = ", rule_size)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
        stride, padding, dilation, out_padding=0, subm=subm,
        transpose=False, use_hash=False)

    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    # convolution
    out_feature = spconv.ops.indice_conv(
        feature, weight, indice_pairs,
        indice_pair_num, outids.shape[0], subm=subm)

    spconv_dense = spconv.SparseConvTensor(
        out_feature, outids, out_spatial_shape_DWH, batch_size).dense()

    sph_out_feature = rule_conv(
        tensor.feature, weight.reshape(-1, in_channels, out_channels),
        rules, rule_size, oz_idx.shape[0])

    assert sph_out_feature.dim() == 2
    assert sph_out_feature.shape[0] == oz_idx.shape[0]

    sphconv_dense = sphconv.SparseConvTensor(
        sph_out_feature, out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

    # print("sphconv = ", sphconv_dense[:, 0, :, :, :])
    # print("spconv = ", spconv_dense[:, 0, :, :, :])
    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)
    print("sphconv shape ", sphconv_dense.shape)
    print("spconv shape ", spconv_dense.shape)

    print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.01))


def assert_correct_cmp_with_spconv(
    indices_zyx: torch.Tensor,
    batch_size: int,
    in_channels: int, out_channels: int,
    spatial_shape_DWH: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    batch_size = 1 #TODO
    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]

    feature = torch.randn(
        (indices_zyx.shape[0], in_channels), dtype=torch.float, device=indices_zyx.device)

    weight = torch.randn((*kernel_size, in_channels, out_channels),
                         dtype=feature.dtype, device=indices_zyx.device)

    one_example = {'voxel': feature, 'coordinates': indices_zyx}
    example = merge_batch_torch([one_example] * batch_size)

    assert_conv_eq(
        example['voxel'], example['coordinates'],
        batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH,
        in_channels=in_channels, out_channels=out_channels,
        weight=weight,
        kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        subm=subm)


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_conv1(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        # assert_correct_cmp_with_spconv(
        #     indices_zyx, batch_size=1, in_channels=4, out_channels=4, spatial_shape_DWH=[2, 2, 2],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices_zyx, batch_size=8, in_channels=4, out_channels=4, spatial_shape_DWH=[2, 2, 2],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices_zyx, batch_size=1, in_channels=4, out_channels=5, spatial_shape_DWH=[2, 2, 2],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices_zyx, batch_size=1, in_channels=4, out_channels=8, spatial_shape_DWH=[3, 3, 3],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, in_channels=16, out_channels=16, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, in_channels=32, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[3, 5, 8],
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

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, in_channels=4, out_channels=4, spatial_shape_DWH=[8, 8, 8],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, in_channels=4, out_channels=4, spatial_shape_DWH=[8, 9, 9],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, in_channels=4, out_channels=5, spatial_shape_DWH=[9, 9, 9],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, in_channels=4, out_channels=8, spatial_shape_DWH=[8, 8, 8],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=32, spatial_shape_DWH=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=32, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, in_channels=128, out_channels=128, spatial_shape_DWH=[10, 9, 8],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_subm_conv3(self):
        indices = torch.tensor([
        [0, 2, 3],
        [0, 4, 1],
        [0, 0, 1],
        [0, 1, 3],
        [0, 2, 2],
        [0, 2, 0],
        [0, 2, 1],
        [0, 4, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 2],
        [0, 1, 4],
        [0, 1, 1],
        [1, 2, 3],
        [1, 2, 2],
        [1, 2, 0],
        [1, 3, 0],
        [1, 4, 2],
        [1, 1, 1],
        [2, 4, 1],
        [2, 2, 0],
        [2, 2, 1],
        [4, 0, 0],
        [4, 1, 0]
        ], dtype=torch.int).cuda()


        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=32, spatial_shape_DWH=[5, 5, 5],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=32, spatial_shape_DWH=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=32, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, in_channels=128, out_channels=128, spatial_shape_DWH=[10, 9, 8],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
    def test_std_conv1(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=8, out_channels=8, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=8, out_channels=8, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=8, out_channels=8, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=8, out_channels=8, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=16, spatial_shape_DWH=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1], subm=False)

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

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, in_channels=4, out_channels=4, spatial_shape_DWH=[8, 8, 8],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, in_channels=4, out_channels=4, spatial_shape_DWH=[8, 8, 8],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=8, in_channels=4, out_channels=4, spatial_shape_DWH=[8, 9, 9],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, in_channels=4, out_channels=8, spatial_shape_DWH=[9, 9, 11],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1],
        #     subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=16, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=16, out_channels=32, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=32, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=64, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=32, out_channels=64, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 0], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 13, 14],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 0], subm=False)



