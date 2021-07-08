
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch
from common import batch_real_test_inputs, batch_artifical_inputs


def assert_correct_cmp_with_spconv(
    indices_zyx: torch.Tensor, # [NNZ, 3]
    batch_size: int,
    in_channels: int, out_channels: int,
    spatial_shape_DWH: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    if subm:
        assert dilation == stride == dilation == [1, 1, 1]
    # TODO
    # batch_size = 1

    feature, indices = batch_artifical_inputs(
        indices_zyx, channel=in_channels, batch_size=batch_size)

    sphconv_tensor = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indices)

    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_DWH, batch_size)

    Sphconv_Conv3d = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
    sph_conv = Sphconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, in_channels, out_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(weight.clone())

    with torch.no_grad():
        spconv_dense = sp_conv(spconv_tensor).dense()
        sphconv_dense = sph_conv(sphconv_tensor).dense()

    # print("sphconv = ", sphconv_dense)
    # print("spconv = ", spconv_dense)
    print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.02).all()



def assert_correct_cmp_with_spconv_real(
    batch_size: int,
    in_channels: int, out_channels: int,
    spatial_shape_DWH: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    # batch_size = 1 # TODO
    if subm:
        assert dilation == stride == dilation == [1, 1, 1]

    feature, indices = batch_real_test_inputs(
        channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)
    # print("indices = ", indices)

    sphconv_tensor = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indices)

    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_DWH, batch_size)

    Sphconv_Conv3d = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
    sph_conv = Sphconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, in_channels, out_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(weight.clone())

    with torch.no_grad():
        spconv_dense = sp_conv(spconv_tensor).dense()
        sphconv_dense = sph_conv(sphconv_tensor).dense()

    # print("sphconv = ", sphconv_dense)
    # print("spconv = ", spconv_dense)
    print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
    assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.02).all()


class TestClass:
    def test_conv3D_1(self):

        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=4, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_conv3D_2(self):

        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=4, out_channels=16, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 8],
            kernel_size=[2, 2, 2], stride=[2, 1, 1], padding=[0, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[4, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[20, 20, 20],
            kernel_size=[3, 3, 3], stride=[1, 1, 2], padding=[1, 1, 0], subm=False)


    def test_conv3D_3(self):

        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=4, out_channels=16, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=16, spatial_shape_DWH=[5, 5, 5],
            kernel_size=[3, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[4, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[6, 6, 20],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=16, spatial_shape_DWH=[77, 40, 20],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=32, out_channels=64, spatial_shape_DWH=[6, 6, 7],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=32, out_channels=64, spatial_shape_DWH=[6, 6, 7],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 0, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 0], subm=False)

    def test_conv3D_4(self):
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=4, out_channels=16, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=16, spatial_shape_DWH=[5, 5, 5],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[4, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[6, 6, 20],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=16, spatial_shape_DWH=[77, 40, 20],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=32, out_channels=64, spatial_shape_DWH=[6, 6, 7],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=32, out_channels=64, spatial_shape_DWH=[6, 6, 7],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_conv3D_5(self):
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 1, 1], stride=[3, 1, 1], padding=[0, 1, 0], subm=False)

        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[2, 1, 1], padding=[0, 0, 0], subm=False)

        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 1, 1], stride=[3, 1, 1], padding=[1, 1, 1], subm=False)

        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=64, out_channels=64, spatial_shape_DWH=[12, 12, 12],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], subm=False)

    def test_conv3D_6(self):
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=4, out_channels=16, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=1, in_channels=16, out_channels=16, spatial_shape_DWH=[5, 5, 5],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 0, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[4, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[6, 6, 20],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], subm=True)

    def test_rule_cache(self):
        assert True
