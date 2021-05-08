
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch

def batch_real_test_inputs(
    channel: int,
    batch_size: int,
    spatial_shape_DWH: List[int]
):
    TEST_FILE_MAX = 4
    vvfe = VoxelizationVFE(resolution_HWD=spatial_shape_DWH[::-1])

    example_list = []
    for i in range(batch_size):
        voxels, coords = vvfe.generate(
            '{:06d}.bin'.format(i %  TEST_FILE_MAX),  torch.device('cuda:0'))
        example_list.append({'voxels': voxels, 'coordinates': coords})
    example = merge_batch_torch(example_list)

    feature, indices = example['voxels'], example['coordinates']
    # feature, [NNZ, 4]
    # original channel is 4, we extend it if needed
    assert channel >= 4;
    if channel > 4:
        feature.resize_((feature.shape[0], channel))
    return feature, indices


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

    feature, indices = batch_artifical_inputs(
        indices_zyx, channel=in_channels, batch_size=batch_size)

    sphconv_tensor = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indices)

    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_DWH, batch_size)

    sph_conv = sphconv.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, out_channels, in_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(
        weight.clone().permute(0, 1, 2, 4, 3).contiguous())

    with torch.no_grad():
        spconv_dense = sp_conv(spconv_tensor).dense()
        sphconv_dense = sph_conv(sphconv_tensor).dense()

    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)

    assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.01).all()



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
    batch_size = 1
    if subm:
        assert dilation == stride == dilation == [1, 1, 1]

    feature, indices = batch_real_test_inputs(
        channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)

    torch.set_printoptions(edgeitems=8000)
    print ("indices = ", indices)

    sphconv_tensor = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indices)

    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_DWH, batch_size)

    sph_conv = sphconv.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, out_channels, in_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(
        weight.clone().permute(0, 1, 2, 4, 3).contiguous())

    with torch.no_grad():
        spconv_dense = sp_conv(spconv_tensor).dense()
        sphconv_dense = sph_conv(sphconv_tensor).dense()

    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)

    assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.01).all()




class TestClass:
    def test_conv3D_1(self):

        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

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


        # assert_correct_cmp_with_spconv_real(
        #     batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 2],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        # assert_correct_cmp_with_spconv_real(
        #     batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[3, 3, 3],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        # assert_correct_cmp_with_spconv_real(
        #     batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[2, 2, 8],
        #     kernel_size=[2, 2, 2], stride=[2, 1, 1], padding=[0, 1, 1], subm=False)
        # assert_correct_cmp_with_spconv_real(
        #     batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[4, 8, 8],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 2], padding=[1, 1, 0], subm=False)
        assert_correct_cmp_with_spconv_real(
            batch_size=3, in_channels=16, out_channels=32, spatial_shape_DWH=[6, 6, 20],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 0], subm=False)



    def test_rule_cache(self):
        assert True
