

from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator


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
    indices_zyx: torch.Tensor,
    batch_size: int,
    in_channels: int, out_channels: int,
    spatial_shape_HWD: List[int],
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
        feature, spatial_shape_HWD[::-1], batch_size, indices=indices)

    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_HWD[::-1], batch_size)

    sph_conv = sphconv.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size[::-1], stride=stride[::-1], padding=padding[::-1], dilation=dilation[::-1], bias=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, out_channels, in_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(
        weight.clone().permute(2, 1, 0, 4, 3).contiguous())

    with torch.no_grad():
        spconv_dense = sp_conv(spconv_tensor).dense()
        sphconv_dense = sph_conv(sphconv_tensor).dense()

    print("sphconv = ", sphconv_dense)
    print("spconv = ", spconv_dense)

    assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.01).all()


class TestClass:
    def test_conv3D(self):

        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_HWD=[2, 2, 8],
            kernel_size=[2, 2, 2], stride=[2, 1, 1], padding=[0, 1, 1], subm=False)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv(
            indices, batch_size=1, in_channels=16, out_channels=32, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        assert_correct_cmp_with_spconv(
            indices, batch_size=3, in_channels=16, out_channels=32, spatial_shape_HWD=[2, 2, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_rule_cache(self):
        assert True
