

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


class TestClass:
    def test_conv3D(self):

        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        # assert_correct_cmp_with_spconv(
        #     indices, batch_size=1, inChannel=4, outChannel=4, spatial_shape_HWD=[2, 2, 2],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        batch_size = 1

        in_channel = 16
        out_channel = 32
        # weight = ?
        kernel_size = [3,3,3]

        feature = torch.randn(
            (indices.shape[0], in_channel), dtype=torch.float, device=indices.device)

        feature, indices = batch_artifical_inputs(
            indices, channel=in_channel, batch_size=batch_size)

        spatial_shape_HWD = [4, 4, 4]

        sphconv_tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_HWD[::-1], batch_size, indices=indices
        )

        spconv_tensor = spconv.SparseConvTensor(
            feature, indices, spatial_shape_HWD[::-1], batch_size)

        sph_conv = sphconv.Conv3d(
            in_channel, out_channel, kernel_size, bias=False).cuda()

        sp_conv = spconv.SparseConv3d(
            in_channel, out_channel, kernel_size, bias=False).cuda()

        # same weight
        weight = torch.randn((*kernel_size, in_channel, out_channel),
                             dtype=torch.float, device=indices.device)

        sph_conv.weight = torch.nn.Parameter(weight.clone())
        sp_conv.weight = torch.nn.Parameter(
            weight.clone().permute(2, 1, 0, 4, 3).contiguous())

        with torch.no_grad():
            spconv_dense = sp_conv(spconv_tensor).dense()
            sphconv_dense = sph_conv(sphconv_tensor).dense()

        assert torch.isclose(spconv_dense, sphconv_dense, rtol=0.01).all()
