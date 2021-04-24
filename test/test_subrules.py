
from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial


def assert_conv_eq(
    features: torch.Tensor,
    indices_bzyx: torch.Tensor,
    batch_size: int,
    spatial_shape_DWH: List[int],
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
        features, spatial_shape_DWH, batch_size, indices=indices_bzyx)

    out_spatial_shape_DWH = spatial_shape_DWH
    if not subm:
        out_spatial_shape_DWH = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape_DWH)

    get_rule_func = get_rules_subm if subm else get_rules
    oz_idx, oz_ptr, rules, rule_size = get_rule_func(
        tensor.z_idx, tensor.z_ptr,
        batch_size, spatial_shape_DWH, out_spatial_shape_DWH,
        kernel_size, stride, padding, dilation)

    # print("rules = ", rules)
    # print("rule_size = ", rule_size)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
        stride, padding, dilation, out_padding=0, subm=subm,
        transpose=False, use_hash=False)

    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    # convolution
    out_features = spconv.ops.indice_conv(
        features, weight.permute(0, 1, 2, 4, 3).contiguous(), indice_pairs,
        indice_pair_num, outids.shape[0], subm=subm)

    spconv_dense = spconv.SparseConvTensor(
        out_features, outids, out_spatial_shape_DWH, batch_size).dense()

    sph_out_features = rule_conv(
        tensor.feature, weight.reshape(-1, outChannel, inChannel),
        rules, rule_size, batch_size, spatial_shape_DWH, out_spatial_shape_DWH, oz_idx.shape[0])

    assert sph_out_features.dim() == 2
    assert sph_out_features.shape[0] == oz_idx.shape[0]

    sphconv_dense = sphconv.SparseConvTensor(
        sph_out_features, out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

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
    spatial_shape_DWH: List[int],
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
        batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH,
        inChannel=inChannel, outChannel=outChannel,
        weight=weight,
        kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        subm=subm)



class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor

    def test_subm_rules(self):

        # try a conv, divide the conv into two tiles
        # 1. DWH, divide on which dimension ?
        #    z ,y, x -> theta, phi,  r
        #    D W H
        #,  we  divide on  (z, y), or  (theta, phi)
        #,          or  (D, W)-mode

        spatial_shape_DWH = [4, 4, 2]
        # we divide it as four [2, 2] tiles on H,W-mode

        indices_zyx = torch.tensor([
            [0, 0, 0],
            [0, 1, 1],
            [0, 2, 0],
            [0, 3, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 2, 0],
            [1, 3, 1],
            [2, 0, 1],
            [2, 1, 0],
            [2, 2, 0],
            [2, 3, 1],
            [3, 0, 1],
            [3, 1, 0],
            [3, 2, 0],
            [3, 3, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, inChannel=4, outChannel=4, spatial_shape_DWH=[4, 4, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

