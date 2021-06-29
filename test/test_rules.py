
from typing import List, Optional

import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial

from common import dup_with_batch_idx, batch_real_test_inputs


def assert_correct_cmp_with_torch(
        indices_zyx: torch.Tensor,
        batch_size: int,
        spatial_shape_DWH: List[int], #[DWH]
        kernel_size: List[int], #[DWH]
        stride: List[int],
        padding: List[int],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):
    assert subm == False

    indices_bzyx = dup_with_batch_idx(indices_zyx, batch_size)

    voxel_features = torch.ones((indices_bzyx.shape[0], 1), device=indices_bzyx.device)
    test_func = get_rules_subm if subm else get_rules
    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)
    out_spatial_shape = spatial_shape_DWH
    if not subm:
        out_spatial_shape = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    print("out shape(DWH) = ", out_spatial_shape)

    oz_idx, oz_ptr, rules, rule_size = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        batch_size,
        spatial_shape_DWH, # [DWH]
        out_spatial_shape,
        kernel_size, stride, padding, dilation, [2, 2])

    # assert torch.sum(indice_pair_num) == torch.sum(rule_size)

    sphconv_dense = sphconv.SparseConvTensor(
        torch.ones((oz_idx.shape[0], 1), device=indices_bzyx.device),
        out_spatial_shape, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense()

    weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32, device=indices_bzyx.device)

    torch_dense = torch.nn.functional.conv3d(tensor.dense(), weight, None, stride, padding, dilation)
    torch_dense[torch_dense != 0] = 1

    print("sphconv_dense = ", sphconv_dense)
    print("torch_dense = ", torch_dense)

    assert (torch.isclose(sphconv_dense, torch_dense)).all()


def assert_correct_cmp_with_spconv(
        indices_zyx: Optional[torch.Tensor] = None, # [NNZ, 3/4]
        batch_size: int = 1,
        spatial_shape_DWH: List[int] = [8, 8, 8],
        kernel_size: List[int] = [3, 3, 3],
        stride: List[int] = [2, 2, 2],
        padding: List[int] = [1, 1, 1],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):

    if indices_zyx is None:
        voxel_features, indices_bzyx  = batch_real_test_inputs(
            spatial_shape_DWH=spatial_shape_DWH)
    else:
        if (indices_zyx.size(1) == 3):
            indices_bzyx = dup_with_batch_idx(indices_zyx, batch_size)
        else:
            indices_bzyx = indices_zyx
        voxel_features = torch.ones((indices_bzyx.shape[0], 1), device=indices_bzyx.device)

    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]
        # padding is ignored for subm conv
        padding = [k//2 for k in kernel_size]

    test_func = get_rules_subm if subm else get_rules

    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)

    out_spatial_shape_DWH = spatial_shape_DWH
    if not subm:
        out_spatial_shape_DWH = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    # print("out shape = ", out_spatial_shape_DWH)
    torch.save(tensor.z_idx, "z_idx.pt")
    torch.save(tensor.z_ptr, "z_ptr.pt")

    # print("z_idx = ", tensor.z_idx)
    # print("z_ptr = ", tensor.z_ptr)

    oz_idx, oz_ptr, rules, rule_size = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        batch_size,
        spatial_shape_DWH,
        out_spatial_shape_DWH,
        kernel_size, stride, padding, dilation, [2, 2])

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
        stride, padding, dilation,
        out_padding=0, subm=subm, transpose=False, use_hash=False)

    # print("outids = ", outids)
    # print("indice_pairs = ", indice_pairs)
    # print("indice_pair_num = ", indice_pair_num)

    # print("oz_idx = ", oz_idx)
    # print("oz_ptr = ", oz_ptr)
    # print("rules = ", rules[:, :, :, :10])
    # print("rule size = ", rule_size)
    print("rules.sum = ", rules.sum())

    # check_rules(rules, indice_pairs)

    assert torch.sum(indice_pair_num) == torch.sum(rule_size)
    if (rule_size.shape[0] == 1) :
        assert (indice_pair_num.view(-1).sort()[0] == rule_size.view(-1).sort()[0]).all()
    else : # tiled version
        assert (indice_pair_num.view(-1).sort()[0] == rule_size.sum(dim=0).view(-1).sort()[0]).all()
        # NTile = rules.shape[0]
        # loadingRule = rules[:,:,0,:].reshape([NTile,-1])
    assert (outids[:,3].sort()[0] == oz_idx.sort()[0]).all()

    if not subm:
        # check oz_ptr
        spconv_dense = spconv.SparseConvTensor(
            torch.ones((outids.shape[0], 1), device=indices_bzyx.device), outids, out_spatial_shape_DWH, batch_size).dense()
        sphconv_dense = sphconv.SparseConvTensor(
            torch.ones((oz_idx.shape[0], 1), device=indices_bzyx.device), out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense()

        # print("sphconv = ", sphconv_dense)
        # print("spconv = ", spconv_dense)
        weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32).cuda()
        torch_dense = torch.nn.functional.conv3d(tensor.dense()[:,0:1,:,:,:], weight, None, stride, padding, dilation)
        torch_dense[torch_dense != 0] = 1

        # print("torch_dense = ", torch_dense)
        assert (torch.isclose(spconv_dense, sphconv_dense).all())


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_rules_1(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        # in subm, padding is ignored, we computed padding by k//2 in conv.py

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[1, 1, 1], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[1, 2, 1], stride=[1, 1, 1],  subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 1], stride=[1, 1, 1],  subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[4, 4, 2],
            kernel_size=[2, 2, 1], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[4, 4, 2],
            kernel_size=[1, 1, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 3, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_DWH=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

    def test_subm_rules_3(self):
        indices = torch.tensor([
            [0, 0, 0],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_DWH=[2, 1, 1],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

    def test_subm_rules_2(self):
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
            indices, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[8, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[9, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[10, 9, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], subm=True)

    def test_std_rules_1(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 4],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 0, 1])

        assert_correct_cmp_with_torch(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 2, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=3, spatial_shape_DWH=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

    def test_std_rules_2(self):
        indices_zyx = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 4],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 0, 1])

        assert_correct_cmp_with_torch(
            indices_zyx, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 2, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=3, spatial_shape_DWH=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])


    def test_std_rules_3(self):
        indices_zyx = torch.tensor([
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
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=8, spatial_shape_DWH=[8, 9, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=3, spatial_shape_DWH=[8, 11, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

    def test_std_rules_4(self):
        indices_zyx = torch.tensor([
            [0, 3, 14],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices_zyx, batch_size=1, spatial_shape_DWH=[6, 6, 20],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])

    def test_rule_real(self):

        indices_bzyx = torch.tensor([[ 0,  0,  7,  5],
        [ 0,  0, 13,  1],
        [ 0,  0,  0,  2],
        [ 0,  0,  1,  2],
        [ 0,  0,  4,  5],
        [ 0,  0,  6,  4],
        [ 0,  0,  7,  1],
        [ 0,  0,  8,  1],
        [ 0,  0,  9,  1],
        [ 0,  0, 10,  1],
        [ 0,  0, 11,  1],
        [ 0,  0, 12,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  2,  1],
        [ 0,  0,  3,  0],
        [ 0,  0,  4,  0],
        [ 0,  0,  4,  4],
        [ 0,  0,  4,  1],
        [ 0,  1, 11,  1],
        [ 0,  1, 12,  1],
        [ 0,  1, 13,  1],
        [ 0,  2,  4,  5],
        [ 0,  2,  5,  6],
        [ 0,  2,  6,  4],
        [ 0,  2,  7,  1],
        [ 0,  2,  7,  2],
        [ 0,  2,  8,  1],
        [ 0,  2,  9,  1],
        [ 0,  2, 10,  1],
        [ 0,  2, 11,  1],
        [ 0,  3,  6,  0],
        [ 0,  4, 13,  3],
        [ 0,  3,  5,  0],
        [ 0,  4,  7,  1],
        [ 0,  4,  8,  1],
        [ 0,  4,  9,  1],
        [ 0,  4, 10,  1],
        [ 0,  4, 11,  1],
        [ 0,  4, 12,  1],
        [ 0,  4, 12,  2],
        [ 0,  4, 13,  1],
        [ 0,  4,  0,  1],
        [ 0,  4,  1,  1],
        [ 0,  4,  3,  1],
        [ 0,  4,  4,  1],
        [ 0,  4,  5,  1],
        [ 0,  4,  6,  1],
        [ 0,  4,  7,  0],
        [ 0,  4,  8,  0],
        [ 0,  5,  2,  0],
        [ 0,  5,  2,  1],
        [ 0,  5,  3,  0],
        [ 0,  5,  3,  1],
        [ 0,  5,  4,  1],
        [ 0,  5,  4,  0],
        [ 0,  5,  5,  1],
        [ 0,  5,  5,  0],
        [ 0,  5,  6,  0],
        [ 0,  5,  6,  1],
        [ 0,  5,  9,  0],
        [ 0,  5, 10,  0],
        [ 0,  6, 12,  3],
        [ 0,  6,  7,  1],
        [ 0,  6, 10,  1],
        [ 0,  6, 11,  1],
        [ 0,  6, 12,  2],
        [ 0,  6, 12,  1],
        [ 0,  6, 13,  1],
        [ 0,  6, 13,  2],
        [ 0,  6,  0,  2],
        [ 0,  6,  1,  2],
        [ 0,  6,  1,  1],
        [ 0,  6,  3,  1],
        [ 0,  6,  4,  1],
        [ 0,  6,  5,  1],
        [ 0,  6,  6,  1],
        [ 0,  6,  7,  0],
        [ 0,  6,  8,  0],
        [ 0,  6,  8,  1],
        [ 0,  6,  9,  1],
        [ 0,  6,  9,  0],
        [ 0,  6,  0,  1],
        [ 0,  6,  2,  1],
        [ 0,  6,  2,  0],
        [ 0,  6,  3,  0],
        [ 0,  6,  4,  0],
        [ 0,  6,  5,  0],
        [ 0,  6,  6,  0],
        [ 0,  7, 12,  2],
        [ 0,  7,  7,  1],
        [ 0,  7,  8,  0],
        [ 0,  7,  8,  1],
        [ 0,  7,  9,  1],
        [ 0,  7,  2,  0],
        [ 0,  7,  3,  1],
        [ 0,  7,  4,  1],
        [ 0,  7,  5,  1],
        [ 0,  7,  6,  1],
        [ 0,  7,  7,  0],
        [ 0,  7,  9,  0],
        [ 0,  7, 10,  0],
        [ 0,  7,  3,  0],
        [ 0,  7,  4,  0],
        [ 0,  7,  6,  0],
        [ 0,  8, 10,  1],
        [ 0,  7, 11,  0],
        [ 0,  8, 11,  1],
        [ 0,  7,  0,  0],
        [ 0,  7,  5,  0],
        [ 0,  8,  7,  1],
        [ 0,  8,  7,  0],
        [ 0,  8,  8,  0],
        [ 0,  8,  8,  1],
        [ 0,  8,  9,  1],
        [ 0,  8, 10,  0],
        [ 0,  7,  1,  0],
        [ 0,  8,  2,  1],
        [ 0,  8,  2,  0],
        [ 0,  8,  3,  0],
        [ 0,  8,  3,  1],
        [ 0,  8,  4,  1],
        [ 0,  8,  4,  0],
        [ 0,  8,  5,  1],
        [ 0,  8,  6,  1],
        [ 0,  8,  6,  0],
        [ 0,  8,  0,  0],
        [ 0,  8,  1,  0],
        [ 0,  9, 13,  1],
        [ 0,  8,  1,  1],
        ], device='cuda:0', dtype=torch.int32)

        assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
            spatial_shape_DWH=[14, 14, 8], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[15, 16, 8], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[50, 20, 50], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[40, 40, 8], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[15, 30, 38], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[20, 50, 18], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)
        # assert_correct_cmp_with_spconv(indices_zyx=indices_bzyx,
        #     spatial_shape_DWH=[15, 16, 18], kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

    def test_rule_submconv(self):
        indices_zyx = torch.tensor([
            # [0, 0, 0, 0],
            # [0, 0, 0, 1],
            # [0, 0, 1, 0],
            # [0, 1, 1, 1],
            [0, 1, 0, 7],
            [0, 1, 1, 6],
            [0, 2, 1, 6],
            [0, 3, 3, 6],
            [0, 3, 3, 3],
            [0, 0, 0, 1],
            [0, 1, 3, 3],
            [0, 1, 3, 4],
            [0, 1, 3, 5],
            [0, 2, 3, 5],
            [0, 3, 3, 5],
            [0, 4, 3, 5],
            [0, 7, 3, 5],
        ], dtype=torch.int).cuda()
        D = 9
        W = 8
        H = 8
        spatial_shape_DWH = [D, W, H]
        inChannel = 16
        outChannel = 16
        batch_size = 1
        voxel_features = torch.arange( indices_zyx.shape[0],
                          dtype=torch.float, device=indices_zyx.device).repeat(inChannel).reshape((indices_zyx.shape[0], inChannel))
        voxel_features = torch.arange( inChannel,
                          dtype=torch.float, device=indices_zyx.device).repeat(indices_zyx.shape[0], 1)
        voxel_features = torch.arange( indices_zyx.shape[0] * inChannel,
                          dtype=torch.float, device=indices_zyx.device).reshape((indices_zyx.shape[0], inChannel))
        voxel_features = torch.ones((indices_zyx.shape[0], inChannel), dtype=torch.float, device=indices_zyx.device)
        # voxel_features = torch.randn((indices_zyx.shape[0], inChannel), dtype=torch.float, device=indices_zyx.device)
        # voxel_features = torch.zeros((indices_zyx.shape[0], inChannel), dtype=torch.float, device=indices_zyx.device) / l
        # voxel_features[0,0] = 1.
        # voxel_features[0,15] = 2.
        # voxel_features[1,0] = 2.
        # voxel_features[1,:] = 3.0
        # voxel_features[2,:] = 1.0

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape_DWH, batch_size, indices=indices_zyx)

        kernel_size = 3
        stride = 1
        padding = 1
        # padding must be 1, I think it's spconv's bug
        dilation = 1

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32

        oz_idx, oz_ptr, rules, rule_size  = get_rules_subm(
            tensor.z_idx, tensor.z_ptr,
            batch_size, spatial_shape_DWH, spatial_shape_DWH,
            [kernel_size, kernel_size, kernel_size],
            [stride, stride, stride],
            [padding, padding, padding],
            [dilation, dilation, dilation],
            [2, 2]
        )

        torch.set_printoptions(edgeitems=100)
        print("tensor.feature = ", tensor.feature)
        print("z_ptr = ", tensor.z_ptr)
        print("rules = ", rules[:,:,:,:4])
        print("ruleSize = ", rule_size)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_zyx, batch_size, spatial_shape_DWH, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, use_hash=False)

        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        # convolution
        weight = torch.zeros((kernel_size, kernel_size, kernel_size,
                             inChannel, outChannel), dtype=torch.float, device=indices_zyx.device)
        weight[0,0,0,1,1] = 888.0
        weight[0,0,1,1,1] = 8.0
        weight = torch.randn((kernel_size, kernel_size, kernel_size,
                             inChannel, outChannel), dtype=torch.float, device=indices_zyx.device)

        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(out_features, outids, spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, inChannel, outChannel)),
            rules, rule_size, oz_idx.shape[0])

        # print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, spatial_shape_DWH, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

        print("spconv out_features = ", out_features)
        print("sphconv out_features = ", sph_out_features)

        print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        print("spconv_dense shape = ", spconv_dense.shape)
        print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        print("sphconv_dense shape = ", sphconv_dense.shape)
        print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
        print("sum  = ", (spconv_dense.sum() - sphconv_dense.sum()).sum())

        assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.1))


    def test_rule_conv_1(self):
        indices_bzyx = torch.tensor([
        [ 0,  0,  2, 12],
        [ 0,  0,  2,  2],
        [ 0,  0,  2, 17],
        [ 0,  0,  2,  3],
        [ 0,  0,  4,  2],
        [ 0,  0,  2,  4],
        [ 0,  0,  2, 16],
        [ 0,  0,  5,  2],
        [ 0,  0,  1, 13],
        [ 0,  1,  2, 13],
        [ 0,  0,  2,  0],
        [ 0,  1,  2, 17],
        [ 0,  0,  3,  5],
        [ 0,  1,  0,  5],
        [ 0,  1,  2, 12],
        [ 0,  1,  3,  7],
        [ 0,  1,  3, 14],
        [ 0,  1,  3, 13],
        [ 0,  1,  3,  2],
        [ 0,  1,  3,  3],
        [ 0,  1,  3,  5],
        [ 0,  1,  3,  4],
        [ 0,  1,  4,  3],
        [ 0,  1,  5,  3],
        [ 0,  1,  5,  2],
        [ 0,  1,  5,  4],
        [ 0,  1,  0,  3],
        [ 0,  1,  0,  2],
        [ 0,  1,  1,  2],
        [ 0,  1,  2,  2],
        [ 0,  1,  2,  3],
        [ 0,  1,  2, 10],
        [ 0,  1,  4,  2],
        [ 0,  1,  5,  7],
        [ 0,  1,  5,  5],
        [ 0,  1,  1,  4],
        [ 0,  1,  5,  6],
        [ 0,  1,  5,  8],
        [ 0,  2,  5,  7],
        [ 0,  1,  1,  3],
        [ 0,  2,  2,  6],
        [ 0,  2,  3,  3],
        [ 0,  2,  3,  2],
        [ 0,  2,  4,  3],
        [ 0,  2,  5,  3],
        [ 0,  2,  5,  8],
        [ 0,  2,  5,  4],
        [ 0,  2,  5,  5],
        [ 0,  2,  0,  2],
        [ 0,  2,  1,  2],
        [ 0,  2,  1,  3],
        [ 0,  2,  1,  4],
        [ 0,  2,  2,  3],
        [ 0,  2,  2,  2],
        [ 0,  2,  1,  1],
        [ 0,  2,  3,  4],
        [ 0,  2,  2,  0],
        [ 0,  2,  0,  5],
        [ 0,  2,  2,  4],
        [ 0,  3,  5,  6],
        [ 0,  3,  3,  2],
        [ 0,  3,  4,  4],
        [ 0,  3,  4,  7],
        [ 0,  3,  5,  3],
        [ 0,  3,  5,  4],
        [ 0,  3,  0,  3],
        [ 0,  3,  0,  2],
        [ 0,  3,  1,  2],
        [ 0,  3,  1,  3],
        [ 0,  3,  2,  2],
        [ 0,  3,  4,  1],
        [ 0,  3,  5,  2],
        [ 0,  3,  1,  1],
        [ 0,  3,  3,  1],
        ], dtype=torch.int).cuda()
        D = 6
        W = 6
        H = 20
        spatial_shape_DWH = [D, W, H]
        inChannel = 64
        outChannel = 64
        batch_size = 1
        voxel_features = torch.arange( indices_bzyx.shape[0],
                          dtype=torch.float, device=indices_bzyx.device).repeat(inChannel).reshape((indices_bzyx.shape[0], inChannel))
        voxel_features = torch.arange( inChannel,
                          dtype=torch.float, device=indices_bzyx.device).repeat(indices_bzyx.shape[0], 1)
        voxel_features = torch.arange( indices_bzyx.shape[0] * inChannel,
                          dtype=torch.float, device=indices_bzyx.device).reshape((indices_bzyx.shape[0], inChannel))
        voxel_features = torch.zeros((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        # voxel_features = torch.ones((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        # voxel_features = torch.randn((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        voxel_features[0,:] = 1.0

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)

        kernel_size = [3, 3, 3]
        stride = [2, 2, 2]
        padding = [1, 1, 1]
        # padding must be 1, I think it's spconv's bug
        dilation = [1, 1, 1]

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32

        out_spatial_shape_DWH = out_spatial(
                spatial_shape_DWH, kernel_size, stride, padding, dilation)
        print("out shape = ", out_spatial_shape_DWH)

        oz_idx, oz_ptr, rules, rule_size  = get_rules(
            tensor.z_idx, tensor.z_ptr,
            batch_size, spatial_shape_DWH, out_spatial_shape_DWH,
            kernel_size,
            stride,
            padding,
            dilation,
            [2, 2]
        )

        torch.set_printoptions(edgeitems=100)
        print("tensor.feature = ", tensor.feature)
        print("z_ptr = ", tensor.z_ptr)
        print("oz_ptr = ", oz_ptr)
        print("rules = ", rules[:,:,:,:4])
        print("ruleSize = ", rule_size)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
            stride, padding, dilation, out_padding=0, subm=False,
            transpose=False, use_hash=False)

        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        # convolution
        weight = torch.randn((*kernel_size, inChannel, outChannel), dtype=torch.float, device=indices_bzyx.device)

        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(
            out_features, outids, out_spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, inChannel, outChannel)),
            rules, rule_size, oz_idx.shape[0])

        # print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

        print("sphconv out_features = ", sph_out_features)

        print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        print("spconv_dense shape = ", spconv_dense.shape)
        print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        print("sphconv_dense shape = ", sphconv_dense.shape)

        print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
        assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.1))


    def test_rule_conv_2(self):
        indices_bzyx = torch.tensor( [
        [0, 0, 0, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 2],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        ],
        dtype=torch.int).cuda()
        D = 3
        W = 3
        H = 3
        spatial_shape_DWH = [D, W, H]
        inChannel = 32
        outChannel = 32
        batch_size = 1
        voxel_features = torch.arange( indices_bzyx.shape[0],
                          dtype=torch.float, device=indices_bzyx.device).repeat(inChannel).reshape((indices_bzyx.shape[0], inChannel))
        voxel_features = torch.arange( inChannel,
                          dtype=torch.float, device=indices_bzyx.device).repeat(indices_bzyx.shape[0], 1)
        voxel_features = torch.arange( indices_bzyx.shape[0] * inChannel,
                          dtype=torch.float, device=indices_bzyx.device).reshape((indices_bzyx.shape[0], inChannel))
        voxel_features = torch.zeros((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        # voxel_features = torch.ones((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        # voxel_features = torch.randn((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        voxel_features[3,:] = 1.0

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        # padding must be 1, I think it's spconv's bug
        dilation = [1, 1, 1]
        subm = True

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32

        out_spatial_shape_DWH = out_spatial(
                spatial_shape_DWH, kernel_size, stride, padding, dilation)
        print("out shape = ", out_spatial_shape_DWH)

        test_func = get_rules_subm if subm else get_rules

        oz_idx, oz_ptr, rules, rule_size  = test_func(
            tensor.z_idx, tensor.z_ptr,
            batch_size, spatial_shape_DWH, out_spatial_shape_DWH,
            kernel_size,
            stride,
            padding,
            dilation,
            [2, 3]
        )

        torch.set_printoptions(edgeitems=100)
        print("tensor.feature = ", tensor.feature)
        print("z_ptr = ", tensor.z_ptr)
        print("oz_ptr = ", oz_ptr)
        print("rules = ", rules[:,:,:,:16])
        print("ruleSize = ", rule_size)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
            stride, padding, dilation, out_padding=0, subm=subm,
            transpose=False, use_hash=False)

        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        assert torch.sum(indice_pair_num) == torch.sum(rule_size)

        if (rule_size.shape[0] == 1) :
            assert (indice_pair_num.view(-1).sort()[0] == rule_size.view(-1).sort()[0]).all()
        else : # tiled version
            assert (indice_pair_num.view(-1).sort()[0] == rule_size.sum(dim=0).view(-1).sort()[0]).all()
            # NTile = rules.shape[0]
            # loadingRule = rules[:,:,0,:].reshape([NTile,-1])
        assert (outids[:,3].sort()[0] == oz_idx.sort()[0]).all()

        # convolution
        weight = torch.randn((*kernel_size, inChannel, outChannel), dtype=torch.float, device=indices_bzyx.device)

        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(
            out_features, outids, out_spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, inChannel, outChannel)),
            rules, rule_size, oz_idx.shape[0])

        print("early distance =", sph_out_features.sum() - out_features.sum())

        # print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)

        print("sphconv out_features = ", sph_out_features)

        print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        print("spconv_dense shape = ", spconv_dense.shape)
        print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        print("sphconv_dense shape = ", sphconv_dense.shape)

        print(" sph out(021)=", sphconv_dense[0,0,0,2,1])
        print(" sp out(021)=", spconv_dense[0,0,0,2,1])

        print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
        print("distance2 = ", (spconv_dense.sum() - sphconv_dense.sum()))
        assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.01))


    def test_rules_sum(self):
        z_idx = torch.load("z_idx.pt")
        z_ptr = torch.load("z_ptr.pt")

        oz_idx, oz_ptr, rules, rule_size = get_rules_subm(
            z_idx,
            z_ptr,
            1,
            [14, 14, 8],
            [14, 14, 8],
            [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2])

        print("rules.sum = ", rules.sum())

