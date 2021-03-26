
from typing import List

import spconv
import sphconv
import torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator
from sphconv.datagen import merge_batch_torch


def dup_with_batch_idx(indices, batch_size):
    example = {"coordinates": indices[:, 1:]}
    return merge_batch_torch([example]*batch_size)["coordinates"]

def assert_correct_cmp_with_spconv(
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: List[int],
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):

    if batch_size > 1:
        indices = dup_with_batch_idx(indices, batch_size)

    voxel_features = torch.ones((indices.shape[0], 1), device=indices.device)

    test_func = get_rules_subm if subm else get_rules

    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape, batch_size, indices=indices)

    assert tensor.z_idx.dim() == 1
    assert tensor.z_ptr.dim() == 3
    assert tensor.grid.dim() == 4
    assert tensor.z_idx.dtype == torch.int32
    assert tensor.z_ptr.dtype == torch.int32
    assert tensor.grid.dtype == torch.int32

    out_spatial_shape = out_spatial(
        spatial_shape, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape)

    oz_idx, oz_ptr, rules, rule_size = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        torch.empty([batch_size, *out_spatial_shape],
                    dtype=torch.int32, device=indices.device),
        batch_size,
        spatial_shape,
        out_spatial_shape,
        kernel_size, stride, padding, dilation)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices, batch_size, spatial_shape[::-1], kernel_size[::-1],
        stride[::-1], padding[::-1], dilation[::-1],
        out_padding=0, subm=subm, transpose=False, grid=None, use_hash=False)

    print("outids = ", outids)
    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    print("oz_idx = ", oz_idx)
    print("oz_ptr = ", oz_ptr)
    print("rules = ", rules)
    print("rule size = ", rule_size)

    # assert torch.sum(indice_pair_num) == torch.sum(rule_size)
    assert (indice_pair_num.view(-1).sort()[0] == rule_size.view(-1).sort()[0]).all()


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_rules(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()
        #####
        ## padding must be 1, to make spconv correct
        ## stride used always be 1, since its submanifold
        #####

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[2, 2, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[2, 3, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_std_rules(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[2, 2, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 2, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

    def test_rule_submconv(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            # [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        D = 2
        W = 2
        H = 2
        spatial_shape = [D, W, H]
        inChannel = 4
        batch_size = 1
        voxel_features = torch.ones(
            (indices.shape[0], inChannel), dtype=torch.float, device=indices.device)

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape, batch_size, indices=indices)

        kernel_size = 2
        stride = 1
        padding = 1
        # I padding must be 1, I think it's spconv's bug
        dilation = 1

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.grid.dim() == 4
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32
        assert tensor.grid.dtype == torch.int32

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

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

        # convolution
        outChannel = 4
        weight = torch.ones((kernel_size, kernel_size, kernel_size,
                             outChannel, inChannel), dtype=torch.float, device=indices.device)

        # print("indice_pairs dtype = ", indice_pairs.dtype)
        # print("indice_pair_num dtype = ", indice_pair_num.dtype)
        # indice_pairs = indice_pairs.int()
        # indice_pair_num = indice_pair_num.int()
        # print("indice_pairs dtype = ", indice_pairs.dtype)
        # print("indice_pair_num dtype = ", indice_pair_num.dtype)
        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(out_features, indices, spatial_shape, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.features, weight.reshape((-1, outChannel, inChannel)),
            rules, rule_size, batch_size, spatial_shape, spatial_shape)

        print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, spatial_shape, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

        # print("sphconv out_features = ", sph_out_features)

        # print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        # print("spconv_dense shape = ", spconv_dense.shape)
        # print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        # print("sphconv_dense shape = ", sphconv_dense.shape)

        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

