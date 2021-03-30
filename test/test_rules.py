
from typing import List

import spconv
import sphconv
import torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial, voxel_generator
from sphconv.datagen import merge_batch_torch


def dup_with_batch_idx(indices_zyx:torch.Tensor, batch_size:int):
    example = {"coordinates": indices_zyx }
    return merge_batch_torch([example]*batch_size)["coordinates"]


def assert_correct_cmp_with_torch(
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape_HWD: List[int], #[HWD]
        kernel_size: List[int], #[HWD]
        stride: List[int],
        padding: List[int],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):
    assert subm == False

    indices = dup_with_batch_idx(indices, batch_size)

    voxel_features = torch.ones((indices.shape[0], 1), device=indices.device)
    test_func = get_rules_subm if subm else get_rules
    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_HWD, batch_size, indices=indices)
    out_spatial_shape = spatial_shape_HWD
    if not subm:
        out_spatial_shape = out_spatial(
            spatial_shape_HWD, kernel_size, stride, padding, dilation)
    print("out shape(DWH) = ", out_spatial_shape)

    oz_idx, oz_ptr, rules, rule_size = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        torch.empty([batch_size, *out_spatial_shape],
                    dtype=torch.int32, device=indices.device),
        batch_size,
        spatial_shape_HWD, # [HWD]
        out_spatial_shape,
        kernel_size, stride, padding, dilation)

    # assert torch.sum(indice_pair_num) == torch.sum(rule_size)

    sphconv_dense = sphconv.SparseConvTensor(
        torch.ones((oz_idx.shape[0], 1)), out_spatial_shape[::-1], batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(indices.device)

    weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32, device=indices.device)

    torch_dense = torch.nn.functional.conv3d(tensor.dense(
        indices.device), weight, None, stride, padding, dilation)
    torch_dense[torch_dense != 0] = 1

    print("sphconv_dense = ", sphconv_dense)
    print("torch_dense = ", torch_dense)

    assert (torch.isclose(sphconv_dense, torch_dense)).all()



def assert_correct_cmp_with_spconv(
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape_HWD: List[int], #[H,W,D]
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):
    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]

    indices = dup_with_batch_idx(indices, batch_size)

    voxel_features = torch.ones((indices.shape[0], 1), device=indices.device)

    test_func = get_rules_subm if subm else get_rules

    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_HWD[::-1], batch_size, indices=indices)

    assert tensor.z_idx.dim() == 1
    assert tensor.z_ptr.dim() == 3
    assert tensor.grid.dim() == 4
    assert tensor.z_idx.dtype == torch.int32
    assert tensor.z_ptr.dtype == torch.int32
    assert tensor.grid.dtype == torch.int32

    out_spatial_shape_HWD = spatial_shape_HWD
    if not subm:
        out_spatial_shape_HWD = out_spatial(
            spatial_shape_HWD, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape_HWD)
    print("z_idx = ", tensor.z_idx)
    print("z_ptr = ", tensor.z_ptr)

    oz_idx, oz_ptr, rules, rule_size = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        torch.empty([batch_size, *out_spatial_shape_HWD],
                    dtype=torch.int32, device=indices.device),
        batch_size,
        spatial_shape_HWD,
        out_spatial_shape_HWD,
        kernel_size, stride, padding, dilation)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices, batch_size, spatial_shape_HWD[::-1], kernel_size[::-1],
        stride[::-1], padding[::-1], dilation[::-1],
        out_padding=0, subm=subm, transpose=False, grid=None, use_hash=False)

    print("outids = ", outids)
    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    print("oz_idx = ", oz_idx)
    print("oz_ptr = ", oz_ptr)
    print("rules = ", rules)
    print("rule size = ", rule_size)

    assert torch.sum(indice_pair_num) == torch.sum(rule_size)
    assert (indice_pair_num.view(-1).sort()[0] == rule_size.view(-1).sort()[0]).all()
    assert (outids[:,1].sort()[0] == oz_idx.sort()[0]).all()


    if not subm:
        # check oz_ptr
        spconv_dense = spconv.SparseConvTensor(
            torch.ones((outids.shape[0], 1)), outids, out_spatial_shape_HWD[::-1], batch_size).dense()
        sphconv_dense = sphconv.SparseConvTensor(
            torch.ones((oz_idx.shape[0],1)), out_spatial_shape_HWD[::-1], batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense()

        print("sphconv = ", sphconv_dense)
        print("spconv = ", spconv_dense)

        weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32, device=indices.device)
        torch_dense = torch.nn.functional.conv3d(tensor.dense(
            indices.device), weight, None, stride, padding, dilation)
        torch_dense[torch_dense != 0] = 1

        print("torch_dense = ", torch_dense)

        assert (spconv_dense == sphconv_dense).all()



class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_rules(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()
        #####
        ## padding must be 1, to make spconv correct
        ## stride used always be 1, since its submanifold
        #####

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 3, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_HWD=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_subm_rules_3(self):
        indices = torch.tensor([
            [0, 0, 0],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_HWD=[2, 1, 1],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

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
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_HWD=[8, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[9, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_HWD=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_HWD=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_HWD=[10, 9, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_std_rules(self):
        indices = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 4],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 0, 1])

        assert_correct_cmp_with_torch(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 2, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_HWD=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

    def test_std_rules_2(self):
        indices = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 4],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 0, 1])

        assert_correct_cmp_with_torch(
            indices, batch_size=1, spatial_shape_HWD=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[2, 2, 2], stride=[1, 2, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_HWD=[8, 3, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])


    def test_std_rules_3(self):
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
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0])

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_HWD=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_HWD=[8, 9, 8],
            kernel_size=[3, 3, 1], stride=[2, 2, 2], padding=[1, 1, 1])

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_HWD=[8, 11, 8],
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
        spatial_shape_DWH = [D, W, H]
        inChannel = 4
        batch_size = 1
        voxel_features = torch.ones(
            (indices.shape[0], inChannel), dtype=torch.float, device=indices.device)

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape_DWH, batch_size, indices=indices)

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
            spatial_shape_DWH,
            spatial_shape_DWH,
            [kernel_size, kernel_size, kernel_size],
            [stride, stride, stride],
            [padding, padding, padding],
            [dilation, dilation, dilation]
        )

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape_DWH, kernel_size, stride, padding, dilation,
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

        spconv_dense = spconv.SparseConvTensor(out_features, indices, spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, outChannel, inChannel)),
            rules, rule_size, batch_size, spatial_shape_DWH, spatial_shape_DWH, oz_idx.shape[0])

        print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, spatial_shape_DWH, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

        # print("sphconv out_features = ", sph_out_features)

        # print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        # print("spconv_dense shape = ", spconv_dense.shape)
        # print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        # print("sphconv_dense shape = ", sphconv_dense.shape)

        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

