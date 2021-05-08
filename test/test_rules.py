
from typing import List

import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial


def dup_with_batch_idx(indices_zyx:torch.Tensor, batch_size:int):
    example = {"coordinates": indices_zyx }
    return merge_batch_torch([example]*batch_size)["coordinates"]


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

    indices_zyx = dup_with_batch_idx(indices_zyx, batch_size)

    voxel_features = torch.ones((indices_zyx.shape[0], 1), device=indices_zyx.device)
    test_func = get_rules_subm if subm else get_rules
    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_DWH, batch_size, indices=indices_zyx)
    out_spatial_shape = spatial_shape_DWH
    if not subm:
        out_spatial_shape = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    print("out shape(DWH) = ", out_spatial_shape)

    oz_idx, oz_ptr, rules, rule_size, _ = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        batch_size,
        spatial_shape_DWH, # [DWH]
        out_spatial_shape,
        kernel_size, stride, padding, dilation)

    # assert torch.sum(indice_pair_num) == torch.sum(rule_size)

    sphconv_dense = sphconv.SparseConvTensor(
        torch.ones((oz_idx.shape[0], 1), device=indices_zyx.device), out_spatial_shape, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(indices_zyx.device)

    weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32, device=indices_zyx.device)

    torch_dense = torch.nn.functional.conv3d(tensor.dense(
        indices_zyx.device), weight, None, stride, padding, dilation)
    torch_dense[torch_dense != 0] = 1

    print("sphconv_dense = ", sphconv_dense)
    print("torch_dense = ", torch_dense)

    assert (torch.isclose(sphconv_dense, torch_dense)).all()



def assert_correct_cmp_with_spconv(
        indices_zyx: torch.Tensor,
        batch_size: int,
        spatial_shape_DWH: List[int],
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int] = [1, 1, 1],
        subm: bool = False):
    if subm:
        assert dilation == [1, 1, 1] and stride == [1, 1, 1]

    indices_zyx = dup_with_batch_idx(indices_zyx, batch_size)

    voxel_features = torch.ones((indices_zyx.shape[0], 1), device=indices_zyx.device)

    test_func = get_rules_subm if subm else get_rules

    tensor = sphconv.SparseConvTensor(
        voxel_features, spatial_shape_DWH, batch_size, indices=indices_zyx)

    assert tensor.z_idx.dim() == 1
    assert tensor.z_ptr.dim() == 3
    assert tensor.z_idx.dtype == torch.int32
    assert tensor.z_ptr.dtype == torch.int32

    out_spatial_shape_DWH = spatial_shape_DWH
    if not subm:
        out_spatial_shape_DWH = out_spatial(
            spatial_shape_DWH, kernel_size, stride, padding, dilation)
    print("out shape = ", out_spatial_shape_DWH)
    print("z_idx = ", tensor.z_idx)
    print("z_ptr = ", tensor.z_ptr)

    oz_idx, oz_ptr, local_rules, rule_size, global_rules = test_func(
        tensor.z_idx,
        tensor.z_ptr,
        batch_size,
        spatial_shape_DWH,
        out_spatial_shape_DWH,
        kernel_size, stride, padding, dilation)

    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
        indices_zyx, batch_size, spatial_shape_DWH, kernel_size,
        stride, padding, dilation,
        out_padding=0, subm=subm, transpose=False, use_hash=False)

    print("outids = ", outids)
    print("indice_pairs = ", indice_pairs)
    print("indice_pair_num = ", indice_pair_num)

    print("oz_idx = ", oz_idx)
    print("oz_ptr = ", oz_ptr)
    print("rules = ", local_rules)
    print("rule size = ", rule_size)

    assert torch.sum(indice_pair_num) == torch.sum(rule_size)
    if (rule_size.shape[0] == 1) :
        assert (indice_pair_num.view(-1).sort()[0] == rule_size.view(-1).sort()[0]).all()
    else : # tiled version
        assert (indice_pair_num.view(-1).sort()[0] == rule_size.sum(dim=0).view(-1).sort()[0]).all()
        # NTile = rules.shape[0]
        # loadingRule = rules[:,:,0,:].reshape([NTile,-1])
        # print("loadingRule.shape  = ", loadingRule.shape)
        # print("uniq loadingRule  = ", loadingRule.unique(dim=1))
        # print("uniq loadingRule. shape  = ", loadingRule.unique(dim=1).shape)

    assert (outids[:,3].sort()[0] == oz_idx.sort()[0]).all()


    if not subm:
        # check oz_ptr
        spconv_dense = spconv.SparseConvTensor(
            torch.ones((outids.shape[0], 1), device=indices_zyx.device), outids, out_spatial_shape_DWH, batch_size).dense()
        sphconv_dense = sphconv.SparseConvTensor(
            torch.ones((oz_idx.shape[0], 1), device=indices_zyx.device), out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense()

        print("sphconv = ", sphconv_dense)
        print("spconv = ", spconv_dense)

        weight = torch.ones((1, 1, *kernel_size),
                        dtype=torch.float32, device=indices_zyx.device)
        torch_dense = torch.nn.functional.conv3d(tensor.dense(
            indices_zyx.device), weight, None, stride, padding, dilation)
        torch_dense[torch_dense != 0] = 1

        print("torch_dense = ", torch_dense)

        assert (spconv_dense == sphconv_dense).all()



class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_subm_rules_1(self):
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
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 2, 3],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[2, 3, 2],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=3, spatial_shape_DWH=[4, 6, 5],
            kernel_size=[2, 3, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[3, 3, 3],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

    def test_subm_rules_3(self):
        indices = torch.tensor([
            [0, 0, 0],
        ], dtype=torch.int).cuda()

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_DWH=[2, 1, 1],
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
            indices, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[8, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[9, 9, 9],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=1, spatial_shape_DWH=[8, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[9, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=2, spatial_shape_DWH=[11, 8, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

        assert_correct_cmp_with_spconv(
            indices, batch_size=8, spatial_shape_DWH=[10, 9, 8],
            kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)

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


    def test_rule_submconv(self):
        indices_zyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()
        D = 2
        W = 2
        H = 2
        spatial_shape_DWH = [D, W, H]
        inChannel = 32
        outChannel = 32
        batch_size = 1
        voxel_features = torch.arange( indices_zyx.shape[0],
                          dtype=torch.float, device=indices_zyx.device).repeat(inChannel).reshape((indices_zyx.shape[0], inChannel))
        voxel_features = torch.arange( inChannel,
                          dtype=torch.float, device=indices_zyx.device).repeat(indices_zyx.shape[0], 1)
        voxel_features = torch.arange( indices_zyx.shape[0] * inChannel,
                          dtype=torch.float, device=indices_zyx.device).reshape((indices_zyx.shape[0], inChannel))

        # voxel_features = torch.zeros((indices_zyx.shape[0], inChannel), dtype=torch.float, device=indices_zyx.device) / 5
        # voxel_features[0,:] = 1.0

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

        oz_idx, oz_ptr, rules, rule_size, global_rules = get_rules_subm(
            tensor.z_idx, tensor.z_ptr,
            batch_size, spatial_shape_DWH, spatial_shape_DWH,
            [kernel_size, kernel_size, kernel_size],
            [stride, stride, stride],
            [padding, padding, padding],
            [dilation, dilation, dilation]
        )

        torch.set_printoptions(edgeitems=100)
        print("tensor.feature = ", tensor.feature)
        print("z_ptr = ", tensor.z_ptr)
        print("rules = ", rules[:,:,:,:4])
        print("ruleSize = ", rule_size)
        print("global_rules = ", global_rules)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_zyx, batch_size, spatial_shape_DWH, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, use_hash=False)

        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        # convolution
        weight = torch.ones((kernel_size, kernel_size, kernel_size,
                             outChannel, inChannel), dtype=torch.float, device=indices_zyx.device)

        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(out_features, outids, spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, outChannel, inChannel)),
            rules, rule_size, global_rules, batch_size, spatial_shape_DWH, spatial_shape_DWH, oz_idx.shape[0])

        # print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, spatial_shape_DWH, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

        print("sphconv out_features = ", sph_out_features)

        print("spconv_dense = ", spconv_dense[0,:,0,0,:])
        print("spconv_dense shape = ", spconv_dense.shape)
        print("sphconv_dense = ", sphconv_dense[0,:,0,0,:])
        print("sphconv_dense shape = ", sphconv_dense.shape)

        assert torch.all(torch.isclose(spconv_dense, sphconv_dense))


    def test_rule_conv(self):
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

        oz_idx, oz_ptr, rules, rule_size, global_rules = get_rules(
            tensor.z_idx, tensor.z_ptr,
            batch_size, spatial_shape_DWH, out_spatial_shape_DWH,
            kernel_size,
            stride,
            padding,
            dilation
        )

        torch.set_printoptions(edgeitems=100)
        print("tensor.feature = ", tensor.feature)
        print("z_ptr = ", tensor.z_ptr)
        print("oz_ptr = ", oz_ptr)
        print("rules = ", rules[:,:,:,:4])
        print("ruleSize = ", rule_size)
        print("global_rules = ", global_rules)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
            stride, padding, dilation, out_padding=0, subm=False,
            transpose=False, use_hash=False)

        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        # convolution
        weight = torch.randn((*kernel_size, outChannel, inChannel), dtype=torch.float, device=indices_bzyx.device)

        out_features = spconv.ops.indice_conv(
            voxel_features, weight.permute((0,1,2,4,3)), indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(
            out_features, outids, out_spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, outChannel, inChannel)),
            rules, rule_size, global_rules, batch_size, spatial_shape_DWH, out_spatial_shape_DWH, oz_idx.shape[0])

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
