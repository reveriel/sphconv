
from typing import List

import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial



class TestClass:
    def test_dense(self):
        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        features = torch.randn(
            (indices_bzyx.shape[0], 4), dtype=torch.float, device=indices_bzyx.device)
        tensor = sphconv.SparseConvTensor(features, [2, 2, 2], 1, indices=indices_bzyx)
        tensor.feature.requires_grad = True
        y = tensor.dense().sum() * 2
        y.backward()
        print("tensor.feature.grad = ", tensor.feature.grad)
        assert((tensor.feature.grad == 2).all())


    def test_result_stable(self):
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
        in_channels = 64
        out_channels = 64
        batch_size = 1
        # voxel_features = torch.arange( indices_bzyx.shape[0],
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(inChannel).reshape((indices_bzyx.shape[0], inChannel))
        # voxel_features = torch.arange( inChannel,
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(indices_bzyx.shape[0], 1)
        # voxel_features = torch.arange( indices_bzyx.shape[0] * inChannel,
        #                   dtype=torch.float, device=indices_bzyx.device).reshape((indices_bzyx.shape[0], inChannel))
        # voxel_features = torch.zeros((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)
        # voxel_features = torch.ones((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device)

        torch.manual_seed(0)
        voxel_features = torch.ones((indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device) * 100
        voxel_features[0,:] = 8.0
        voxel_features[3,:] = 16.0

        subm = True
        kernel_size = [3, 3, 3]
        stride = [2, 2, 2]
        padding = [1, 1, 1]
        # padding must be 1, I think it's spconv's bug
        dilation = [1, 1, 1]

        spconv_tensor = spconv.SparseConvTensor(
            voxel_features, indices_bzyx, spatial_shape_DWH, batch_size)

        Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
        sp_conv = Spconv_Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()


        sph_tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)

        Sphconv_Conv3d = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
        sph_conv = Sphconv_Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

        # convolution
        weight = torch.randn((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device) / 2
        # weight = torch.randint( 1, 660, (*kernel_size, inChannel, outChannel), dtype=torch.float, device=indices_bzyx.device) / 2
        weight[1, :,:] = 8.0
        weight[-1, :,:] = 100.
        weight[1,2,0, :5,:] = 1/64


        spconv_tensor.features.requires_grad = True
        sph_conv.weight = torch.nn.Parameter(weight.clone())
        sp_conv.weight = torch.nn.Parameter(weight.clone())


        spconv_sum = spconv_dense = sp_conv(spconv_tensor).dense().sum()
        spconv_sum.backward()

        print("spconv: d weight = ", sp_conv.weight.grad)
        print("spconv: d feature = ", spconv_tensor.features.grad)






