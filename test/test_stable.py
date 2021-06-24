
from typing import List

import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.sphconv_cuda import get_rules, get_rules_subm, rule_conv
from sphconv.utils import out_spatial


class TestClass:
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
        inChannel = 64
        outChannel = 64
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
        voxel_features = torch.ones((indices_bzyx.shape[0], inChannel), dtype=torch.float, device=indices_bzyx.device) * 100
        voxel_features[0,:] = 8.0
        voxel_features[3,:] = 16.0

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

        # torch.set_printoptions(edgeitems=100)
        # print("tensor.feature = ", tensor.feature)
        # print("z_ptr = ", tensor.z_ptr)
        # print("oz_ptr = ", oz_ptr)
        # print("rules = ", rules[:,:,:,:4])
        # print("ruleSize = ", rule_size)

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices_bzyx, batch_size, spatial_shape_DWH, kernel_size,
            stride, padding, dilation, out_padding=0, subm=False,
            transpose=False, use_hash=False)


        # convolution
        weight = torch.randn((*kernel_size, inChannel, outChannel), dtype=torch.float, device=indices_bzyx.device) / 2
        # weight = torch.randint( 1, 660, (*kernel_size, inChannel, outChannel), dtype=torch.float, device=indices_bzyx.device) / 2
        weight[1, :,:] = 8.0
        weight[-1, :,:] = 100.
        weight[1,2,0, :5,:] = 1/64

        out_features_gpu = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs,
            indice_pair_num, outids.shape[0])

        out_features_cpu = spconv.ops.indice_conv(
            voxel_features.cpu(), weight.cpu(), indice_pairs.cpu(),
            indice_pair_num.cpu(), outids.shape[0])

        spconv_dense_gpu = spconv.SparseConvTensor(
            out_features_gpu, outids, out_spatial_shape_DWH, batch_size).dense()
        spconv_dense_cpu = spconv.SparseConvTensor(
            out_features_cpu, outids, out_spatial_shape_DWH, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.feature, weight.reshape((-1, inChannel, outChannel)),
            rules, rule_size, batch_size, spatial_shape_DWH, out_spatial_shape_DWH, oz_idx.shape[0])

        sph_out_features_1 = sph_out_features
        # for i in range(20):
        #     sph_out_features_new = rule_conv(
        #             tensor.feature, weight.reshape((-1, inChannel, outChannel)),
        #             rules, rule_size, batch_size, spatial_shape_DWH, out_spatial_shape_DWH, oz_idx.shape[0])
            # print("new distance = ", (sph_out_features_new - sph_out_features_1).abs().sum())

        # print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, out_spatial_shape_DWH, batch_size, z_ptr=oz_ptr, z_idx=oz_idx).dense(tensor.device)
        sphconv_dense_1 = sphconv_dense

        spconv_dense = spconv_dense_cpu
        sphconv_dense = sphconv_dense.cpu()

        print("distance = ", (spconv_dense - sphconv_dense).abs().sum())
        print("cpu gpu dist = ", (spconv_dense_cpu - spconv_dense_gpu.cpu()).abs().sum())
        assert torch.all(torch.isclose(spconv_dense, sphconv_dense, rtol=0.01))


        subm = False
        Sphconv_Conv3d = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
        sph_conv = Sphconv_Conv3d(
            inChannel, outChannel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()
        sph_conv.weight = torch.nn.Parameter(weight)

        with torch.no_grad():
            sphconv_dense = sph_conv(tensor).dense()

        sphconv_dense_1 = sphconv_dense
        # for i in range(20):
        #     with torch.no_grad():
        #         tensor = sphconv.SparseConvTensor(
        #             voxel_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)
        #         sphconv_dense_new = sph_conv(tensor).dense()
                # print("new sphconv distance = ", (sphconv_dense_new - sphconv_dense_1).abs().sum())


