from typing import List
import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch
from sphconv.utils import out_spatial
from sphconv.sphconv_cuda import rule_conv_d_feature, rule_conv_backward
from common import batch_real_test_inputs


def d_torch_conv(
    indice_bzyx: torch.Tensor,
    feature_: torch.Tensor,
    weight: torch.Tensor,
    spatial_shape_DWH: List[int],
    batch_size:int, ic, oc, kernel_size, stride, padding, dilation,
    subm:bool, lib:str):
    """ return y and tensor.grad """

    assert (subm == False)

    feature:torch.Tensor = feature_.clone()

    tensor_0 = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indice_bzyx).dense()
    tensor_1 = spconv.SparseConvTensor(
        feature, indice_bzyx, spatial_shape_DWH, batch_size).dense()
    assert(torch.isclose(tensor_0, tensor_1).all())
    tensor = tensor_0
    tensor.requires_grad = True
    conv = torch.nn.Conv3d(ic, oc, kernel_size, stride, padding, dilation, bias=False)

    conv.weight = torch.nn.Parameter(weight.permute(3,4,0,1,2).contiguous().clone())

    y = conv(tensor).sum()
    y.backward()

    return y, tensor.grad


def d_feature_conv(
    indice_bzyx: torch.Tensor,
    feature_: torch.Tensor,
    weight: torch.Tensor,
    spatial_shape_DWH: List[int],
    batch_size:int, ic, oc, kernel_size, stride, padding, dilation,
    subm:bool, lib:str):

    feature:torch.Tensor = feature_.clone()
    feature.requires_grad = True

    if lib == 'spconv':
        tensor = spconv.SparseConvTensor(
            feature, indice_bzyx, spatial_shape_DWH, batch_size)
        conv_class = spconv.SubMConv3d if subm else spconv.SparseConv3d
        conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)
    elif lib == 'sphconv':
        tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_DWH, batch_size, indices=indice_bzyx)
        tensor.feature.retain_grad()
        conv_class = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
        conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False,
            tile_size=[2,2])
    # conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)

    conv.weight = torch.nn.Parameter(weight.clone())

    y = conv(tensor).dense().sum()
    y.backward()

    # if lib == 'sphconv':
    #     torch.save(tensor.feature.grad, "d_feature_ref.pt")

    return y, feature.grad

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
        tensor_dense = tensor.dense()
        tensor_dense[0,:,0,0,0] *= 2
        tensor_dense[0,:,0,0,1] *= 3
        y = tensor_dense.sum()
        y.backward()
        print("tensor.feature.grad = ", tensor.feature.grad)
        assert((tensor.feature.grad[0,:] == 2).all())
        assert((tensor.feature.grad[1,:] == 3).all())

    def test_init_tensor(self):
        torch.autograd.set_detect_anomaly(True)
        indices_bzyx = torch.tensor([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        features = torch.randn(
            (indices_bzyx.shape[0], 4), dtype=torch.float, device=indices_bzyx.device,
            requires_grad=True)
        tensor = sphconv.SparseConvTensor(features, [2, 2, 2], 1, indices=indices_bzyx)
        # tensor.requires_grad = True
        # tensor.feature.requires_grad = True
        tensor_dense = tensor.dense()
        tensor_dense[0,:,0,0,0] *= 2
        tensor_dense[0,:,0,0,1] *= 3
        y = tensor_dense.sum()
        y.backward()
        print("features.grad = ", features.grad)
        assert((features.grad[1,:] == 2).all())
        assert((features.grad[0,:] == 3).all())

    def test_init_tensor2(self):
        torch.autograd.set_detect_anomaly(True)
        spatial_shape_DWH = [80, 80, 80]
        in_channels = 32
        batch_size = 1
        features, indices_bzyx = batch_real_test_inputs(
            channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)
        sph_features = features.clone()
        sp_features = features.clone()
        sph_features.requires_grad = True
        sp_features.requires_grad = True
        sph_tensor = sphconv.SparseConvTensor(sph_features, spatial_shape_DWH, batch_size, indices=indices_bzyx)
        sp_tensor = spconv.SparseConvTensor(sp_features, indices_bzyx, spatial_shape_DWH, batch_size)
        trans = torch.randn((80, 80)).cuda()
        sph_y = (sph_tensor.dense() * trans).sum()
        sp_y = (sp_tensor.dense() * trans).sum()
        sph_y.backward()
        sp_y.backward()
        # print("features.grad = ", sph_features.grad)
        assert(torch.isclose(sph_y, sp_y).all())

    def test_conv(self):
        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()
        spatial_shape_DWH = [3, 3, 3]
        in_channels = 16
        out_channels = 16
        batch_size = 1
        torch.manual_seed(0)
        features = torch.ones((
            indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)

        kernel_size = [3, 3, 3]
        stride = [2, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.ones((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        # sp_y, sp_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        # sph_y, sph_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")
        # assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        # assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        sp_y, sp_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="spconv")
        sph_y, sph_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="sphconv")

        t_y, t_d_dense = d_torch_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="")

        print("t_d_dense = ", t_d_dense)
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        assert(torch.isclose(t_y, sph_y, rtol=0.01).all())
        print("sp_d_f = ", sp_d_f[:,:])
        print("sph_d_f = ", sph_d_f[:,:])
        print("distance = ", (sp_d_f - sph_d_f).abs().sum())
        assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        # print("spconv_y = ", spconv_y)
        # # print("spconv: d weight = ", sp_conv.weight.grad)
        # print("spconv: d feature = ", spconv_feature.grad[:, 0])

        # sphconv_y = sph_conv(sphconv_tensor).dense().sum()
        # sphconv_y.backward()
        # print("sphconv_y = ", sphconv_y)
        # # print("sphconv: d weight = ", sph_conv.weight.grad)
        # print("sphconv: d feature = ", sphconv_feature.grad[:, 0])
        # assert(sp_conv.weight.grad.sum().isclose(sph_conv.weight.grad.sum()))
        # assert(sp_conv.weight.grad.sum().isclose(sph_conv.weight.grad.sum()))

    def test_conv_d_feature1(self):
        spatial_shape_DWH = [14, 14, 8]
        in_channels = 32
        out_channels = 32
        batch_size = 1

        features, indices_bzyx = batch_real_test_inputs(
            channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.ones((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        sp_y, sp_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="spconv")
        sph_y, sph_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="sphconv")

        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        print("sp_d_f = ", sp_d_f[:,0])
        print("sph_d_f = ", sph_d_f[:,0])
        print("distance = ", (sp_d_f - sph_d_f).abs().sum())
        assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())


    def test_conv_d_feature2(self):
        torch.set_printoptions(edgeitems=200)
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

        spatial_shape_DWH = [14, 14, 8]
        in_channels = 16
        out_channels = 16
        batch_size = 1

        # features, indices_bzyx = batch_real_test_inputs(
        #     channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)
        features = torch.ones((
            indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.ones((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        sp_y, sp_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        sph_y, sph_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")

        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        # print("sp_d_f = ", sp_d_f[:,0])
        # print("sph_d_f = ", sph_d_f[:,0])
        print("distance0 =", (sp_y - sph_y).abs().sum())
        print("distance = ", (sp_d_f - sph_d_f).abs().sum())
        print("distance2 = ", (sp_d_f - sph_d_f).sum())
        print("diff = ", (sp_d_f - sph_d_f)[:,0])
        assert(torch.isclose(sp_d_f[:,0], sph_d_f[:,0], rtol=0.01).all())
        assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())


    def test_unit_df(self):

        d_featureOut = torch.load("d_featureOut.pt")
        print("d_featureOut.device =", d_featureOut.device)
        assert((d_featureOut == 1).all())
        # rule_reverse = torch.load("rule_reverse2.pt")
        rules = torch.load("rules.pt")

        print("rules.sum = ", rules.sum())

        rule_reverse = torch.cat((rules[:,:,1:2,:], rules[:,:,0:1,:]), dim=2).contiguous()

        rule_size = torch.load("rule_size.pt")
        print("rule_size.sum = ", rule_size.sum())
        d_feature_ref = torch.load("d_feature_ref.pt")
        weight = torch.ones((27, 16, 16)).cuda()
        for i in range(1):
            d_feature = rule_conv_d_feature(#d_featureOut,
                torch.ones(129, 16).cuda(),
                weight,
                rule_reverse, rule_size, [7, 7], d_featureOut.size(0))
            d_feature2, _ = rule_conv_backward(
                torch.ones(129, 16).cuda(),
                #    d_featureOut,
                torch.ones(129, 16).cuda(),
                weight,
                rule_reverse, rule_size, [7, 7])
            print("dist0 = ", (d_feature - d_feature_ref).abs().sum())
            print("dist02 = ", (d_feature2 - d_feature_ref).abs().sum())

        print("diff = ", (d_feature - d_feature_ref)[:,0])
        print("dist0 = ", (d_feature - d_feature_ref).abs().sum())
        print("dist1 = ", (d_feature - d_feature_ref).sum())
        assert(torch.isclose(d_feature, d_feature_ref).all())




