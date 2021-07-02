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
    weight_: torch.Tensor,
    spatial_shape_DWH: List[int],
    batch_size:int, ic, oc, kernel_size, stride, padding, dilation,
    subm:bool, lib:str):
    """ return y and tensor.grad """

    # torch only support non subm conv
    assert (subm == False)

    feature:torch.Tensor = feature_.clone()
    weight = weight_.clone()

    tensor_0 = sphconv.SparseConvTensor(
        feature, spatial_shape_DWH, batch_size, indices=indice_bzyx).dense()
    tensor_1 = spconv.SparseConvTensor(
        feature, indice_bzyx, spatial_shape_DWH, batch_size).dense()
    assert(torch.isclose(tensor_0, tensor_1).all())
    tensor = tensor_0
    tensor.requires_grad = True
    conv = torch.nn.Conv3d(ic, oc, kernel_size, stride, padding, dilation, bias=False)

    conv.weight = torch.nn.Parameter(weight.permute(4,3,0,1,2).contiguous())

    y = conv(tensor).sum()
    y.backward()

    return y, tensor.grad, conv.weight.grad.permute(2,3,4,1,0)


def d_weight_conv(
    indice_bzyx: torch.Tensor,
    feature_: torch.Tensor,
    weight: torch.Tensor,
    spatial_shape_DWH: List[int],
    batch_size:int, ic, oc, kernel_size, stride, padding, dilation,
    subm:bool, lib:str):

    feature:torch.Tensor = feature_.detach().clone()
    feature.requires_grad = True #! ??, this affects the final y.

    if lib == 'spconv':
        tensor = spconv.SparseConvTensor(
            feature, indice_bzyx, spatial_shape_DWH, batch_size)
        conv_class = spconv.SubMConv3d if subm else spconv.SparseConv3d
        conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)
    elif lib == 'sphconv':
        tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_DWH, batch_size, indices=indice_bzyx)
        conv_class = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
        conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False,
            tile_size=[2,2])
    # conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)

    conv.weight = torch.nn.Parameter(weight.detach().clone())
    conv.weight.requires_grad = True

    y = conv(tensor).dense().sum()
    y.backward()

    # if lib == 'sphconv':
    #     torch.save(tensor.feature.grad, "d_feature_ref.pt")

    return y, conv.weight.grad


def d_feature_conv(
    indice_bzyx: torch.Tensor,
    feature_: torch.Tensor,
    weight: torch.Tensor,
    spatial_shape_DWH: List[int],
    batch_size:int, ic, oc, kernel_size, stride, padding, dilation,
    subm:bool, lib:str):

    feature:torch.Tensor = feature_.detach().clone()
    feature.requires_grad = True

    if lib == 'spconv':
        tensor = spconv.SparseConvTensor(
            feature, indice_bzyx, spatial_shape_DWH, batch_size)
        conv_class = spconv.SubMConv3d if subm else spconv.SparseConv3d
        conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)
    elif lib == 'sphconv':
        tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_DWH, batch_size, indices=indice_bzyx)
        # tensor.feature.retain_grad()
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

    def test_conv_dfeature1(self):
        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()
        spatial_shape_DWH = [4, 4, 4]
        in_channels = 16
        out_channels = 16
        batch_size = 1
        torch.manual_seed(0)
        features = torch.ones((
            indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)

        kernel_size = [2, 2, 1]
        stride = [1, 1, 1]
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

        t_y, t_d_dense, _ = d_torch_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="")

        print("sp, sph, t  output y = ", sp_y, sph_y, t_y)
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

    def test_conv_dfeature2(self):
        spatial_shape_DWH = [20, 20, 20]
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

    def test_conv_dweight1(self):
        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()
        spatial_shape_DWH = [3, 3, 3]
        in_channels = 32
        out_channels = 32
        batch_size = 1
        torch.manual_seed(0)
        features = torch.ones((
            indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
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

        sp_y, sp_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        sph_y, sph_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")

        # t_y, t_d_dense, t_d_w = d_torch_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="")

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        print("sph_d_w = ", sph_d_w[:,:,:,0,0])

        print("sp_d_w ci = ", sp_d_w[0,0,0,:,0])
        print("sph_d_w ci = ", sph_d_w[0,0,0,:,0])

        print("sp_d_w co = ", sp_d_w[0,0,0,0,:])
        print("sph_d_w co = ", sph_d_w[0,0,0,0,:])

        print("distance = ", (sp_d_w - sph_d_w).abs().sum())
        assert(torch.isclose(sp_d_w, sph_d_w, rtol=0.01).all())


    def test_conv_dweight2(self):
        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()

        spatial_shape_DWH = [3, 3, 3]
        in_channels = 32
        out_channels = 32
        batch_size = 1
        torch.manual_seed(0)
        features = torch.randn((
            indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.randn((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        # sp_y, sp_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        # sph_y, sph_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")
        # assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        # assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        sp_y, sp_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        sph_y, sph_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")

        # t_y, t_d_dense, t_d_w = d_torch_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="")

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        print("sph_d_w = ", sph_d_w[:,:,:,0,0])

        print("sp_d_w ci = ", sp_d_w[0,0,0,:,0])
        print("sph_d_w ci = ", sph_d_w[0,0,0,:,0])

        print("sp_d_w co = ", sp_d_w[0,0,0,0,:])
        print("sph_d_w co = ", sph_d_w[0,0,0,0,:])

        print("distance = ", (sp_d_w - sph_d_w).abs().sum())
        assert(torch.isclose(sp_d_w, sph_d_w, rtol=0.01).all())


    def test_conv_dweight3(self):
        in_channels = 32
        out_channels = 32
        batch_size = 1
        spatial_shape_DWH = [5, 8, 5]

        features, indices_bzyx = batch_real_test_inputs(
            channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)

        # features = torch.arange(indices_bzyx.shape[0],
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(in_channels).reshape((indices_bzyx.shape[0], in_channels))
        # features = torch.arange( in_channels,
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(indices_bzyx.shape[0], 1)
        # features = torch.arange( indices_bzyx.shape[0] * in_channels,
        #                   dtype=torch.float, device=indices_bzyx.device).reshape((indices_bzyx.shape[0], in_channels)) / 2
        # features = torch.ones((indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device)
        features = torch.randint(1, 555, (indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device) / 2
        torch.manual_seed(0)

        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.randn((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        # sp_y, sp_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        # sph_y, sph_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")
        # assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        # assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        sp_y, sp_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        sph_y, sph_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")

        # t_y, t_d_dense, t_d_w = d_torch_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="")

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        print("sph_d_w = ", sph_d_w[:,:,:,0,0])

        print("sp_d_w ci = ", sp_d_w[0,0,0,:,0])
        print("sph_d_w ci = ", sph_d_w[0,0,0,:,0])

        print("sp_d_w co = ", sp_d_w[0,0,0,0,:])
        print("sph_d_w co = ", sph_d_w[0,0,0,0,:])

        print("distance = ", (sp_d_w - sph_d_w).abs().sum())
        print("distance0 = ", (sp_d_w - sph_d_w).sum())
        print("distance = ", (sp_d_w - sph_d_w).reshape(-1, in_channels,out_channels)[13] )
        assert(torch.isclose(sp_d_w, sph_d_w, rtol=0.1).all())



    def test_conv_dweight4(self):
        in_channels = 16
        out_channels = 16
        batch_size = 1
        spatial_shape_DWH = [4, 4, 4]

        indices_bzyx = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ], dtype=torch.int).cuda()

        # features, indices_bzyx = batch_real_test_inputs(
        #     channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)

        # features = torch.arange(indices_bzyx.shape[0],
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(in_channels).reshape((indices_bzyx.shape[0], in_channels))
        # features = torch.arange( in_channels,
        #                   dtype=torch.float, device=indices_bzyx.device).repeat(indices_bzyx.shape[0], 1)
        # features = torch.arange( indices_bzyx.shape[0] * in_channels,
        #                   dtype=torch.float, device=indices_bzyx.device).reshape((indices_bzyx.shape[0], in_channels)) / 2
        features = torch.randn((indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device) * 1
        # features = torch.randint(1, 555, (indices_bzyx.shape[0], in_channels), dtype=torch.float, device=indices_bzyx.device) / 2
        torch.manual_seed(0)

        kernel_size = [2, 2, 1]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        dilation = [1, 1, 1]

        # convolution
        weight = torch.randn((*kernel_size, in_channels, out_channels), dtype=torch.float, device=indices_bzyx.device)

        # sp_y, sp_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="spconv")
        # sph_y, sph_d_f = d_feature_conv(
        #     indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
        #     out_channels, kernel_size, stride, padding, dilation, subm=True, lib="sphconv")
        # assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        # assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        sp_y, sp_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="spconv")
        sph_y, sph_d_w = d_weight_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="sphconv")

        t_y, t_d_dense, t_d_w = d_torch_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="")

        print("t_d_w = ", t_d_w[:,:,:,0,0])
        print("sp, sph, t  output y = ", sp_y, sph_y, t_y)
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
        assert(torch.isclose(t_y, sp_y, rtol=0.01).all())
        assert(torch.isclose(t_y, sph_y, rtol=0.01).all())

        print("sp_d_w = ", sp_d_w[:,:,:,0,0])
        print("sph_d_w = ", sph_d_w[:,:,:,0,0])

        print("sp_d_w ci = ", sp_d_w[1,1,0,:,0])
        print("sph_d_w ci = ", sph_d_w[1,1,0,:,0])

        print("sp_d_w co = ", sp_d_w[1,1,0,0,:])
        print("sph_d_w co = ", sph_d_w[1,1,0,0,:])

        print("distance = ", (sp_d_w - sph_d_w).abs().sum())
        print("distance0 = ", (sp_d_w - sph_d_w).sum())
        print("distance[13] = ", (sp_d_w - sph_d_w).reshape(-1, in_channels,out_channels)[:]  )
        print("spdw = ", (sp_d_w).reshape(-1, in_channels,out_channels)[:]  )
        print("sphdw = ", (sph_d_w).reshape(-1, in_channels,out_channels)[:]  )
        assert(torch.isclose(t_d_w, sph_d_w, rtol=0.01).all())


        # print("distance = ", (t_d_w - sp_d_w).abs().sum())
        # print("distance0 = ", (t_d_w - sp_d_w).sum())
        # print("distance = ", (t_d_w - sp_d_w).reshape(-1, in_channels,out_channels)[13] )
        # assert(torch.isclose(t_d_w, sp_d_w, rtol=0.01).all())






