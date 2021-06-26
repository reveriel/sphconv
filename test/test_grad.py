
from typing import List

import spconv
from torch.jit import load
import sphconv
import torch
from sphconv.datagen import merge_batch_torch
from sphconv.utils import out_spatial

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
    elif lib == 'sphconv':
        tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_DWH, batch_size, indices=indice_bzyx)
        conv_class = sphconv.SubMConv3d if subm else sphconv.SparseConv3d
    conv = conv_class(ic, oc, kernel_size, stride, padding, dilation, bias=False)

    conv.weight = torch.nn.Parameter(weight.clone())

    y = conv(tensor).dense().sum()
    y.backward()

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
        assert(torch.isclose(sp_d_f, sph_d_f, rtol=0.01).all())

        sp_y, sp_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="spconv")
        sph_y, sph_d_f = d_feature_conv(
            indices_bzyx, features, weight, spatial_shape_DWH, batch_size, in_channels,
            out_channels, kernel_size, stride, padding, dilation, subm=False, lib="sphconv")
        assert(torch.isclose(sp_y, sph_y, rtol=0.01).all())
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






