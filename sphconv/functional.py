from sphconv.utils import out_spatial
from typing import List
from torch.autograd.function import NestedIOFunction

import torch

from sphconv.sphconv_cuda import (rule_conv, rule_conv_backward,
                                  to_dense, to_dense_backward,
                                  init_tensor, init_tensor_backward)


class ConvFunction(torch.autograd.Function):
    """ Applies a 3D convolution on sparse 3d tensor

    Args:

        input: the input tensor of shape (minibatch, in_channels, iD, iH, iW)
        depth: depth tensor of shape (minibatch, iH, iW), of type int
        weight: a 3d filter of shape
                    (out_channels, in_channels/groups, kD, kH, kW)
        bias: optional bias tensor of shape (out_channels). Default: None
        stride: the stride of the cconvolving kernel, can be a single number or
            a tuple (sD, sH, sW). Default: 1
        padding: implicit paddings on both sides of the input. Can be a single
            number or a tuple (padH, padW). Default: 0
        dilation: the spacing between kernel elements. Can be a snigle number
            or a tuple (dD, dH, dW). Default: 1
        groups: split into groups, in_channels shouldd be divisible by the
            number of groups. Default: 1
        see https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    Returns:

        feature: a tensor of shape (N, out_channels, oT, oH, oW)
        depth: tensor of shape (N, oT, oH, oW)
        thick: tensor of shape (N, oH, oW), where
            oT = ?
            oH = floor((iH + 2 x padH - dH x (kH - 1) -1) / sH + 1)
            oW = floor((iW + 2 x padW - dW x (kW - 1) -1) / sW + 1)

    Examples:

        filters = torch.torch.randn(33, 16, 3, 3)
        depth = torch.randn()

        """

    @staticmethod
    def forward(ctx: NestedIOFunction,
                feature: torch.Tensor,
                weight: torch.Tensor,  # [KKK, iC, oC]
                rules: torch.Tensor,  # [NTile, kernelVolume, 2, NNZ]
                rule_size: torch.Tensor,
                batch_size: int,
                spatial_shape_HWD: List[int],
                out_spatial_shape_HWD: List[int],
                outNNZ: int):
        ctx.save_for_backward(feature, weight, rules, rule_size)
        ctx.batch_size = batch_size
        ctx.spatial_shape_HWD = spatial_shape_HWD
        ctx.out_spatial_shape_HWD = out_spatial_shape_HWD
        return rule_conv(feature, weight, rules, rule_size,
                         batch_size, spatial_shape_HWD,
                         out_spatial_shape_HWD, outNNZ)

    @staticmethod
    def backward(ctx: NestedIOFunction,
                 d_featureOut: torch.Tensor):  # [NNC, oC]
        # print("d_featureOut.shape = ", d_featureOut.shape)
        feature, weight, rules, rule_size = ctx.saved_tensors

        # d_bias
        # TODO: split rules
        rule_reverse = torch.cat((rules[:,:,1:2,:], rules[:,:,0:1,:]), dim=2).contiguous()
        # print("rule_reverse shape = ", rule_reverse.shape)
        # .contiguous()
        d_feature, d_weight = rule_conv_backward(
            d_featureOut, feature,  # bias,
            weight.permute(0, 2, 1).contiguous(),
            rule_reverse, rule_size,
            ctx.batch_size, ctx.spatial_shape_HWD, ctx.out_spatial_shape_HWD)

        # TODO: no bias now
        # should match the input of forward
        return d_feature, d_weight, None, None, None, None, None, None


class ToDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: NestedIOFunction,
                feature: torch.Tensor,
                z_idx: torch.Tensor,
                z_ptr: torch.Tensor,
                shape: List[int]):        # B C D H W
        ctx.save_for_backward(z_idx, z_ptr)
        # sphconv_cuda.to_dense(feature, depth, thick, D)
        B, C, D, H, W = shape
        res = torch.zeros((B, D, W, H, C), device=feature.device, dtype=feature.dtype)
        to_dense(feature, z_idx, z_ptr, D, res)
        # from B D W H C
        # to B C D W H
        return res.permute((0, 4, 1, 2, 3)).contiguous()

    @staticmethod
    def backward(ctx: NestedIOFunction,
                 d_featureOut):            # B C D W H
        # print("d_featureOut = ", d_featureOut)
        z_idx, z_ptr = ctx.saved_tensors
        # BCDWH to BDWHC
        d_feature = to_dense_backward(
            d_featureOut.permute((0, 2, 3, 4, 1)).contiguous(),
            z_idx, z_ptr)
        return d_feature, None, None, None


class InitTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: NestedIOFunction,
                raw_feature: torch.Tensor,  # [NNZ, C]
                indices_bzyx: torch.Tensor, # [NNZ, 4]
                batchsize: int,
                spatial_shape_HWD: List[int]):
        feature_out, zidx, zptr, fiber_size = init_tensor(
            raw_feature, indices_bzyx, batchsize, spatial_shape_HWD)
        ctx.save_for_backward(zptr, fiber_size, indices_bzyx)
        ctx.mark_non_differentiable(zidx, zptr)
        return feature_out, zidx, zptr

    @staticmethod
    def backward(ctx: NestedIOFunction,
                 d_featureOut: torch.Tensor,  # B C D W H
                 d_zidx: torch.Tensor,
                 d_zptr: torch.Tensor):
        # import pdb; pdb.set_trace()
        zptr, fiber_size, indices_bzyx = ctx.saved_tensors

        d_feature = init_tensor_backward(
            d_featureOut, zptr, fiber_size, indices_bzyx)
        return d_feature, None, None, None
