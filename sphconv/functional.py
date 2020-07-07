import torch

import sphconv_cuda


class SphConvFunction(torch.autograd.Function):
    """ Applies a 3D convolution on Range Images

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
    def forward(ctx,
                feature,
                depth,
                thick,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups):
        sD, sH, sW = stride
        padD, padH, padW = padding
        dD, dH, dW = dilation
        # print("bias = ", bias)

        if bias is None:
            feature_out, depth_out, thick_out = sphconv_cuda.forward(
                feature,
                depth,
                thick,
                weight,
                # bias,
                sD, sH, sW,
                padD, padH, padW,
                dD, dH, dW,
                groups)
        else:
            raise Exception("bias not immplemented yet")

        variables = [feature, depth, thick,
                     weight, bias, stride, padding, dilation, groups]
        ctx.save_for_backward(*variables)

        ctx.mark_non_differentiable(depth_out, thick_out)

        return feature_out, depth_out, thick_out

    @staticmethod
    def backward(ctx, d_featureOut, d_depthOut, d_thickOut):

        # bias
        feature, depth, thick, \
            weight, stride, padding, dilation, groups = ctx.saved_tensors

        sD, sH, sW = stride
        padD, padH, padW = padding
        dD, dH, dW = dilation

        # d_bias
        d_feature, d_weight = sphconv_cuda.backward(
            feature,
            depth,
            thick,
            d_featureOut,
            # bias,
            weight,
            sD, sH, sW,
            padD, padH, padW,
            dD, dH, dW,
            groups)

        # no bias now
        return d_feature, None, None, d_weight, None, None, None, None, None

