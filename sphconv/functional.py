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

        a tensor of shape (N, out_channels, oH, oW), where
            oH = floor((iH + 2 x padH - dH x (kH - 1) -1) / sH + 1)
            oW = floor((iW + 2 x padW - dW x (kW - 1) -1) / sW + 1)

    Examples:

        filters = torch.torch.randn(33, 16, 3, 3)
        depth = torch.randn()

        """

    @staticmethod
    def forward(ctx, feature, depth, thick,
                weight, bias, stride, padding, dilation, groups):
        sD, sH, sW = bias
        padD, padH, padW = padding
        dD, dH, dW = dilation

        outputs = sphconv_cuda.forward(
            feature,
            depth,
            thick,
            weight,
            bias,
            sD, sH, sW,
            padD, padH, padW,
            dD, dH, dW)

        variables = [feature, depth, thick,
                     weight, bias, stride, padding, dilation, groups]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):

        feature, depth, thick, weight, bias, stride, padding, dilation, groups = ctx.saved_tensors

        sD, sH, sW = bias
        padD, padH, padW = padding
        dD, dH, dW = dilation

        feature_bp, weight_bp, bias_bp = sphconv_cuda.backward(
            feature, depth, thick, gradOutput,
            weight, sD, sH, sW, padD, padH, padW, dD, dH, dW, groups)
        return feature_bp, None, None, weight_bp, bias_bp, None, None, None, None
