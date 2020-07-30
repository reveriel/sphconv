import torch

import sphconv_cuda

class ConvFunction(torch.autograd.Function):
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
                weight,
                in_rules,
                out_rules,
                num_in,
                # bias,
                stride,
                padding,
                dilation,
                groups,
                D,
                oT,
                subm):

        # if bias is None:
        feature_out, = sphconv_cuda.indice_conv_gemm(
            feature,
            weight,
            # bias,
            in_rules,
            out_rules,
            num_in,
            oT,
            *stride,
            *padding,
            *dilation,
            groups)
        # else:
        #     raise Exception("bias not immplemented yet")


        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.D = D
        ctx.subm = subm

        variables = [feature, weight, in_rules, out_rules, num_in]
        ctx.save_for_backward(*variables)


        return feature_out

    @staticmethod
    def backward(ctx, d_featureOut):

        # bias
        feature, weight, in_rules, out_rules, num_in = ctx.saved_tensors

        # d_bias
        d_feature, d_weight = sphconv_cuda.conv_backward_gemm(
            feature,
            d_featureOut,
            # bias,
            weight,
            in_rules,
            out_rules,
            num_in,
            *ctx.stride,
            *ctx.padding,
            *ctx.dilation,
            ctx.groups,
            ctx.subm)

        # no bias now
        # should match the input of forward
        return d_feature, d_weight, None, None, None, None, None, None, None, None, None, None


class ToDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, depth, thick, D):
        ctx.T = feature.size(2)
        ctx.save_for_backward(depth, thick)
        dense = sphconv_cuda.to_dense(feature, depth, thick, D)
        return dense

    @staticmethod
    def backward(ctx, d_featureOut):
        # print("d_featureOut = ", d_featureOut)
        depth, thick = ctx.saved_tensors
        # print("func depth = ", depth)
        # print("func thick = ", thick)
        d_feature = sphconv_cuda.to_dense_backward(d_featureOut, depth, thick, ctx.T)
        return d_feature, None, None, None
