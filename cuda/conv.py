import torch
import torch.nn as nn
import torch.nn.functional as F
from python.DepthImage import DepthImage

from .utils import _triple, _calculate_fan_in_and_fan_out_hwio

import math

import sphconv_cuda


class Test:
    def print(self):
        gemm_cuda.test()


class SphConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, depth, thick,
                weights, bias, stride, padding, dilation, groups)
    sD, sH, sW = bias
    padD, padH, padW = padding
    dD, dH, dW = dilation

    outputs = sphconv_cuda.forward(
        feature,
        depth,
        thick,
        weights,
        bias,
        sD, sH, sW,
        padD, padH, padW,
        dD, dH, dW)

    variables = [feature, depth, thick,
                 weights, bias, stride, paddingg, dilation, groups]
    ctx.save_for_backward(*variables)

    return outputs

    @staticmethod
    def backward(ctx, gradOutput):

        feature, depth, thick, weights, bias, stride, padding, dilation, groups = ctx.saved_tensors

        feature_bp, weights_bp, bias_bp = sphconv_cuda.backward(
            feature, depth, thick, gradOutput,
            weights, sD, sH, sW, padD, padH, padW, dD, dH, dW)
        return feature_bp, None, None, weights_bp, bias_bp, None, None, None, None


class SphConvBase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, subm):
        super(SphConvBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.conv1x1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.subm = subm

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            # self.bias = torch.empty(0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.subm is True:
            s += ', subm={subm}'
        return s.format(**self.__dict__)


def SphConv(SphConvBase):
    """
            Applies a 3D convolution on Depth Images

    Parameters:
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

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, subm=False):

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert groups == 1, "groups not supported yet"

        super(SphConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, subm)

    def forward(self, input: DepthImage):
        return SphConvFunction.apply(
            input.feature,
            input.depth,
            input.thick,
            self.weight,
            self.bias,
            self.stride,
            self.paddinig,
            self.dilation,
            self.groups)
