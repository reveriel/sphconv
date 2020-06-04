import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DepthImage import DepthImage


from torch._six import container_abcs
from itertools import repeat

torch.manual_seed(42)

def _triple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 3))

def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


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
    """ Applies a 3D convolution on DepthImages

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
        """r"""
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

        assert groups == (1,1,1), "groups not supported yet"

        super(SphConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, subm)

    def forward(self, input:DepthImage):

        # out_tensor = depconv3d(features, depth, self.weight,
        #                        self.bias, self.stride, self.padding, self.dilation,
        #                        self.groups)

        feature = input.feature
        depth = input.depth

        return DepthImage(feature, depth)


