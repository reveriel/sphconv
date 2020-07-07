import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sphconv
from sphconv.modules import SphModule
from sphconv.functional import SphConvFunction
from sphconv.rangevoxel import RangeVoxel

from .utils import _triple, _calculate_fan_in_and_fan_out_hwio


class Convolution(SphModule):
    """Base class for all convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, subm):
        super(SphModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.conv1x1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.subm = subm

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            # self.bias = None
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


class Conv3d(Convolution):
    """ a 3D convolution Module on Range Images """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, subm=False):
        """
        Args:
        ----
            in_channels (int)
            out_channels (int)
            kernel_size (int or tuple): size of the convolving kernel
            stride (int or tuple, optional): the stride of the cconvolving kernel.  Default: 1
            padding (int or tuple, optional): implicit paddings on both sides
                of the input. Default: 0
            dilation (int or tuple, optional): the spacing between kernel elements.
                Default: 1
            groups (int, optional): split into groups, in_channels shouldd be
                divisible by the number of groups. Default: 1
            bias: (bool, optional): if True, add a bias, Default to True
            see https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

            subm: (bool, optional): if True, compute submanifold sparse convolution,
                see paper, https://github.com/facebookresearch/SparseConvNet

       """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert groups == 1, "groups not supported yet"

        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, subm)

    def forward(self, input: RangeVoxel):

        batch_size, inChannel, iD, iH, iW = input.shape
        # print("forward input shape =", input.shape)
        # print("self inch = ", self.in_channels)

        assert (self.in_channels == inChannel), "input channel does not match \
            Expect: {}, got {}".format(self.in_channels, inChannel)

        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        padD, padH, padW = self.padding
        dD, dH, dW = self.dilation

        oD = math.floor((iD + 2 * padD - dD * (kD - 1) - 1) / sD + 1)
        oH = math.floor((iH + 2 * padH - dH * (kH - 1) - 1) / sH + 1)
        oW = math.floor((iW + 2 * padW - dW * (kW - 1) - 1) / sW + 1)

        new_shape = (batch_size, self.out_channels, oD, oH, oW)

        feature, depth, thick = SphConvFunction.apply(
            input.feature,
            input.depth,
            input.thick,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)
        return RangeVoxel(feature, depth, thick, shape=new_shape)
