import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sphconv.functional import ConvFunction
from sphconv.modules import SphModule
from sphconv.sphconv_cuda import get_rules_subm
# from sphconv.rangevoxel import RangeVoxel
from sphconv.tensor import SparseConvTensor
from sphconv.utils import _calculate_fan_in_and_fan_out_hwio, _triple


class Convolution(SphModule):
    """Base class for all convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, subm, name,
                 indice_key=None, **kwargs):
        super(SphModule, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.conv1x1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.name = name

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
        # if self.output_padding != (0,) * len(self.output_padding):
            # s += ', output_padding={output_padding}'
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
                 bias=False, subm=False, indice_key=None, name=None, **kwargs):
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
            groups, bias, subm, name, indice_key=indice_key, **kwargs)

    def forward(self, input: SparseConvTensor):
        start_time = time.time();
        batch_size, inChannel, iD, iH, iW = input.shape

        assert (self.in_channels == inChannel), "input channel does not match \
            Expect: {}, got {}".format(self.in_channels, inChannel)

        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        padD, padH, padW = self.padding
        dD, dH, dW = self.dilation

        if self.subm:
            new_shape = (batch_size, self.out_channels, iD, iH, iW)

        else:
            oD = math.floor((iD + 2 * padD - dD * (kD - 1) - 1) / sD + 1)
            oH = math.floor((iH + 2 * padH - dH * (kH - 1) - 1) / sH + 1)
            oW = math.floor((iW + 2 * padW - dW * (kW - 1) - 1) / sW + 1)

            new_shape = (batch_size, self.out_channels, oD, oH, oW)

        datas = input.find_rule(self.indice_key)
        print("========== found in dicts ===========")

        if self.indice_key is not None and datas is not None:
            new_depth, new_thick, in_rules, out_rules, num_in = datas

        else: # not found, compute it
            if self.subm:
                new_depth, new_thick, in_rules, out_rules, num_in = \
                    None, None, None, None, None
                    # sphconv_cuda.get_indice_pairs_subm(
                    #     input.depth, input.thick,
                    #     iD,
                    #     *self.kernel_size,
                    #     *self.stride,
                    #     *self.padding,
                    #     *self.dilation,
                    #     self.groups)

            else:
                new_depth, new_thick, in_rules, out_rules, num_in = \
                    None, None, None, None, None
                    # sphconv_cuda.get_indice_pairs(
                    #     input.depth, input.thick,
                    #     iD,
                    #     *self.kernel_size,
                    #     *self.stride,
                    #     *self.padding,
                    #     *self.dilation,
                    #     self.groups)
                oT = new_depth.size(1)
            #     print("thickness: iT = {}, oT = {}, iFullness = {:.3f}, oFullness = {:.3f}, iEmpty = {}, oEmpty = {},"\
            #           .format(iT, oT,
            #             input.thick.sum().item() / (batch_size * iT * iH * iW),
            #             new_thick.sum().item() / (batch_size * oT * oH * oW),
            #             batch_size * iT * iH * iW - input.thick.sum().item(),
            #             batch_size * oT * oH * oW - new_thick.sum().item(),
            #             # num_in.sum().item() / input.thick.sum().item(),
            #             flush=True
            #             ))
            # print("reuse: ReuseRate = {:.3f}\n".format(
            #             num_in.sum().item() / input.thick.sum().item()))
            # print("inputsize: size = {:.3f}\n".format(input.feature.element_size() * input.feature.nelement()
            #                                         +input.depth.element_size() * input.depth.nelement()
            #                                     +input.thick.element_size() * input.thick.nelement()
            # ) )

            input.indice_dict[self.indice_key] = (new_depth, new_thick,
                in_rules, out_rules, num_in)

        oT = new_depth.size(1)
        # print("oT =", oT);
        feature  = ConvFunction.apply(
            input.feature,
            self.weight,
            in_rules,
            out_rules,
            num_in,
            # self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            iD,
            oT,
            self.subm)


        # torch.cuda.synchronize()
        # print("time: time = {:.3f}\n".format((time.time() - start_time) * 1000
        # ))
        return None
        # return RangeVoxel(feature, new_depth, new_thick, shape=new_shape)

# for compatible with spconv
SparseConv3d = Conv3d


class SubMConv3d(Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, subm=False, indice_key=None,
                 **kwargs):
        super(SubMConv3d, self).__init__(in_channels,
                                         out_channels, kernel_size, subm=True,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                                         bias=bias, indice_key=indice_key,
                                         **kwargs)
