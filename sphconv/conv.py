import math
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sphconv.functional import ConvFunction
from sphconv.modules import SphModule
from sphconv.sphconv_cuda import get_rules, get_rules_subm
# from sphconv.rangevoxel import RangeVoxel
from sphconv.tensor import SparseConvTensor
from sphconv.utils import (_calculate_fan_in_and_fan_out, _triple,
                           kaiming_uniform_, out_spatial)

save_rule_size_count = 0

def tile_size_stringfy(tile_size: List[int]) -> str:
    return "t{}_{}".format(tile_size[0], tile_size[1])

def save_rule_size(rule_size: torch.Tensor, tile_size: List[int]):
    global save_rule_size_count
    DIR = "log/"
    filename:str =  "{}_{}.pt".format(tile_size_stringfy(tile_size), save_rule_size_count)
    save_rule_size_count += 1
    torch.save(rule_size, DIR + filename)

def divUp(x, y):
    return  (x + y - 1) // y

def get_tile_grid_shape(out_spatial_shape_DWH: List[int], tile_size: List[int]) -> List[int]:
    oH = out_spatial_shape_DWH[0]
    oW = out_spatial_shape_DWH[1]
    tile_grid_h = divUp(oH, tile_size[0])
    tile_grid_w = divUp(oW, tile_size[1])
    res =  [tile_grid_h, tile_grid_w]
    return res

class Convolution(SphModule):
    """Base class for all convolutions."""

    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size: List[int],
                 stride: List[int], padding: List[int], dilation: List[int],
                 groups: int, tile_size: List[int],
                 bias: bool, subm: bool, name: str,
                 indice_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.conv1x1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.tile_size = tile_size
        self.subm = subm
        self.indice_key = indice_key
        self.name = name

        self.weight = nn.Parameter(
            torch.empty((*kernel_size, in_channels, out_channels)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            # self.bias = None
            self.register_parameter('bias', None)
            # self.bias = torch.empty(0)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
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

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=False, subm=False,
                 tile_size=[2, 2],
                 indice_key=None,
                 name=None, **kwargs):
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
            groups, tile_size, bias, subm, name, indice_key=indice_key, **kwargs)

    def forward(self, input: SparseConvTensor):
        # start_time = time.time()
        batch_size, in_channels = input.B, input.C
        out_channels = self.weight.shape[4]

        assert (self.in_channels == in_channels), "input channel does not match \
            Expect: {}, got {}".format(self.in_channels, in_channels)

        in_spatial_shape_DWH = [input.D, input.W, input.H]

        out_spatial_shape_DWH = in_spatial_shape_DWH
        if not self.subm:
            out_spatial_shape_DWH = out_spatial(
                in_spatial_shape_DWH, self.kernel_size, self.stride, self.padding, self.dilation)

        # print("out spatial shape = ", out_spatial_shape_DWH)

        get_rule_func = get_rules_subm if self.subm else get_rules

        if self.indice_key is not None:
            datas = input.find_rule(self.indice_key)
            if datas is not None:
                oz_idx, oz_ptr, rules, rule_size, tile_grid_shape = datas
            else:
                oz_idx, oz_ptr, rules, rule_size = get_rule_func(
                    input.z_idx, input.z_ptr,
                    batch_size, in_spatial_shape_DWH, out_spatial_shape_DWH,
                    self.kernel_size, self.stride, self.padding, self.dilation,
                    self.tile_size)

                tile_grid_shape = get_tile_grid_shape(out_spatial_shape_DWH, self.tile_size)
                input.rule_cache[self.indice_key] = (oz_idx, oz_ptr, rules, rule_size, tile_grid_shape)
        else:
            oz_idx, oz_ptr, rules, rule_size = get_rule_func(
                input.z_idx, input.z_ptr,
                batch_size, in_spatial_shape_DWH, out_spatial_shape_DWH,
                self.kernel_size, self.stride, self.padding, self.dilation,
                self.tile_size)
            tile_grid_shape = get_tile_grid_shape(out_spatial_shape_DWH, self.tile_size)

        self.rule_size = rule_size

        out_feature = ConvFunction.apply(
            input.feature, self.weight.view(
                (-1, in_channels, out_channels)),
            rules, rule_size, tile_grid_shape, oz_idx.shape[0])

        # torch.cuda.synchronize()
        # print("time: time = {:.3f}\n".format((time.time() - start_time) * 1000
        # ))
        out_tensor = SparseConvTensor(out_feature, out_spatial_shape_DWH,
                                batch_size, z_ptr=oz_ptr, z_idx=oz_idx)
        out_tensor.rule_cache = input.rule_cache
        return out_tensor


# for compatible with spconv
SparseConv3d = Conv3d


class SubMConv3d(Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=False, subm=False, indice_key=None,
                 **kwargs):
        super(SubMConv3d, self).__init__(in_channels,
                                         out_channels, kernel_size, subm=True,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                                         bias=bias, indice_key=indice_key,
                                         **kwargs)
        self.padding = tuple([k//2 for k in self.kernel_size])
