import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from python.DepthImage import DepthImage


from torch._six import container_abcs
from itertools import repeat, product

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

        # out_tensor = depconv3d(features, depth, self.weight,
        #                        self.bias, self.stride, self.padding, self.dilation,
        #                        self.groups)

        feature = input.feature
        depth = input.depth
        feature, depth = sphconv3d(feature, depth, self.weight, self.bias,
                                   self.stride, self.paddinig, self.dilation, self.groups)

        return DepthImage(feature, depth)


def sphconv3d(feature, depth, weight, bias, stride, padding, dilation, groups):
    """
    feature: [B, Thick, W, H, iC]
    depth : [B, W, H]
    weight : [oC, iC, kD, kH, kW]
    """

    B, T, iH, iW, iC = feature.shape
    oC, iC, kD, kH, kW = weight.shape
    sD, sH, sW = stride
    padD, padH, padW = padding
    dD, dH, dW = dilation
    oH = int(math.floor((iH + 2 * padH - dH * (kH - 1) - 1) / sH + 1))
    oW = int(math.floor((iW + 2 * padW - dW * (kW - 1) - 1) / sW + 1))
    iD = T # TODO
    oD = iD + 2 * padD - (kD-1)
    oD = int(math.floor((iD + 2 * padD - dD * (kD - 1) - 1) / sD + 1))

    # mask: which needs to be gathered
    # diff

    # stride , subm
    # stride complicate the input- kernel coresponding, kernel-ouput
    # subm, too

    # gatter,
    # mask, which to gather ??
    #  we don't need mask to gather, all values are usefull
    # near-by differ less than _
    # and stride .
    # mask of shape [B, H, W]
    #

    # depth = depth.unsqueeze(1)  # B,1,H,W
    # unfolded_depth = F.unfold(depth.float(), kernel_size=(kH, kW), dilation=dilation[1:],
    #                           padding=padding[1:], stride=stride[1:])
    # # B , kH * kW * 1 , N
    # depth_diff = (unfolded_depth -
    #               unfolded_depth[:, kH * kW // 2, :].unsqueeze(1))
    # mask = torch.abs(depth_diff) < kD  # B , kH * kW * 1 , N
    # print("mask shape = ", mask.shape)
    # mask = F.fold(mask.float(), output_size=(oH, oW), kernel_size=(kH, kW), dilation=dilation[1:],
    #               padding=padding[1:], stride=stride[1:])
    # print("mask.shape",  mask.shape)
    # TODO: stride
    # usefull_feature = torch.masked_select(feature, mask.bool())
    # usefull_feature = usefull_feature.reshape(B, iD, -1, iC)

    # print(usefull_feature.shape)
    feature = feature.reshape(B, T * iH * iW,  iC)
    weight = weight.permute(1, 0, 2, 3, 4).contiguous().reshape(iC, oC*kD*kH*kW)
    res = torch.matmul(feature, weight)  # B, iD*iH * iW , oC * kD*kH *kW
    res = res.reshape(B, T, iH, iW, oC, kD, kH, kW)
    print("res shape", res.shape)
    # then gather add
    of = torch.zeros(B, oD+2*(kD-padD), oH+2*(kH-padH), oW+2*(kW-padW), oC)
    ## TODO: +2
    # I want to express,
    #  element at (x,y,z, i,j,k) in res
    #                                   (x, y, i,j,k)
    # goes to (x-i,y-j,z-k)
    #                       (x -i, y -j, D(x,y)-k )
    # , two choices, addd paddding early or not
    for i, j, k in product(range(kD), range(kH), range(kW)):
        elem = res[:, :, :, :, :, kD-i-1, kH-j-1, kW-k-1]  # B,iD,iH,iW,oC
        # print("elem shape", elem.shape)
        # print("of shape", of.shape)

        # of,  B, iD, oH, oW, oC
        # first, same dimension
        of[:, i+padD: -(kD-i), j+padH: -(kH-j), k+padW: -(kW-k), :] += elem

    print("of.shape =", of.shape)

    # print(of.reshape(7, 7, 7)[:])

    # return of[:,1-padD:-(1-padD+1), kW//2: -(kW//2+1), kH//2:-(kH//2+1),: ], depth
    of = of[:, kD//2+1:-(kD//2+1), kW//2+1: -(kW//2+1), kH//2+1:-(kH//2+1), :]
    # of = torch.flip(of, [1,2,3])
    return of , depth
    # return of[:, 2: -2, 2:-2, 2:-2, :], depth


def test():
    B, iD, iH, iW, iC = 2, 5, 8, 8, 3
    oC, kH, kW, kD = 2, 3, 3, 3
    feature = torch.randn(B, iD, iH, iW, iC)
    depth = torch.randint(8, (B, iH, iW))
    weight = torch.randn(oC, iC, kD, kH, kW)
    sphconv3d(feature, depth, weight, None,
              _triple(1), _triple(1), _triple(1), _triple(1))

# test()
