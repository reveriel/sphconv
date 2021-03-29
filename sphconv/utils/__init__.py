# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from itertools import product, repeat
from typing import List

import torch
from sphconv.utils.voxel_generator import (VoxelGenerator, VoxelGeneratorV2,
                                           VoxelGeneratorV3)
from torch._six import container_abcs
from torch.nn.init import calculate_gain


def out_spatial(
    in_spatial: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1]
) -> List[int]:
    iH, iW, iD = in_spatial
    kH, kW, kD = kernel_size
    sH, sW, sD = stride
    padH, padW, padD = padding
    dH, dW, dD = dilation

    oH = math.floor((iH + 2 * padH - dH * (kH - 1) - 1) / sH + 1)
    oW = math.floor((iW + 2 * padW - dW * (kW - 1) - 1) / sW + 1)
    oD = math.floor((iD + 2 * padD - dD * (kD - 1) - 1) / sD + 1)

    return [oH, oW, oD]


def _triple(x):
    """If x is a single number, repeat three times."""
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 3))


# copy form pytorch
# this is a private function in pytorch
# customized for my weight
def _calculate_fan_in_and_fan_out(tensor):
    """Init convolution weight. Copied from pytorch."""
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    # if dimensions == 2:  # Linear
    #     fan_in = tensor.size(-2)
    #     fan_out = tensor.size(-1)
    # else:
    num_input_fmaps = tensor.size(-1)
    num_output_fmaps = tensor.size(-2)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[..., 0, 0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                self.__class__.__name__ = str(layer_class.__name__)
                super().__init__(*args, **kw)
        return DefaultArgLayer

    return layer_wrapper
