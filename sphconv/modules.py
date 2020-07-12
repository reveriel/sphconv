from collections import OrderedDict

import spconv
import torch
from torch import nn
import time
import sys

import sphconv


def is_sph_module(module):
    sphconv_modules = (SphModule, )
    return isinstance(module, sphconv_modules)

def is_sparse_conv(module):
    from sphconv.conv import Convolution
    return isinstance(module, Convolution)


def _mean_update(vals, m_vals, t):
    outputs = []
    if not isinstance(vals, list):
        vals = [vals]
    if not isinstance(m_vals, list):
        m_vals = [m_vals]
    for val, m_val in zip(vals, m_vals):
        output = t / float(t + 1) * m_val + 1 / float(t + 1) * val
        outputs.append(output)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


class SphModule(nn.Module):
    """ place holder, all module subclass from this will take sptensor in SparseSequential.
    """
    pass


class Sequential(SphModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = SparseSequential(
                  SparseConv2d(1,20,5),
                  nn.ReLU(),
                  SparseConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = SparseSequential(OrderedDict([
                  ('conv1', SparseConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', SparseConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = SparseSequential(
                  conv1=SparseConv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=SparseConv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if is_sph_module(module):
                assert isinstance(input, sphconv.RangeImage)
                input = module(input)
            else:
                raise Exception(" ?")
        return input