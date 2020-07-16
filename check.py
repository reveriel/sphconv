from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import unittest

import sphconv
from sphconv import RangeVoxel
from kitti_reader import get_range_voxels, get_voxels
import spconv

torch.manual_seed(42)

# /////////////////////////////////////////////////
# check the correctness of conv, against
#   1. spconv
#   2. torch.conv3d
# use, same input, same weight
# ////////////////////////////////////////////////

configs = {
    'in_channels': 64,
    'out_channels': 64,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'bias': False,
    'subm': False,
}


def run(conv_configs, batch_size=1, channel=4):
    """Check sphconv and spconv's results. """

    input_sph = get_range_voxels(0, batch_size=batch_size, channel=channel)
    print("input_sphconv shape =", input_sph.shape)
    input_sp = get_voxels(0, batch_size=batch_size, channel=channel)
    print("input_spconv shape =", input_sp.spatial_shape)

    conv = sphconv.Conv3d(configs['in_channels'],
                          configs['out_channels'],
                          configs['kernel_size'],
                          stride=configs['stride'],
                          padding=configs['padding'],
                          dilation=configs['dilation'],
                          groups=configs['groups'],
                          bias=configs['bias'],
                          subm=configs['subm']).cuda()
    conv_ref = spconv.SparseConv3d(configs['in_channels'],
                                   configs['out_channels'],
                                   configs['kernel_size'],
                                   stride=configs['stride'],
                                   padding=configs['padding'],
                                   dilation=configs['dilation'],
                                   groups=configs['groups'],
                                   bias=configs['bias']).cuda()

    print("input_sp = ")

    print("===="*20)

    with torch.no_grad():
        res_ref:spconv.SparseConvTensor = conv_ref(input_sp)

    print("conv ref's result = ")
    print(res_ref)
    print("conv ref spatial shape =", res_ref.spatial_shape)
    print("conv ref sum = ", torch.sum(res_ref.dense()))
    print("===="*20)
    # exit(0)

    # ==================================
    #
    # ===================================

    print("input sph = ")
    print(input_sph)
    print("===="*20)

    with torch.no_grad():
        res:RangeVoxel = conv(input_sph)

    print("conv's result = ")
    print("conv's result sum = ", torch.sum(res.dense()))

    print(res)
    print("===="*20)

run(configs, batch_size=1, channel=configs['in_channels'])


def check_equal(first, second, verbose) -> bool:
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x))
            print("y = {}".format(y))
            print('-' * 80)
        assert (x.shape == y.shape), "shape no same, Index: {}, x shape = {}, y shape = {}".format(
            i, x.shape, y.shape)
        try:
            np.testing.assert_allclose(
                x, y, err_msg="Index: {}".format(i), rtol=1e-6, atol=1e-6)
            return True
        except Exception as e:
            print(e)
            return False


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(variables, with_cuda, verbose):

    # python impl
    feature, depth, weight, bias, stride, padding, dilation, groups = variables
    dense_output = RangeVoxel(*python.conv.sphconv3d(*variables)).dense()

    # conv3d as ref
    dense_input = RangeVoxel(feature, depth).dense()
    print("dense_input.shape", dense_input.shape)
    ref_output = F.conv3d(dense_input.permute(0, 4, 1, 2, 3).contiguous(), weight, bias=bias, stride=stride,
                          padding=padding, dilation=dilation, groups=groups)
    print("ref_output.shape = ", ref_output.shape)
    # print(ref_output.reshape(4,3,3))

    # baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    # cpp_values = cpp.lltm.LLTMFunction.apply(*variables)

    print('Forward: Baseline (Python) vs. C++ ... ', end='')
    return check_equal([dense_output.permute(0, 4, 1, 2, 3)], [ref_output], verbose)

    # if with_cuda:
    #     cuda_values = cuda.lltm.LLTMFunction.apply(*variables)
    #     print('Forward: Baseline (Python) vs. CUDA ... ', end='')
    #     check_equal(baseline_values, cuda_values, verbose)
    #     print('Ok')


def check_backward(variables, with_cuda, verbose):
    baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    (baseline_values[0] + baseline_values[1]).sum().backward()
    grad_baseline = get_grads(variables)

    zero_grad(variables)

    cpp_values = cpp.lltm.LLTMFunction.apply(*variables)
    (cpp_values[0] + cpp_values[1]).sum().backward()
    grad_cpp = get_grads(variables)

    print('Backward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(grad_baseline, grad_cpp, verbose)
    print('Ok')

    if with_cuda:
        zero_grad(variables)
        cuda_values = cuda.lltm.LLTMFunction.apply(*variables)
        (cuda_values[0] + cuda_values[1]).sum().backward()
        grad_cuda = get_grads(variables)

        print('Backward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(grad_baseline, grad_cuda, verbose)
        print('Ok')


parser = argparse.ArgumentParser()
# parser.add_argument('direction', choices=['forward', 'backward'], nargs='+')
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('unittest_args', nargs='*')
options = parser.parse_args()
sys.argv[1:] = options.unittest_args


B, iD, iH, iW, iC = options.batch_size, 3, 3, 3, 1
oC, kH, kW, kD = 1, 3, 3, 3

# if 'backward' in options.direction:
# check_backward(variables, options.cuda, options.verbose)


class TestForward(unittest.TestCase):

    # if options.cuda:
    #     import cuda.lltm
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    def setUp(self):

        if options.cuda:
            import cuda.lltm
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.kwargs = {'dtype': torch.float32,
                       'device': device,
                       'requires_grad': True}

    def test_one_one(self):
        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.ones(B, iD, iH, iW, iC, **self.kwargs)
        weight = torch.ones(oC, iC, kD, kH, kW, **self.kwargs)
        depth = torch.randint(1, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        self.assertTrue(check_forward(
            variables, options.cuda, options.verbose))

    def test_one_range(self):

        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.ones(B, iD, iH, iW, iC, **self.kwargs)
        weight = torch.arange(
            oC*iC*kD*kH*kW, **self.kwargs).reshape(oC, iC, kD, kH, kW)
        depth = torch.randint(1, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        self.assertTrue(check_forward(
            variables, options.cuda, options.verbose))

    def test_range_one(self):
        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.arange(
            B*iD*iH*iW*iC, **self.kwargs).reshape(B, iD, iH, iW, iC)
        weight = torch.ones(oC, iC, kD, kH, kW, **self.kwargs)
        depth = torch.randint(1, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        check_forward(variables, options.cuda, options.verbose)

    def test_range_range(self):
        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.arange(
            B*iD*iH*iW*iC, **self.kwargs).reshape(B, iD, iH, iW, iC)
        weight = torch.arange(
            oC*iC*kD*kH*kW, **self.kwargs).reshape(oC, iC, kD, kH, kW)
        depth = torch.randint(1, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        self.assertTrue(check_forward(
            variables, options.cuda, options.verbose))

    def test_rand_rand(self):

        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.randn(B, iD, iH, iW, iC, **
                              self.kwargs).reshape(B, iD, iH, iW, iC)
        weight = torch.randn(oC, iC, kD, kH, kW, **
                             self.kwargs).reshape(oC, iC, kD, kH, kW)
        depth = torch.randint(1, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        self.assertTrue(check_forward(
            variables, options.cuda, options.verbose))

    def test_one_one_depth(self):
        B, iD, iH, iW, iC = 1, 3, 3, 3, 1
        oC, kH, kW, kD = 1, 3, 3, 3

        feature = torch.ones(B, iD, iH, iW, iC, **self.kwargs)
        weight = torch.ones(oC, iC, kD, kH, kW, **self.kwargs)
        depth = torch.randint(3, (B, iH, iW), **self.kwargs)
        variables = [feature, depth, weight, None,
                     (1, 1, 1), (1, 1, 1), (1, 1, 1), 1]
        self.assertTrue(check_forward(
            variables, options.cuda, options.verbose))


if __name__ == '__main__':
    # unittest.main()
    pass
