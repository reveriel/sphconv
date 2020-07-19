from __future__ import division
from __future__ import print_function

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

import unittest

import sphconv
from sphconv import RangeVoxel
from kitti_reader import get_range_voxels, get_voxels
from test_utils import RangeVoxel2SparseTensor
import spconv

torch.manual_seed(42)

# /////////////////////////////////////////////////
# check the correctness of conv, against
#   1. spconv
#   2. torch.conv3d
# use, same input, same weight
# ////////////////////////////////////////////////

configs = {
    'in_channels': 4,
    'out_channels': 4,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'bias': False,
    'subm': False,
}

random.seed(520)

def generate_test_RangeVoxel(N, C, T, D, H, W,
                             feature_option: str = "",
                             depth_option: str = "",
                             thick_option: str = ""
                             ):
    """Generate artificial data for testing.

        feature_option(str,optional):  "range"

        depth_option(str,optional): "random"

    """

    # N, C, T, D, H, W = 1, 1, 1, 4, 4, 4
    if feature_option == "range":
        feature = torch.arange(
            N * C * T * H * W, dtype=torch.float).reshape(N, C, T, H, W)
    else:
        feature = torch.ones((N, C, T, H, W))

    if depth_option == "random":
        # no duplicates
        depth_one_slice = list(range(D))
        depth_all = []
        for i in range(H * W):
            random.shuffle(depth_one_slice)
            depth_all.append(torch.tensor(depth_one_slice, dtype=torch.int32))
        depth = torch.stack(depth_all, dim=1).reshape(1, D, H, W)[:,:T,:,:].expand(N, T, H, W)

    else:
        depth = torch.zeros((N, T, H, W), dtype=torch.int32)

    if thick_option == "random":
        thick = torch.randint(T+1, (N, H, W), dtype=torch.int32)
        while torch.sum(thick) == 0:
            thick = torch.randint(T+1, (N, H, W), dtype=torch.int32)
    else:
        thick = torch.ones((N, H, W), dtype=torch.int32)

    shape = (N, C, D, H, W)
    return RangeVoxel(feature, depth, thick, shape).cuda()


def run(conv_configs, batch_size=1, channel=4):
    """Check sphconv and spconv's results. """

    input_sph = get_range_voxels(0, batch_size=batch_size, channel=channel)
    print("input_sphconv shape =", input_sph.shape)
    # input_sp = get_voxels(0, batch_size=batch_size, channel=channel)
    input_sp = RangeVoxel2SparseTensor(input_sph)
    print("input_spconv shape =", input_sp.spatial_shape)
    print("input_sp's coordinate = ", input_sp.indices)

    conv = sphconv.Conv3d(configs['in_channels'],
                          configs['out_channels'],
                          configs['kernel_size'],
                          stride=configs['stride'],
                          padding=configs['padding'],
                          dilation=configs['dilation'],
                          groups=configs['groups'],
                          bias=configs['bias'],
                          subm=configs['subm']).cuda()

    one_weight = torch.ones((configs['out_channels'], configs['in_channels'],
        configs['kernel_size'], configs['kernel_size'], configs['kernel_size'] ))

    conv.weight = torch.nn.Parameter(one_weight.cuda())

    conv_ref = spconv.SparseConv3d(configs['in_channels'],
                                   configs['out_channels'],
                                   configs['kernel_size'],
                                   stride=configs['stride'],
                                   padding=configs['padding'],
                                   dilation=configs['dilation'],
                                   groups=configs['groups'],
                                   bias=configs['bias']).cuda()

    conv_ref.weight = torch.nn.Parameter(
        torch.ones((configs['kernel_size'], configs['kernel_size'], configs['kernel_size'],
        configs['in_channels'], configs['out_channels']
        )).cuda())

    print("input_sp = ")

    print("===="*20)

    with torch.no_grad():
        res_ref:spconv.SparseConvTensor = conv_ref(input_sp)

    print("conv ref's result = ")
    print(res_ref.dense())
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
    res_dense = res.dense()
    print(res_dense)
    print("conv's result sum = ", torch.sum(res_dense))

    # print(res)
    print("===="*20)

# run(configs, batch_size=1, channel=configs['in_channels'])


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

    def setUp(self):
        pass

    def test1(self):
        rangeV = generate_test_RangeVoxel(1, 1, 1, 4, 4, 4)
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(1, 1, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(torch.ones(1, 1, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(1, 1, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, 1, 1).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        print("conv ref's result = ")
        res_ref_dense = res_ref.dense()
        print(res_ref_dense)
        print("conv ref spatial shape =", res_ref.spatial_shape)
        print("conv ref sum = ", torch.sum(res_ref.dense()))


        print("conv's result = ")
        res_dense = res.dense()
        print(res_dense)
        print("conv's result sum = ", torch.sum(res_dense))

        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))


    def test2(self):
        # add batch size
        rangeV = generate_test_RangeVoxel(16, 1, 1, 4, 4, 4)
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(1, 1, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(torch.ones(1, 1, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(1, 1, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, 1, 1).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        res_ref_dense = res_ref.dense()
        res_dense = res.dense()
        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))

    def test3(self):
        # add channel
        iC = oC = 32
        rangeV = generate_test_RangeVoxel(2, iC, 1, 4, 4, 4)
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(iC, oC, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(torch.ones(oC, iC, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(iC, oC, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, iC, oC).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        res_ref_dense = res_ref.dense()
        res_dense = res.dense()
        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))

    def test4(self):
        # add channel, value is generated by arange
        iC = oC = 32
        rangeV = generate_test_RangeVoxel(2, iC, 1, 4, 4, 4, feature_option="range")
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(iC, oC, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(torch.ones(oC, iC, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(iC, oC, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, iC, oC).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        res_ref_dense = res_ref.dense()
        res_dense = res.dense()
        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))

    def test5(self):
        # add channel, value is generated by arange, weight too
        iC = oC = 32
        rangeV = generate_test_RangeVoxel(2, iC, 1, 4, 4, 4, feature_option="range")
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(iC, oC, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(
            torch.arange(1 * 1 * 3 * 3 * 3, dtype=torch.float)
            .reshape(1, 1, 3, 3, 3)
            .expand(oC, iC, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(iC, oC, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(
            torch.arange(3 * 3 * 3 * 1 * 1, dtype=torch.float)
            .reshape(3, 3, 3, 1, 1)
            .expand(3, 3, 3, iC, oC).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        res_ref_dense = res_ref.dense()
        res_dense = res.dense()
        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))

    def test6(self):
        # add batch size, depth is random
        rangeV = generate_test_RangeVoxel(1, 1, 1, 3, 4, 5, depth_option="random")
        input_spconv = RangeVoxel2SparseTensor(rangeV)

        conv = sphconv.Conv3d(1, 1, 3, padding=0).cuda()
        conv.weight = torch.nn.Parameter(torch.ones(1, 1, 3, 3, 3).cuda())
        conv_ref = spconv.SparseConv3d(1, 1, 3, bias=False).cuda()
        conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, 1, 1).cuda())

        with torch.no_grad():
            res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

        with torch.no_grad():
            res: RangeVoxel = conv(rangeV)

        res_ref_dense = res_ref.dense()
        res_dense = res.dense()

        print("corrs = ", input_spconv.indices)
        self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))

    # def test7(self):
    #     # add batch size, depth is random, thick too
    #     rangeV = generate_test_RangeVoxel(
    #         1, 1, 3, 3, 4, 5, depth_option="random", thick_option="random")
    #     input_spconv = RangeVoxel2SparseTensor(rangeV)

    #     conv = sphconv.Conv3d(1, 1, 3, padding=0).cuda()
    #     conv.weight = torch.nn.Parameter(torch.ones(1, 1, 3, 3, 3).cuda())
    #     conv_ref = spconv.SparseConv3d(1, 1, 3, bias=False).cuda()
    #     conv_ref.weight = torch.nn.Parameter(torch.ones(3, 3, 3, 1, 1).cuda())

    #     with torch.no_grad():
    #         res_ref: spconv.SparseConvTensor = conv_ref(input_spconv)

    #     with torch.no_grad():
    #         res: RangeVoxel = conv(rangeV)

    #     res_ref_dense = res_ref.dense()
    #     res_dense = res.dense()

    #     print("corrs = ", input_spconv.indices)
    #     self.assertTrue(check_equal(res_ref_dense, res_dense, verbose=True))



if __name__ == '__main__':
    unittest.main()
    pass
