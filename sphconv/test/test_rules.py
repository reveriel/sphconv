
from types import TracebackType

import torch
import spconv
import sphconv
import sphconv.sphconv_cuda
from numpy.lib import stride_tricks
from sphconv.utils import voxel_generator


class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_rules(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        D = 2
        W = 2
        H = 2
        spatial_shape = [D, W, H]
        inChannel = 4
        batch_size = 1
        voxel_features = torch.ones((indices.shape[0], inChannel), device=indices.device)
        NNZ = indices.shape[0]


        tensor = sphconv.SparseConvTensor(
            voxel_features, indices, spatial_shape, batch_size)

        kernel_size = 2
        stride = 1
        padding = 0
        dilation = 1
        groups = 1

        zo_idx, zo_ptr, rules, rule_size = sphconv.sphconv_cuda.get_rules_subm(
            tensor.z_idx, tensor.z_ptr,
            tensor.grid,
            kernel_size, kernel_size, kernel_size,
            stride, stride, stride,
            padding, padding, padding,
            dilation, dilation, dilation,
            groups
        )

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

        print("outids = ", outids)
        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        assert torch.full(torch.eq(zo_idx, tensor.z_idx))
        assert torch.full(torch.eq(zo_ptr, tensor.z_ptr))


