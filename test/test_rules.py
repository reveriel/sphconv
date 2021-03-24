
from types import TracebackType

import spconv
import sphconv
from sphconv.sphconv_cuda import get_rules_subm, rule_conv
import torch
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
            voxel_features, spatial_shape, batch_size, indices=indices)

        kernel_size = 2
        stride = 1
        padding = 0
        dilation = 1

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.grid.dim() == 4
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32
        assert tensor.grid.dtype == torch.int32

        zo_idx, zo_ptr, rules, rule_size = get_rules_subm(
            tensor.z_idx,
            tensor.z_ptr,
            tensor.grid,
            batch_size,
            spatial_shape,
            spatial_shape,
            [kernel_size, kernel_size, kernel_size],
            [stride, stride, stride],
            [padding, padding, padding],
            [dilation, dilation, dilation]
        )

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

        print("outids = ", outids)
        print("indice_pairs = ", indice_pairs)
        print("indice_pair_num = ", indice_pair_num)

        print("rules = ", rules)
        print("rule size = ", rule_size)
        assert tensor.grid[0][0][0][0] == 0
        assert tensor.grid[0][0][1][0] == 1
        assert tensor.grid[0][1][0][0] == 2
        assert tensor.grid[0][1][1][0] == 3

        assert torch.all(torch.eq(zo_idx, tensor.z_idx))
        assert torch.all(torch.eq(zo_ptr, tensor.z_ptr))


    def test_rule_conv(self):
        indices = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            # [0, 0, 1, 1],
        ], dtype=torch.int).cuda()
        D = 2
        W = 2
        H = 2
        spatial_shape = [D, W, H]
        inChannel = 4
        batch_size = 1
        voxel_features = torch.ones(
            (indices.shape[0], inChannel), dtype=torch.float, device=indices.device)

        tensor = sphconv.SparseConvTensor(
            voxel_features, spatial_shape, batch_size, indices=indices)

        kernel_size = 2
        stride = 1
        padding = 1
        # I padding must be 1, I think it's spconv's bug
        dilation = 1

        assert tensor.z_idx.dim() == 1
        assert tensor.z_ptr.dim() == 3
        assert tensor.grid.dim() == 4
        assert tensor.z_idx.dtype == torch.int32
        assert tensor.z_ptr.dtype == torch.int32
        assert tensor.grid.dtype == torch.int32

        zo_idx, zo_ptr, rules, rule_size = get_rules_subm(
            tensor.z_idx,
            tensor.z_ptr,
            tensor.grid,
            batch_size,
            spatial_shape,
            spatial_shape,
            [kernel_size, kernel_size, kernel_size],
            [stride, stride, stride],
            [padding, padding, padding],
            [dilation, dilation, dilation]
        )

        outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, kernel_size, stride, padding, dilation,
            out_padding=0, subm=True, transpose=False, grid=None, use_hash=False)

        # convolution
        outChannel = 4
        weight = torch.ones((kernel_size, kernel_size, kernel_size,
                             outChannel, inChannel), dtype=torch.float, device=indices.device)

        # print("indice_pairs dtype = ", indice_pairs.dtype)
        # print("indice_pair_num dtype = ", indice_pair_num.dtype)
        # indice_pairs = indice_pairs.int()
        # indice_pair_num = indice_pair_num.int()
        # print("indice_pairs dtype = ", indice_pairs.dtype)
        # print("indice_pair_num dtype = ", indice_pair_num.dtype)
        out_features = spconv.ops.indice_conv(
            voxel_features, weight, indice_pairs, indice_pair_num, outids.shape[0])

        spconv_dense = spconv.SparseConvTensor(out_features, indices, spatial_shape, batch_size).dense()
        # print("spconv out_features = ", out_features)
        sph_out_features = rule_conv(
            tensor.features, weight.reshape((-1, outChannel, inChannel)),
            rules, rule_size, batch_size, spatial_shape, spatial_shape)

        print("sph_out_features 's type is ", type(sph_out_features))
        sphconv_dense = sphconv.SparseConvTensor(
            sph_out_features, spatial_shape, batch_size, z_ptr=tensor.z_ptr, z_idx=tensor.z_idx).dense(tensor.device)

        # print("sphconv out_features = ", sph_out_features)

        # print("spconv_dense = ", spconv_dense[0,0,:,:,:])
        # print("spconv_dense shape = ", spconv_dense.shape)
        # print("sphconv_dense = ", sphconv_dense[0,0,:,:,:])
        # print("sphconv_dense shape = ", sphconv_dense.shape)

        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

