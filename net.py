
import torch
import torch.nn as nn
import inspect
import spconv
import numpy as np
import sphconv

# simple CNN for benchmarks

# input: voxels,

# output: feature maps?


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


class SingleLayerConv(nn.Module):
    def __init__(self,
                 num_input_features,
                 output_shape,
                 use_norm=True):
        super(SingleLayerConv, self).__init__()
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(
                bias=False, use_hash=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(
                bias=False, use_hash=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = nn.Empty
            BatchNorm1d = nn.Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 16, 3, indice_key="subm0"),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        t = time.time()
        torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        torch.cuda.synchronize()
        print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SingleLayerSphConv(nn.Module):
    def __init__(self,
                 num_input_features,
                 output_shape,
                 ):

        self.middle_conv = spconv.SparseSequential(
            sphconv.Conv3D(num_input_features, 16, 3, indice_key="subm0"),
        )
    def forward(self, *input, **kwargs):
        return super().forward(*input, **kwargs)
