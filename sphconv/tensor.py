from typing import List, Optional

import torch

from sphconv.functional import InitTensorFunction, ToDenseFunction

class SparseTensorBase:
    """ a simple plain old python object"""

    def __init__(self,
                 B: int,
                 D: int, W: int, H: int,
                 C: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 itype: torch.dtype,
                 feature: torch.Tensor,
                 z_idx: torch.Tensor,
                 z_ptr: torch.Tensor):
        self.B = B
        self.D = D
        self.W = W
        self.H = H
        self.C = C
        self.device = device
        self.dtype = dtype
        self.itype = itype
        self.feature = feature
        self.z_idx = z_idx
        self.z_ptr = z_ptr


class SparseConvTensor(SparseTensorBase):
    def __init__(self,
                 raw_feature: torch.Tensor,                   # [NNZ, C]
                 spatial_shape_DWH: List[int],            # [D, W, H] or, zyx
                 batch_size: int,                         # B
                 indices: Optional[torch.Tensor] = None,  # [NNZ, 4], bzyx
                 z_idx: Optional[torch.Tensor] = None,    # [NNZ]
                 z_ptr: Optional[torch.Tensor] = None     # [B, H, W]
                 ):
        """
        if z_idx and z_ptr is not empty, we create an object using them
        otherwise, we create it from 'voxel_features' and 'indices'

        features: voxel features of shape? [NNZ, C],  NNZ voxels in B batches
        indices: [K, 4]
            cooridnate format [b, z, y, x] or [b, k, j, i]

        spatial_shape: of order [z, y, x], or [k, j, i]
                    or, a        D, W, H

        """

        self.rule_cache = {}
        self.requires_grad = False

        if z_idx is not None and z_ptr is not None:
            super().__init__(
                batch_size, spatial_shape_DWH[0], spatial_shape_DWH[1], spatial_shape_DWH[2],
                raw_feature.shape[-1],
                raw_feature.device, raw_feature.dtype,
                z_idx.dtype, raw_feature, z_idx, z_ptr)
            return

        elif indices is not None:
            # we need to reorder the feature
            assert indices.dtype == torch.int32
            assert indices.dim() == 2
            assert indices.shape[1] == 4
            assert raw_feature.dim() == 2
            assert raw_feature.shape[0] == indices.shape[0]
            assert raw_feature.device == indices.device

            super().__init__(
                batch_size, spatial_shape_DWH[0], spatial_shape_DWH[1], spatial_shape_DWH[2],
                raw_feature.shape[-1],
                raw_feature.device, raw_feature.dtype,
                indices.dtype, None, None, None)

            # all nonzeros are in the 'val'
            # size: NNZ * C
            # the 'val' stores all nonzeros
            # its a pointer points to an array of
            self.feature = None

            # the 'z_idx' stores the z indexes of the elements in 'val'
            # if val[k] = A[b,x,y,z,c], then z_idx[k]  = z
            # size: NNZ
            self.z_idx = None

            # the 'z_ptr' stores the locations in 'val' that start a 'C' vector.
            # if val[k] = A[b,x,y,z,c], then  z_ptr[b,x,y] <= k < z_ptr[b,x,y + 1]
            # equivalent to
            #     Index z_ptr[B][H][W];
            #
            # size: B * H * W
            # shape: B, H, W
            self.z_ptr = None

            self.feature, self.z_idx, self.z_ptr = InitTensorFunction.apply(
                raw_feature, indices, self.B, [self.D, self.W, self.H])

    @property
    def shape(self):
        return [self.B, self.C, self.D, self.H, self.W]

    def find_rule(self, key: str):
        if key is None:
            return None
        if key in self.rule_cache:
            return self.rule_cache[key]
        return None

    def dense(self, device=None):
        """
        return dense tensor of shape ,B C D W H
        """
        # res = torch.zeros(
        #     (self.B, self.D, self.W, self.H, self.C), device=device if device else self.device)

        # to_dense(self.feature, self.z_idx, self.z_ptr, self.D, res) # B D W H C

        # return res.permute((0, 4, 1, 2, 3))
        return ToDenseFunction.apply(self.feature, self.z_idx, self.z_ptr, self.shape)
