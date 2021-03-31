from typing import List, Optional

import torch

from sphconv.sphconv_cuda import init_tensor, to_dense


class SparseTensorBase:
    """ a simple plain old python object"""

    def __init__(self,
                 B: int,
                 H: int, W: int, D: int,
                 C: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 itype: torch.dtype,
                 feature: torch.Tensor,
                 z_idx: torch.Tensor,
                 z_ptr: torch.Tensor):
        self.B = B
        self.H = H
        self.W = W
        self.D = D
        self.C = C
        self.device = device
        self.dtype = dtype
        self.itype = itype
        self.feature = feature
        self.z_idx = z_idx
        self.z_ptr = z_ptr


class SparseConvTensor(SparseTensorBase):
    def __init__(self,
                 feature: torch.Tensor,                   # [NNZ, C]
                 spatial_shape_DWH: List[int],            # [D, W, H]
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

        if z_idx is not None and z_ptr is not None:
            super().__init__(
                batch_size, spatial_shape_DWH[2], spatial_shape_DWH[1], spatial_shape_DWH[0],
                feature.shape[-1],
                feature.device, feature.dtype,
                z_idx.dtype, feature, z_idx, z_ptr)
            return

        elif indices is not None:
            # we need to reorder the feature
            assert indices.dtype == torch.int32
            assert indices.dim() == 2
            assert indices.shape[1] == 4
            assert feature.dim() == 2
            assert feature.shape[0] == indices.shape[0]
            assert feature.device == indices.device

            super().__init__(
                batch_size, spatial_shape_DWH[2], spatial_shape_DWH[1], spatial_shape_DWH[0],
                feature.shape[-1],
                feature.device, feature.dtype,
                indices.dtype, None, None, None)

            # the number of non zero elements
            NNZ = feature.shape[0]

            # all nonzeros are in the 'val'
            # size: NNZ * C
            # the 'val' stores all nonzeros
            # its a pointer points to an array of
            self.feature = torch.empty(
                (NNZ, self.C), device=self.device, dtype=self.dtype)

            # the 'z_idx' stores the z indexes of the elements in 'val'
            # if val[k] = A[b,x,y,z,c], then z_idx[k]  = z
            # size: NNZ
            self.z_idx = torch.zeros(
                (NNZ), device=self.device, dtype=self.itype)

            # the 'z_ptr' stores the locations in 'val' that start a 'C' vector.
            # if val[k] = A[b,x,y,z,c], then  z_ptr[b,x,y] <= k < z_ptr[b,x,y + 1]
            # equivalent to
            #     Index z_ptr[B][H][W];
            #
            # size: B * H * W
            # shape: B, H, W
            self.z_ptr = None
            #  torch.new_emp
            #     (B * H * W), device=self.device, dtype=self.ptrtype)

            # size   B H W D, for counting in get_rules() and init_csf()

            self.z_ptr = init_tensor(
                feature, indices, self.B, [self.H, self.W, self.D], self.feature, self.z_idx)

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
        res = torch.zeros(
            (self.B, self.H, self.W, self.D, self.C), device=device if device else self.device)

        to_dense(self.feature, self.z_idx, self.z_ptr, self.D, res)

        return res.permute((0, 4, 3, 2, 1))
