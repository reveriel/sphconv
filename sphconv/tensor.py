import torch
from typing import List

def init_csf(features: torch.Tensor, indices: torch.Tensor,
             B: int,  # batch size
             H: int, W: int, D: int, C: int,
             NNZ: int,
             grid: torch.Tensor,
             val: torch.Tensor,
             z_idx: torch.Tensor, device) -> torch.Tensor:
    """
    init val, z_idx, and z_ptr from features and indices
    return, z_ptr,
    """

    grid.zero_()

    b = indices[:, 0].long()
    x = indices[:, 3].long()
    y = indices[:, 2].long()
    z = indices[:, 1].long()

    # set one on occupied voxels
    grid.index_put_((b, x, y, z), torch.ones(NNZ, dtype=torch.int32, device=device))

    # prefix sum on each D fiber
    # sumgrid = torch.cumsum(grid, dim=3)
    #  now we know the thickness on each fiber.
    #  stored at, {:, :, :, D-1}
    #  z_ptr is the prefix sum on these numbers

    # fiber_size = sumgrid[:, :, :, self.D-1]
    fiber_size = torch.sum(grid, dim=3)

    z_ptr = torch.cumsum(fiber_size.reshape(-1), dim=0).type(torch.int32)
    assert z_ptr.shape[0] == B * H * W
    assert z_ptr[-1] == NNZ

    for i in range(NNZ):
        b = indices[i, 0]
        x = indices[i, 3]
        y = indices[i, 2]
        z = indices[i, 1]
        # zptr shape B H W
        zptr_idx = ((b * H + x) * W + y)
        # zptr_idx =  b * H * W  + x * W + y

        val_pos = z_ptr[zptr_idx]
        # fill z_idx
        fiber_pos = fiber_size[b, x, y]
        # fiber_pos -= 1
        val[val_pos - fiber_pos] = features[i]
        z_idx[val_pos - fiber_pos] = z
        fiber_size[b, x, y] = fiber_pos - 1
        # fill val
    return z_ptr.reshape(B,H,W)


class SparseConvTensor:
    def __init__(self,
                 voxel_features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape: List[int], batch_size: int,
                 grid: torch.Tensor = None
                 ):
        """

        features: voxel features of shape? [NNZ, C],  NNZ voxels in B batches
        indices: [K, 4]
            cooridnate format [b, z, y, x] or [b, k, j, i]

        spatial_shape: of order [z, y, x], or [k, j, i]
                    or, a        D, W, H
        """
        self.spatial_shape = spatial_shape
        self.B = batch_size
        self.H = spatial_shape[2]
        self.W = spatial_shape[1]
        self.D = spatial_shape[0]
        self.C = voxel_features.shape[-1]
        self.device = voxel_features.device
        assert voxel_features.device == indices.device
        self.dtype = voxel_features.dtype
        self.idxtype = torch.int32
        self.ptrtype = torch.int32

        indices = indices.type(self.idxtype)

        # the number of non zero elements
        self.NNZ = voxel_features.shape[0]
        # all nonzeros are in the 'val'
        # size: NNZ * C
        # the 'val' stores all nonzeros
        # its a pointer points to an array of
        self.val = torch.empty(
            (self.NNZ, self.C), device=self.device, dtype=self.dtype)

        # the 'z_idx' stores the z indexes of the elements in 'val'
        # if val[k] = A[b,x,y,z,c], then z_idx[k]  = z
        # size: NNZ
        self.z_idx = torch.zeros((self.NNZ), device=self.device, dtype=self.idxtype)

        # invariant: capacity >= nnz
        # self.capacity = self.NNZ

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

        # size   B H W D
        if grid == None:
            grid = torch.empty((self.B, self.H, self.W, self.D),
                               device=self.device, dtype=torch.int32)
        self.grid = grid

        self.z_ptr = init_csf(voxel_features, indices, self.B, self.H, self.W,
                              self.D, self.C, self.NNZ, grid, self.val, self.z_idx,
                              self.device)

        self.indice_dict = {}

    @property
    def shape(self):
        return [self.B, self.C, self.D, self.H, self.W]


    def find_indice_pair(self, key:str):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self):
        """
        return dense tensor of shape ,B C D W H
        """

        # res = torch.zeros((self.B, self.C, self.D, self.W, self.H))
        res = torch.zeros((self.B, self.H, self.W, self.D, self.C))
        zptr_flat = self.z_ptr.reshape(-1);

        # TODO, parallel
        # fill val to res, based on z index
        for b in range(self.B):
            for x in range(self.H):
                for y in range(self.W):
                    zptr_idx = ((b * self.H + x) * self.W + y)
                    start_p = 0 if zptr_idx == 0 else zptr_flat[zptr_idx - 1]
                    end_p = zptr_flat[zptr_idx]
                    for z_p in range(start_p, end_p):
                        z = self.z_idx[z_p]
                        res[b, x, y, z] = self.val[z_p]

        return res.permute((0, 4, 3, 2, 1)).contiguous()

        # return torch.randn(
        #     (self.batch_size, self.C, self.D, self.W, self.H))
