import torch
from typing import List


class SparseConvTensor:

    def __init__(self,
                 features: torch.Tensor,
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
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.H = spatial_shape[2]
        self.W = spatial_shape[1]
        self.D = spatial_shape[0]
        self.C = features.shape[-1]
        self.device = features.device
        assert features.device == indices.device
        self.dtype = features.dtype
        self.idxtype = torch.int64
        self.ptrtype = torch.int64
        B = self.batch_size
        W = self.W
        H = self.H
        D = self.D
        C = self.C

        indices = indices.type(self.idxtype)

        # the number of non zero elements
        NNZ = features.shape[0]

        # all nonzeros are in the 'val'
        # size: NNZ * C
        # the 'val' stores all nonzeros
        # its a pointer points to an array of
        self.val = torch.empty(
            (NNZ, self.C), device=self.device, dtype=self.dtype)

        # the 'z_idx' stores the z indexes of the elements in 'val'
        # if val[k] = A[b,x,y,z,c], then z_idx[k]  = z
        # size: NNZ
        self.z_idx = torch.zeros((NNZ), device=self.device, dtype=self.idxtype)

        # invariant: capacity >= nnz
        self.capacity = NNZ

        # the 'z_ptr' stores the locations in 'val' that start a 'C' vector.
        # if val[k] = A[b,x,y,z,c], then  z_ptr[b,x,y] <= k < z_ptr[b,x,y + 1]
        # equivalent to
        #     Index z_ptr[B][H][W];
        # but B H W is not static constant, we manage it at runtime
        # and we append a guard at z_ptr[B-1][H-1][W]
        #
        # size: B * H * W + 1
        self.z_ptr = None
        #  torch.new_empty(
        #     (B * H * W), device=self.device, dtype=self.ptrtype)

        # size   B H W D
        if grid == None:
            grid = torch.empty((B, H, W, D),
                               device=self.device, dtype=self.idxtype)

        grid.zero_()
        # sphconv_C.init_feature(self.val, self.z_idx, self.z_ptr, features,
        # indices, grid)
        # do it in python?
        b = indices[:, 0]
        x = indices[:, 3]
        y = indices[:, 2]
        z = indices[:, 1]

        # set one on occupied voxels
        grid.index_put_((b, x, y, z), torch.ones(NNZ, dtype=self.idxtype))

        # prefix sum on each D fiber
        # sumgrid = torch.cumsum(grid, dim=3)
        #  now we know the thickness on each fiber.
        #  stored at, {:, :, :, D-1}
        #  z_ptr is the prefix sum on these numbers

        # fiber_size = sumgrid[:, :, :, self.D-1]
        fiber_size = torch.sum(grid, dim=3)

        # self.z_idx ?

        self.z_ptr = torch.cumsum(fiber_size.reshape(-1), dim=0)
        assert self.z_ptr.shape[0] ==  B * H * W
        assert self.z_ptr[-1] == NNZ

        # for each voxel
        print ("NNZ = ", NNZ)
        for i in range(NNZ):
            b = indices[i, 0]
            x = indices[i, 3]
            y = indices[i, 2]
            z = indices[i, 1]
            # zptr shape B H W
            zptr_idx = ((b * H + x) * W + y)
            # zptr_idx =  b * H * W  + x * W + y

            val_pos = self.z_ptr[zptr_idx]
            # fill z_idx
            fiber_pos = fiber_size[b, x, y]
            # fiber_pos -= 1
            self.val[val_pos - fiber_pos] = features[i]
            self.z_idx[val_pos - fiber_pos] = z
            fiber_size[b, x, y] = fiber_pos - 1
            # fill val

        self.NNZ = NNZ

    def dense(self):
        """
        return dense tensor of shape ,B C D W H
        """

        # res = torch.zeros((self.batch_size, self.C, self.D, self.W, self.H))
        res = torch.zeros((self.batch_size, self.H, self.W, self.D, self.C))

        # fill val to res, based on z index
        for b in range(self.batch_size):
            for x in range(self.H):
                for y in range(self.W):
                    zptr_idx = ((b * self.H + x) * self.W + y)
                    start_p = 0 if zptr_idx == 0 else self.z_ptr[zptr_idx - 1]
                    end_p = self.z_ptr[zptr_idx]
                    for z_p in range(start_p, end_p):
                        z = self.z_idx[z_p]
                        res[b, x, y, z] = self.val[z_p]

        return res.permute((0, 4, 3, 2, 1)).contiguous()

        # return torch.randn(
        #     (self.batch_size, self.C, self.D, self.W, self.H))



