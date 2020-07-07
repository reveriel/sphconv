import torch

class RangeVoxel(object):
    """ Voxels like RangeImage

    feature:
        Tensor of shape [B, C, T, H, W]
        TODO: C's position ?
            T: thickness
            B: batchsize
            C: channel
            H,W, height and width, two spatial demension
            D: Depth, the depth dimesion is stored compactly

    depth:
        int Tensor of shape [B, T, H, W]

    thick: tensor of shape [B, H, W]
        the max number of non empty voxels on 'D' dimension

    shape: tuple contains (B,C,D,H,W)

    """

    def __init__(self, feature: torch.Tensor, depth, thick, shape):
        """
        """
        B, C, D, H, W = shape
        B_f, C_f, T_f, H_f, W_f = feature.shape
        B_d, T_d, H_d, W_d = depth.shape
        B_t, H_t, W_t = thick.shape
        assert (B == B_d == B_t == B_f
            and H == H_d == H_t == H_f
            and W == W_d == W_t == W_f
            and C == C_f
            and T_d == T_f), \
            "dimension not match, feature.shape={}, depth.shape={}, \
                 thick.shape={},  shape={}".\
                 format(feature.shape, depth.shape, thick.shape, shape)

        self.feature = feature
        self.depth = depth
        self.thick = thick
        self.shape = shape

    def dense(self, device=None):
        """Convert to dense 3D tensor

        return a 3D tensor of shape (batchsize, D, H, W, C)

        """
        return to_dense(self.feature, self.depth, self.thick, self.shape[2])

    def cuda(self):
        """Move to CUDA."""
        self.feature = self.feature.cuda()
        self.depth = self.depth.cuda()
        self.thick = self.thick.cuda()
        return self

def to_dense(feature, depth, thick, D, device=None):
    """Convert to dense 3D tensor

    return a 3D tensor of shape (B, C, D, H, W)
        where D = max_depth

    """
    B, C, T, H, W = feature.shape
    buffer = torch.zeros((B, C, D, H, W), device=device)
    for b in range(B):
        for x in range(H):
            for y in range(W):
                for t in range(thick[b, x, y]):
                    z = depth[b, t, x, y]
                    buffer[b, :, z, x, y] = feature[b, :, t, x, y]
    return buffer

