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
        check_size(shape, feature.shape, depth.shape, thick.shape)

        self.feature = feature
        self.depth = depth
        self.thick = thick
        self.shape = shape
        self.indice_dict = {}

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

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

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


def check_size(shape, feature_shape, depth_shape, thick_shape):
    B, C, D, H, W = shape
    B_f, C_f, T_f, H_f, W_f = feature_shape
    B_d, T_d, H_d, W_d = depth_shape
    B_t, H_t, W_t = thick_shape
    assert (B == B_d == B_t == B_f), \
        "batch dimension not match, feature.shape={}, depth.shape={}," \
            "thick.shape={},  shape={}".\
            format(feature_shape, depth_shape, thick_shape, shape)
    assert ( H == H_d == H_t == H_f), \
        "hight dimension not match, feature.shape={}, depth.shape={}," \
            "thick.shape={},  shape={}".\
            format(feature_shape, depth_shape, thick_shape, shape)
    assert (W == W_d == W_t == W_f), \
        "width dimension not match, feature.shape={}, depth.shape={}," \
            "thick.shape={},  shape={}".\
            format(feature_shape, depth_shape, thick_shape, shape)
    assert (C == C_f), \
        "channel dimension not match, feature.shape={}, depth.shape={}," \
            "thick.shape={},  shape={}".\
            format(feature_shape, depth_shape, thick_shape, shape)
    assert (T_d == T_f), \
        "thick dimension not match, feature.shape={}, depth.shape={}," \
            "thick.shape={},  shape={}".\
            format(feature_shape, depth_shape, thick_shape, shape)
