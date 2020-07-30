import torch
import sphconv_cuda
from sphconv.functional import ToDenseFunction

class RangeVoxel(object):
    """ Voxels like RangeImage

    feature:
        Tensor of shape [B, T, H, W, C]
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

        return a 3D tensor of shape (batchsize, C, D, H, W)

        """
        return to_dense(self.feature, self.depth, self.thick, self.D, self.feature.device)

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

    @property
    def batch_size(self):
        return self.shape[0]

    @property
    def channel_size(self):
        return self.shape[1]

    @property
    def D(self):
        return self.shape[2]

    @property
    def H(self):
        return self.shape[3]

    @property
    def W(self):
        return self.shape[4]

    @property
    def T(self):
        return self.feature.size(1)

def to_dense(feature, depth, thick, D, device=None):
    """Convert to dense 3D tensor

    return a 3D tensor of shape (B, C, D, H, W)
        where D = max_depth

    """

    return ToDenseFunction.apply(feature, depth, thick, D)

def check_size(shape, feature_shape, depth_shape, thick_shape):
    B, C, D, H, W = shape
    B_f, T_f, H_f, W_f, C_f = feature_shape
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
