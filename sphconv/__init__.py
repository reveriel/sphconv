
from sphconv.test_utils import generate_test_image, check


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
        assert (B == B_d == B_t == B_f and H == H_d == H_t == H_f and
                W == W_d == W_t == W_f and C == C_f and T_d == T_f), \
            "dimension not match, feature.shape={}, depth.shape={}, thick.shape={}\n\
            shape={}".format(feature.shape, (B_d, H_d, W_d), thick.shape, shape)

        self.feature = feature
        self.depth = depth
        self.thick = thick
        self.shape = shape

    def dense(self, device=None):
        """Convert to dense 3D tensor

        return a 3D tensor of shape (batchsize, D, H, W, C)
        """
        return to_dense(self.feature, self.depth, self.thick, self.shape[2])


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


def test_dense():
    B, D, H, W, C, T = 2, 4, 3, 3, 1, 2
    img = generate_test_image(B, C, D, H, W, T)
    print(check(img))


def check(input: RangeVoxel):
    B, C, T, H, W = input.feature.shape
    D = input.shape[2]

    dense = input.dense()
    # print("dense .shape =", dense.shape)
    # print("dense", dense.reshape((2, 4, 3, 3)))
    # print("depth = ", input.depth)
    # if abs(dense.sum() - input.feature.sum()) > 1e7:
    #     return False
    for b in range(B):
        for x in range(H):
            for y in range(W):
                for t in range(input.thick[b, x, y]):
                    z = input.depth[b, t, x, y]
                    if sum(abs(dense[b, :, z, x, y] - input.feature[b, :, t, x, y])) > 1e-6:
                        print("dense[", b, ":", z, x, y, "]=",
                              dense[b, :, z, x, y])
                        print("feature[", b, ":", t, x, y, "]=",
                              input.feature[b, :, t, x, y])
                        return False
    return True
