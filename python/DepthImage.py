import torch
import random
import numpy as np


def generate_test_image(B, C, D, H, W, T):
    """
        B: batchsize
        C, channel
        D, H, W : depth, height, width
        T: thickness, the max number of non-empty voxels on Depth
    """
    assert T <= D, "T must be less then D, got T={}, D={}".format(T, D)
    feature = torch.randn(B, C, T, H, W)
    depth = torch.randint(D, (B, T, H, W))
    thick = torch.randint(T, (B, H, W))

    # sort depth
    for b in range(B):
        for i in range(H):
            for j in range(W):
                values, _ = torch.sort(depth[b, :, i, j])
                depth[b, :, i, j] = values

    return DepthImage(feature, depth, thick, D)


class DepthImage():
    """
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
    """

    def __init__(self, feature: torch.Tensor, depth, thick, D):
        B, _, T, H, W = feature.shape
        B_d, T_d, H_d, W_d = depth.shape
        B_t, H_t, W_t = thick.shape
        assert (B == B_d == B_t and H == H_d == H_t and W == W_d == W_t and
                T == T_d), \
            "dimension not match, feature.shape={}, depth.shape={}, thick.shape={}"\
            .format(feature.shape, (B_d, H_d, W_d), thick.shape)

        self.feature = feature
        self.depth = depth
        self.thick = thick
        self.D = D

    def dense(self, device=None):
        """Convert to dense 3D tensor
        return a 3D tensor of shape (batchsize, D, H, W, C)
            where D = max_dpeth if specified
        """
        return to_dense(self.feature, self.depth, self.thick, self.D)


def to_dense(feature, depth, thick, D, device=None):
    """Convert to dense 3D tensor
    return a 3D tensor of shape (B, C, D, H, W)
        where D = max_dpeth if specified
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
    img = generate_test_image(B, D, H, W, C, T)
    print(check(img))


def check(input: DepthImage):
    B, C, T, H, W = input.feature.shape
    D = input.D

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
                    if sum(abs(dense[b,:, z, x, y] - input.feature[b,:, t, x, y])) > 1e-6:
                        print("dense[",b,":", z, x,y,"]=", dense[b,:,z,x,y])
                        print("feature[",b,":", t, x,y,"]=", input.feature[b,:,t,x,y])
                        return False
    return True


test_dense()
