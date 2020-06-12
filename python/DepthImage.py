import torch
import random
import numpy as np


def generate_test_image(B, D, H, W, C, T):
    """
        B: batchsize
        T: thickness, the max number of non-empty voxels on Depth
        H, W, D: height,width, depth
        C, channel
    """
    assert T <= D, "T must be less then D, got T={}, D={}".format(T, D)
    feature = torch.randn(B, T, H, W, C)
    indice_map = np.ndarray((B, H, W), dtype=dict)
    # map from i to spatial indice
    thick = torch.randint(T, (B, H, W))

    for b in range(B):
        for x in range(H):
            for y in range(W):
                max_t = thick[b, x, y]
                indice_map[b, x, y] = dict()
                for t in range(max_t):
                    z = random.randint(0, D-1)
                    while z in indice_map[b, x, y]:
                        z = random.randint(0, D-1)
                    indice_map[b, x, y][t] = z
    return DepthImage(feature, indice_map, thick, D)


class DepthImage():
    """
    two parts

    feature:
        Tensor of shape [B, T, H, W, C]
        TODO: C's position ?
            T: thickness
            B: batchsize
            C: channel
            H,W, height and width, two spatial demension
            D: Depth, the depth dimesion is stored compactly

    indice_map, a matrix of hashmap of shape [B, H, W]
        stores indice_map for each spatial location
        maps from indice on 'T' to spatial location on 'D'

    thick: tensor of shape [B, H, W]
        the max number of non empty voxels on 'D'
    """

    def __init__(self, feature: torch.Tensor, indice_map, thick, D):
        B, T, H, W, C = feature.shape
        B_i, H_i, W_i = indice_map.shape
        B_t, H_t, W_t = thick.shape
        assert (B == B_i == B_t and H == H_i == H_t and W == W_i == W_t), \
            "dimension not match, feature.shape={}, indice_map.shape={}, thick.shape={}"\
            .format(feature.shape, (B_i, H_i, W_i), thick.shape)

        self.feature = feature
        self.indice_map = indice_map
        self.thick = thick
        self.D = D

    def dense(self, device=None):
        """Convert to dense 3D tensor
        return a 3D tensor of shape (batchsize, D, H, W, C)
            where D = max_dpeth if specified
        """
        return to_dense(self.feature, self.indice_map, self.thick, self.D)


def to_dense(feature, indice_map, thick, D, device=None):
    """Convert to dense 3D tensor
    return a 3D tensor of shape (batchsize, D, H, W, C)
        where D = max_dpeth if specified
    """
    B, T, H, W, C = feature.shape
    buffer = torch.zeros((B, D, H, W, C), device=device)
    for b in range(B):
        for x in range(H):
            for y in range(W):
                max_t = thick[b, x, y]
                for t in range(max_t):
                    z = indice_map[b, x, y][t]
                    buffer[b, z, x, y] = feature[b, t, x, y]
    return buffer


def test_dense():
    B, D, H, W, C, T = 2, 4, 3, 3, 1, 2
    img = generate_test_image(B, D, H, W, C, T)
    print(check(img))


def check(input: DepthImage):
    B, T, H, W, C = input.feature.shape
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
                max_t = input.thick[b, x, y]
                for t in range(max_t):
                    z = input.indice_map[b, x, y][t]
                    if sum(abs(dense[b, z, x, y] - input.feature[b, t, x, y])) > 1e-6:
                        return False
    return True


test_dense()
