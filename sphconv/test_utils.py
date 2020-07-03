
import numpy as np
import torch
from . import RangeVoxel

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

    return RangeVoxel(feature, depth, thick, (B, C, D, H, W))


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