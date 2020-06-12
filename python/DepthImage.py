import torch
import random


def generate_test_image(B, D, H, W, C, T):
    """
        B: batchsize
        T: thickness, the max number of non-empty voxels on Depth
        H, W, D: height,width, depth
        C, channel
    """
    feature = torch.randn(B, T, H, W, C)
    indice_map = [[[dict() for x in range(H)] for y in range(W)] for b in range(B)]
    # map from i to spatial indice
    thick = torch.randint(T, (B, H, W))

    for b in range(B):
        for x in range(H):
            for y in range(W):
                thick = thick[b, x, y]
                for i in range(thick):
                    z = random.randint(D)
                    while z not in indice_map[b, x, y][i]:
                        z = random.randint(D)
                    indice_map[b, x, y][i] = z
    return DepthImage(feature, indice_map, thick)

class DepthImage():
    """
    two parts

    feature:
        Tensor of shape [B, D, H, W, C]
        TODO: C's position ?
            B: batchsize
            D: Depth, the number of layers of depth image
              the 'i'th layer's depth is self.depth[x,y] + i
            C: channel
            H,W, height and width, two spatial demension

    depth:
        integer Tensor of shape [B, H, W]
            the quantized depth, non negative, start from 0
    """
    def __init__(self, feature: torch.Tensor, depth):
        B, D, H, W, C = feature.shape
        Bd, Hd, Wd = depth.shape

        assert (B == Bd and H == Hd and W ==
                Wd), "feature and depth dimension not match, feature.shape={}, depth.shape={}".format(feature.shape, depth.shape)
        self.feature = feature
        self.depth = depth


    def dense(self, max_depth: int = 0, device=None):
        """Convert to dense 3D tensor
        return a 3D tensor of shape (batchsize, D, H, W, C)
            where D = max_dpeth if specified
        """
        B, D, H, W, C = self.feature.shape
        print("B, D, H, W, C", B, D, H, W, C)
        if max_depth == 0:
            max_depth = int(torch.max(self.depth)) + D
        print("B, max_depth, H, W, C", B, max_depth, H, W, C)
        buffer = torch.zeros((B, max_depth, H, W, C), device=device)
        depth_idx = self.depth.reshape(B, 1, H, W, 1).expand(B, 1, H, W, C)
        for i in range(D):
            f_i = self.feature[:, i, :, :, :].reshape(B, 1, H, W, C)
            idx_i = depth_idx + i
            buffer.scatter_add_(1, idx_i.long(), f_i)
        return buffer


def test_dense():
    B, D, H, W, C = 2, 2, 3, 3, 1
    feature = torch.randn(B, D, H, W, C)
    depth = torch.randint(2, (B, H, W))
    print(check(DepthImage(feature, depth)))


def check(input: DepthImage):
    B, D, H, W, C = input.feature.shape
    dense = input.dense()
    # print("dense .shape =", dense.shape)
    # print("dense", dense.reshape((2, 4, 3, 3)))
    # print("depth = ", input.depth)
    flag_pass = False
    if abs(dense.sum() - input.feature.sum()) > 1e7:
        return False
    for bi in range(B):
        for di in range(D):
            for x in range(H):
                for y in range(W):
                    for ci in range(C):
                        real_depth = input.depth[bi, x, y] + di
                        if abs(dense[bi, real_depth, x, y, ci] - input.feature[bi, di, x, y, ci]) > 1e7:
                            print("dense[", bi, real_depth, x, y, ci,
                                  "] = ", dense[bi, real_depth, x, y, ci])
                            print(
                                "input.feature[", bi, x, y, ci, "] = ", input.feature[bi, di,  x, y, ci])
                            return False
    return True


test_dense()
