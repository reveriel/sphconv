
import sphconv.taichi.backbone as backbone
import sphconv.conv_taichi as conv_taichi
import torch
import numpy as np


points_file = "sphconv/taichi/points0.npy"


convs = []
convs.append(torch.nn.Conv3d(1, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(1, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
weights = [conv.weight for conv in convs]

# out_size = (64, 512, 512, 16)
out_size = (2,2)
output = torch.zeros(size=out_size)

points = torch.from_numpy(np.load(points_file))

# print(weights)

conv_taichi.init(
)

for i in range(50):
    conv_taichi.forward(output, points, tuple(weights))

conv_taichi.profiler_print()

# print(points.size)


# if __name__ == '__main__':
#     print("hello")



