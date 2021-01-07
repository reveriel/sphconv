
import sphconv.taichi.backbone as backbone
import sphconv.conv_taichi as conv_taichi
import torch
import numpy as np


points_file = "sphconv/taichi/points0.npy"


convs = []
convs.append(torch.nn.Conv3d(4 ,16, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(16, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(16, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(32, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
convs.append(torch.nn.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)))

weights = [conv.weight for conv in convs]

out_size = (128, 128, 3, 64)
# out_size = (2,2)
output = torch.zeros(size=out_size)

points = torch.from_numpy(np.load(points_file))
print(points.shape)


# https://github.com/pytorch/pytorch/issues/36568

# class TensorContainer(torch.nn.Module):
#     def __init__(self, tensor_dict):
#         super().__init__()
#         for key,value in tensor_dict.items():
#             setattr(self, key, value)

# tensor_dict = {'points' : points}
# tensors = TensorContainer(tensor_dict)
# tensors = torch.jit.script(tensors)
# tensors.save('points0.pt')

# print(weights)

conv_taichi.init()

for i in range(50):
    conv_taichi.forward(output, points, tuple(weights))

print(output)

conv_taichi.profiler_print()

# print(points.size)


# if __name__ == '__main__':
#     print("hello")



