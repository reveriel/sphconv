import spconv
import torch

H = 5
W = 5
D = 5
N = H * W * 1
inChannel = 1
outChannel = 1
kernelSize = 3
feature = torch.ones((N, inChannel))

indice = []
for x in range(H):
    for y in range(W):
        z = 0
        indice.append(torch.tensor([0, x, y, z], dtype=torch.int))

indice = torch.stack(indice, dim=0)
print(indice)

input = spconv.SparseConvTensor(feature.cuda(), indice.cuda(), (H, W, D), 1)

conv = spconv.SparseConv3d(outChannel, inChannel, 3,
                           bias=False, stride=2, padding=0, use_hash=False)

conv.weight = torch.nn.Parameter(torch.ones(3, 3, 3, 1, 1).cuda())

with torch.no_grad():
    res = conv(input)

print(res.dense())

