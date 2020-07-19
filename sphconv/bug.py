import spconv
import torch

N = 9
inChannel = 1
outChannel = 1
kernelSize = 3
feature = torch.ones((N, inChannel))

indice = []
for x in range(3):
    for y in range(3):
        z = 0;
        indice.append(torch.tensor([0,x,y,z], dtype=torch.int))

indice = torch.stack(indice, dim=0)
print(indice)

input = spconv.SparseConvTensor(feature.cuda(), indice.cuda(), (3,3,3),1)

conv = spconv.SparseConv3d(outChannel, inChannel, 3, bias=False, use_hash=False)
conv.weight = torch.nn.Parameter(torch.ones(3,3,3, 1, 1).cuda())

with torch.no_grad():
    res = conv(input)

print(res.dense())




