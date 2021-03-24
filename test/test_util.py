# from sphconv.utils import *

# point to raw voxelFeature
# raw voxelFeature processed with VFE
# voxel feature convert to sphconv.Feature and spconv.SparseTensor

# to dense to check eqaulity

# run convolution on both.

import spconv
import sphconv
import torch
from datagen import VoxelizationVFE

POINTS_FILE = "000003.bin"

class TestClass:
    # voxel feature convert to sphconv.Feature and spconv.SparseTensor
    def test_todense(self):
        vvfe = VoxelizationVFE(resolution=[512,511,64])
        voxels, coords = vvfe.generate(POINTS_FILE)
        assert voxels.shape[1] == 4
        assert coords.shape[1] == 4
        spconv_tensor = spconv.SparseConvTensor(voxels, coords, vvfe.resolution[::-1], 1)
        spconv_dense = spconv_tensor.dense() # torch
        # N C D W H
        assert spconv_dense.shape[0] == 1
        assert spconv_dense.shape[1] == 4
        assert spconv_dense.shape[2] == vvfe.resolution[2]
        assert spconv_dense.shape[3] == vvfe.resolution[1]
        assert spconv_dense.shape[4] == vvfe.resolution[0]

        sphconv_tensor = sphconv.SparseConvTensor(voxels, vvfe.resolution[::-1], 1,indices=coords )
        sphconv_dense = sphconv_tensor.dense()

        assert spconv_dense.shape == sphconv_dense.shape
        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

    # run convolution on both.




