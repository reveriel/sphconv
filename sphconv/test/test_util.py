# from sphconv.utils import *

# point to raw voxelFeature
# raw voxelFeature processed with VFE
# voxel feature convert to sphconv.Feature and spconv.SparseTensor

# to dense to check eqaulity

# run convolution on both.

from collections import defaultdict

import numpy as np
import spconv
import sphconv
import torch
from sphconv.utils import VoxelGeneratorV3, voxel_generator
from sphconv.vfe import SimpleVoxel

POINTS_FILE = "sphconv/test/000003.bin"

def merge_second_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes'
                ,'points'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        elif key in ['points']:
            continue
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


class VoxelizationVFE():
    """voxeliation + VFE"""

    def __init__(self, resolution=[512, 512, 64]):
        self.resolution = resolution
        self.voxel_generator = VoxelGeneratorV3(
            None,
            [6, -40, -3, 70.4, 40, 1],
            [6, -45, 87.5, 70.4, 45, 100], 2, 24000,
            self.resolution, coord_system="spherical")

        self.vfe = SimpleVoxel(4)
        ## TODO VFE works on CPU, make it on GPU


    def generate(self, file:str) -> torch.Tensor :
        points = np.fromfile(file, dtype=np.float32, count=-1).reshape([-1, 4])
        res = self.voxel_generator.generate(points)

        # coordinate is concatenated with batch_id
        res = merge_second_batch([res])

        raw_voxels, coordinates, num_points_per_voxel, voxel_num = res['voxels'], res[
            'coordinates'], res['num_points_per_voxel'], res['voxel_num']
        # return raw_voxels, coordinates, num_points_per_voxel, voxel_num

        raw_voxels = torch.from_numpy(raw_voxels)
        coordinates = torch.from_numpy(coordinates)
        num_points_per_voxel = torch.from_numpy(num_points_per_voxel)

        voxels = self.vfe(raw_voxels, num_points_per_voxel, coordinates)
        return voxels, coordinates


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

        sphconv_tensor = sphconv.SparseConvTensor(voxels,  coords, vvfe.resolution[::-1], 1)
        sphconv_dense = sphconv_tensor.dense()

        assert spconv_dense.shape == sphconv_dense.shape
        assert torch.all(torch.eq(spconv_dense, sphconv_dense))

    # run convolution on both.




