from typing import List, Tuple
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch


def dup_with_batch_idx(indices_zyx: torch.Tensor, batch_size: int):
    """ convert zyx data to bzyx"""
    example = {"coordinates": indices_zyx}
    return merge_batch_torch([example]*batch_size)["coordinates"]


def batch_artifical_inputs(
    indices_zyx: torch.Tensor,
    channel: int,
    batch_size: int
):
    """
    create batched inputs from indices_zyx
    return, features, and indices_bzyx
    """
    feature = torch.randn(
        (indices_zyx.shape[0], channel), dtype=torch.float, device=indices_zyx.device)

    one_example = {'voxel': feature, 'coordinates': indices_zyx}
    example = merge_batch_torch([one_example] * batch_size)

    return example['voxel'], example['coordinates']


def batch_real_test_inputs(
    channel: int = 4,
    batch_size: int = 1,
    spatial_shape_DWH: List[int] = [4, 8, 8]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    return feature, indices_bzyx
    """
    TEST_FILE_MAX = 4
    vvfe = VoxelizationVFE(resolution_HWD=spatial_shape_DWH[::-1])

    example_list = []
    for i in range(batch_size):
        voxels, coords = vvfe.generate(
            '{:06d}.bin'.format(i %  TEST_FILE_MAX),  torch.device('cuda:0'))
        example_list.append({'voxels': voxels, 'coordinates': coords})
    example = merge_batch_torch(example_list)

    feature, indices_bzyx = example['voxels'], example['coordinates']
    torch.set_printoptions(edgeitems=200)
    # feature, [NNZ, 4]
    # original channel is 4, we extend it if needed
    assert channel >= 4
    if channel > 4:
        feature = feature.repeat((1, (channel + 3) //4))
        feature = feature[:, :channel]

    # print("indices_bzyx = ", indices_bzyx)

    return feature, indices_bzyx

