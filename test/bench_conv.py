#
# compare with spconv


from typing import List

import spconv
import sphconv
import torch
from sphconv.datagen import VoxelizationVFE, merge_batch_torch
from time import time


def batch_artifical_inputs(
    indices_zyx: torch.Tensor,
    channel: int,
    batch_size: int
):
    """
    create batched inputs from indices_zyx
    """
    features = torch.randn(
        (indices_zyx.shape[0], channel), dtype=torch.float, device=indices_zyx.device)

    one_example = {'voxel': features, 'coordinates': indices_zyx}
    example = merge_batch_torch([one_example] * batch_size)

    return example['voxel'], example['coordinates']

def batch_real_test_inputs(
    channel: int,
    batch_size: int,
    spatial_shape_DWH: List[int]
):
    TEST_FILE_MAX = 4
    vvfe = VoxelizationVFE(resolution_HWD=spatial_shape_DWH[::-1])

    example_list = []
    for i in range(batch_size):
        voxels, coords = vvfe.generate(
            '{:06d}.bin'.format(i %  TEST_FILE_MAX),  torch.device('cuda:0'))
        example_list.append({'voxels': voxels, 'coordinates': coords})
    example = merge_batch_torch(example_list)

    feature, indices = example['voxels'], example['coordinates']
    # feature, [NNZ, 4]
    # original channel is 4, we extend it if needed
    assert channel >= 4;
    if channel > 4:
        feature = feature.repeat((1, (channel + 3) //4))
        feature = feature[:, :channel]
    return feature, indices


def bench_against_spconv(
    loop: int,
    batch_size: int,
    in_channels: int, out_channels: int,
    spatial_shape_DWH: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1, 1],
    subm: bool = False
):
    if subm:
        assert dilation == stride == dilation == [1, 1, 1]

    feature, indices = batch_real_test_inputs(
        channel=in_channels, batch_size=batch_size, spatial_shape_DWH=spatial_shape_DWH)

    # torch.set_printoptions(edgeitems=8000)
    print ("indices shape = ", indices.shape)
    if (indices.shape[0] == 0) :
        print("error")
    spconv_tensor = spconv.SparseConvTensor(
        feature, indices, spatial_shape_DWH, batch_size)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(loop):
        sphconv_tensor = sphconv.SparseConvTensor(
            feature, spatial_shape_DWH, batch_size, indices=indices)
    end.record()
    torch.cuda.synchronize()
    print("init sphconv time = {:0.6f}".format((start.elapsed_time(end)) / loop))

    sph_conv = sphconv.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, subm=subm).cuda()

    Spconv_Conv3d = spconv.SubMConv3d if subm else spconv.SparseConv3d
    sp_conv = Spconv_Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, use_hash=False).cuda()

    # same weight
    weight = torch.randn((*kernel_size, out_channels, in_channels),
                         dtype=torch.float, device=indices.device)

    sph_conv.weight = torch.nn.Parameter(weight.clone())
    sp_conv.weight = torch.nn.Parameter(weight.clone())

    with torch.no_grad():
        x = sp_conv(spconv_tensor)
        start.record()
        for i in range (loop):
            x = sp_conv(spconv_tensor)
            # x = sp_conv(x)
            # x = sp_conv(x)
            spconv_sparse = x
        end.record()
        torch.cuda.synchronize()
        print("spconv conv() time = {:.6f}".format((start.elapsed_time(end)) / loop))

        # torch.cuda.synchronize()
        # start_time = time()
        # for i in range (loop):
        #     spconv_dense = spconv_sparse.dense()
        # torch.cuda.synchronize()
        # end_time = time()
        # print("spconv dense() time = {:.6f}".format((end_time - start_time) / loop))

    with torch.no_grad():
        x = sph_conv(sphconv_tensor)
        start.record()
        for i in range (loop):
            x = sph_conv(sphconv_tensor)
            # x = sph_conv(x)
            # x = sph_conv(x)
            sphconv_sparse = x
        end.record()
        torch.cuda.synchronize()
        print("sphconv conv() time = {:.6f}".format((start.elapsed_time(end)) / loop))

        # torch.cuda.synchronize()
        # start_time = time()
        # for i in range (loop):
        #     sphconv_dense = sphconv_sparse.dense()
        # torch.cuda.synchronize()
        # end_time = time()
        # print("sphconv dense() time = {:.6f}".format((end_time - start_time) / loop))


class TestClass:
    def test_speed(self):

        # bench_against_spconv(
        #     loop=10, batch_size=1, in_channels=32, out_channels=32, spatial_shape_DWH=[20, 256, 256],
        #     kernel_size=[3, 3, 3], stride=[3, 3, 3], padding=[1, 1, 1], subm=False)

        # bench_against_spconv(
        #     loop=10, batch_size=1, in_channels=64, out_channels=64, spatial_shape_DWH=[40, 512, 400],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        # bench_against_spconv(
        #     loop=10, batch_size=1, in_channels=64, out_channels=64, spatial_shape_DWH=[40, 512, 400],
        #     kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 1, 1], subm=False)

        bench_against_spconv(
            loop=10, batch_size=1, in_channels=32, out_channels=32, spatial_shape_DWH=[40, 512, 400],
            kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], subm=False)

        # bench_against_spconv(
        #     loop=10, batch_size=1, in_channels=16, out_channels=32, spatial_shape_DWH=[32, 256, 256],
        #     kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], subm=True)


if __name__ == '__main__':
    t = TestClass()
    t.test_speed()
    print("--------------------------------")

