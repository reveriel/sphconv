# convert conventional point cloud data (in xyz format)
# to RangeVoxel

import numpy as np
import torch

import sphconv
from sphconv import RangeVoxel

# the resolution of the LiDAR is 0.09 dgree for 5Hz. At 10Hz, the resolution is around 0.1728 degree.
# Ideally, W comes out to be 520

class VoxelGenerator(object):
    def __init__(self, v_res, h_res, d_res,
                 v_range, h_range, d_range, log):
        self.v_res = v_res
        self.h_res = h_res
        self.d_res = d_res
        self.v_range = v_range
        self.h_range = h_range
        self.d_range = d_range
        self.log = log

    def generate(self, points):
        return xyz2RangeVoxel(points, self.v_res, self.h_res, self.d_res,
                              self.v_range, self.h_range, self.d_range, self.log)

def xyz2RangeVoxel(points,
                   v_res=64,
                   h_res=512,
                   d_res=512,
                   v_range=(76.6, 103.4),
                   h_range=(-45, 45),
                   d_range=(6, 70.4),
                   log=True,
                   verbose=False,
                   device=None
                   ) -> RangeVoxel:
    """ Convert points(xyz) to RangVoxels.

    The points should be from lidar, and are from one frame.

    Args:
    -----
        pionts (numpy tensor of shape [M, NDim]) :
            M is the number of points, NDim is the number of features,
            asssume the first three are x,y,z coordinates.

        vres (int) : vertical resolution, a 64 line lidar is most suitted by
            v_res=64. Default to 64.
        h_res (int) : horizontal resolution, Default to 512
        d_res (int) : resolution on depth dimension, Default to 512

        v_range (int, int) : vertial range, or the theta range in spherical
            coordinate, in degree.
        h_range (int, int) : horizontal range, or the phi range in spherical
            coordinate, in degree.
        d_range (int, int) : distance range.

        log (bool) : if use log on distance.  Default to True

        verbose (bool) : gives more debug info

    Returns:
    -------
        RangeImage, of spatial shape [D, H, W] = [d_res, v_res, h_res],
            of T = 1,
            of B = 1.

    TODO:

            CUDA accelerate this.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    Channel = points.shape[1]

    x2y2 = x * x + y * y
    r = np.sqrt(x2y2 + z * z)
    thetas = np.arccos(z / r)
    phis = np.arcsin(y / np.sqrt(x2y2))

    delta_phi = np.radians(h_range[1] - h_range[0]) / h_res
    delta_theta = np.radians(v_range[1] - v_range[0]) / v_res

    theta_idx = ((thetas - np.radians(v_range[0])) / delta_theta).astype(int)
    phi_idx = ((phis - np.radians(h_range[0])) / delta_phi).astype(int)

    if log:
        delta_r = (np.log(d_range[1]) - np.log(d_range[0])) / d_res
        depth_idx = (np.log(r) - np.log(d_range[0])) / delta_r
    else:
        delta_r = (d_range[1] - d_range[0]) / d_res
        depth_idx = (r - d_range[0]) / delta_r

    # clamp index
    if verbose:
        # how many points are clampped
        pass

    theta_idx[theta_idx < 0] = 0
    theta_idx[theta_idx >= v_res] = v_res - 1
    phi_idx[phi_idx < 0] = 0
    phi_idx[phi_idx >= h_res] = h_res - 1
    depth_idx[depth_idx < 0] = 0
    depth_idx[depth_idx >= d_res] = d_res - 1

    # in this way, later points of same coordinate
    # overwrites earlier points data.
    # later points seems to be with bigger z, not universaly correct
    feature = torch.zeros((1, 1, v_res, h_res, Channel))
    for i in range(0, Channel):
        feature[0, 0, theta_idx, phi_idx, i] = torch.from_numpy(points[:, i])

    # TODO, what about default depth ?
    # maybe we should ignore them / or use neigbour points' depth?
    # or rand ?
    depth = torch.zeros((1, 1, v_res, h_res), dtype=torch.int32)
    depth[0, 0, theta_idx, phi_idx] = torch.from_numpy(depth_idx.astype(np.int32))

    thick = torch.ones((1, v_res, h_res), dtype=torch.int32)

    return RangeVoxel(feature, depth, thick, shape=(1, Channel, d_res, v_res, h_res))


# batch data

def merge_rangevoxel_batch(voxel_list: [RangeVoxel]) -> RangeVoxel:
    """ Merge a list of RangeVoxel to a batch. Move to GPU.

        must be of the same shape, I don't check it

    """
    feature_list = [ x.feature for x in voxel_list]
    feature = torch.cat(feature_list, dim=0)
    depth_list = [ x.depth for x in voxel_list]
    depth = torch.cat(depth_list, dim=0)
    thick_list = [ x.thick for x in voxel_list]
    thick = torch.cat(thick_list, dim=0)

    shape = list(voxel_list[0].shape)
    shape[0] = len(voxel_list)

    return RangeVoxel(feature, depth, thick, shape=shape)

