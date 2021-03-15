import torch

class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        """
        features: voxel features of shape? [K, C],  K voxels in B batches
        indices: [K, 4]
        spatial_shape: of order [z, y, x], or [k, j, i]
                    or, a        D, W, H
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.C = features.shape[-1]
        self.H = spatial_shape[2]
        self.W = spatial_shape[1]
        self.D = spatial_shape[0]

    def dense(self):
        return torch.randn(
            (self.batch_size, self.C, self.D, self.W, self.H))

