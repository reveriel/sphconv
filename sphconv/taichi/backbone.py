
import torch
import sphconv.conv_taichi

class BackBoneFunction(torch.autograd.Function):

    def __init__(self) -> None:
        super(BackBoneFunction, self).__init__()
        # self.__handel__ = ;

    @staticmethod
    def forward(ctx, points, *weights):
        return;

    @staticmethod
    def backward(ctx, *grad_outputs):
        return
