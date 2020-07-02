from torch.utils.cpp_extension import load

# sphconv_cuda = load(
#     name='sphocnv_cuda',
#     source=[]
# )
lltm_cuda = load(
    'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)
