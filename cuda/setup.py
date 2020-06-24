from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
from setup_helpers.cudnn import WITH_CUDNN, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR

import os
import sys
import glob




# subprocess.run(["git", "submodule", "update", "--init", "cuda/cutlass"])

library_dirs = [CUDNN_LIB_DIR]
include_dirs = [CUDNN_INCLUDE_DIR]

setup(
    name='lltm_cuda',
    ext_modules=[
        # CUDAExtension('lltm_cuda', [
        #     'lltm_cuda.cpp',
        #     'lltm_cuda_kernel.cu',
        # ]),
        CUDAExtension(name='sphconv_cuda',
        sources=[
            'sphconv_cuda.cpp',
            'ConvolutionBuildingBlocks/winograd4x4.cu'],
        extra_compile_args={'cxx': ['-O3'],
                'nvcc':['-O3',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-I./cutlass/include',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math']},
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        extra_link_args=['-lcudnn']
        ),
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })