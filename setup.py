from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
from setup_helpers.cudnn import WITH_CUDNN, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR

import os
import sys
import glob

# subprocess.run(["git", "submodule", "update", "--init", "sphconv/cutlass"])

library_dirs = [CUDNN_LIB_DIR]
include_dirs = [CUDNN_INCLUDE_DIR]

setup(
    name='sphconv',
    version='0.1.1',
    author='Guo.Xing.',
    author_email='reveriel@hotmail.com',
    setup_requires = ['torch>=1.3.0'],
    packages=['sphconv'],
    ext_modules=[
        CUDAExtension(
            name='sphconv_cuda',
            sources=[
                'sphconv/src/all.cpp',
                'sphconv/src/indice_conv.cu',
                'sphconv/src/indice.cu',
                'sphconv/src/to_dense.cu',
            ],
            extra_compile_args={
                'cxx': [ '-O3',
                '-I./sphconv/include' ],
                'nvcc':[
                    '-O3',
                    # '-fPIC',
                    #  '--compiler-options',
                    #  '"-fPIC -Wall -O2"',
                     '--shared',
                    '-lineno',
                    # '-DDEBUG',
                    # '-g',
                    '-gencode', 'arch=compute_75,code=sm_75',
                    '-I./sphconv/include',
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
    cmdclass={
        'build_ext': BuildExtension
    })
