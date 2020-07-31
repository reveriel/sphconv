from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
from setup_helpers.cudnn import WITH_CUDNN, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR

import os
import sys
import glob

# subprocess.run(["git", "submodule", "update", "--init", "sphconv/cutlass"])

os.system('make -j%d' % os.cpu_count())

setup(
    name='sphconv',
    version='0.1.4',
    install_requires=['torch'],
    setup_requires = ['torch>=1.3.0'],
    packages=['sphconv'],
    package_dir={'sphconv': './sphconv'},
    ext_modules=[
        CUDAExtension(
            name='sphconv_cuda',
            include_dirs=['./', 'sphconv/include'],
            sources=[
                'sphconv/src/all.cpp'
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
            extra_link_args=[
                '-L./sphconv/objs/'
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Guo.Xing.',
    author_email='reveriel@hotmail.com',
    description='hh',
    keywords='',
    url='https://github.com/reveriel/sphconv',
    zip_safe=False,
)
