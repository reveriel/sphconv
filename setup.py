from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
from setup_helpers.cudnn import WITH_CUDNN, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR
import platform
import multiprocessing

import os
import sys
import glob

# subprocess.run(["git", "submodule", "update", "--init", "sphconv/cutlass"])

from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        try:
            import torch
        except ImportError:
            sys.stderr.write("Pytorch is required to build this package\n")
            sys.exit(-1)

        self.pytorch_dir = os.path.dirname(torch.__file__)
        self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                      "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                      "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
                      "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # for kenlm - avoid seg fault
                      # "-DPYTHON_EXECUTABLE=".format(sys.executable),
                      ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(extdir))
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--", "-j{}".format(multiprocessing.cpu_count())])
        os.chdir(cwd)


old_cuda_extension =   CUDAExtension(
            name='sphconv_cuda',
            include_dirs=['./', os.getcwd()+'/sphconv/include'],
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


setup(
    name='sphconv',
    version='0.1.4',
    install_requires=['torch'],
    setup_requires = ['torch>=1.3.0'],
    # packages=['sphconv'],
    packages=find_packages(),
    license="MIT",
    ext_modules=[
        CMakeExtension("sphconv/taichi")
    ],
    cmdclass={'build_ext': CMakeBuild, },
    author='Guo.Xing.',
    author_email='reveriel@hotmail.com',
    description='hh',
    keywords='',
    url='https://github.com/reveriel/sphconv',
    zip_safe=False,
)
