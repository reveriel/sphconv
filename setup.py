import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

from setup_helpers.cudnn import CUDNN_INCLUDE_DIR, CUDNN_LIB_DIR, WITH_CUDNN

LIBTORCH_ROOT = str(Path(torch.__file__).parent)

PYTHON_VERSION = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

# subprocess.run(["git", "submodule", "update", "--init", "sphconv/cutlass"])

#  print(get_python_inc())")  \
import distutils.sysconfig as sysconfig
from distutils.sysconfig import get_python_inc

#  print(sysconfig.get_config_var('LIBDIR'))")

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            '-DCMAKE_PREFIX_PATH={}'.format(LIBTORCH_ROOT),
            '-DPYBIND11_PTYHON_VERSION={}'.format(PYTHON_VERSION),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
            '-DPYTHON_INCLUDE_DIR={}'.format(get_python_inc()),
            '-DPYTHON_LIBRARY={}'.format(sysconfig.get_config_var('LIBDIR')),
            '-DSPHCONV_VERSION_INFO={}'.format(self.distribution.get_version()),
            ]

        cfg = 'Debug' if self.debug else 'Release'
        assert cfg == "Release", "pytorch ops don't support debug build."
        build_args = ['--config', cfg]
        print(cfg)

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "sphconv"))]
            # cmake_args += ['-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "spconv"))]
            cmake_args += ['-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "sphconv"))]
            cmake_args += ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE"]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(str(Path(extdir) / "sphconv"))]
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j' + str(os.cpu_count())]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("|||||CMAKE ARGS|||||", cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)

packages = find_packages()

setup(
    name='sphconv',
    version='0.1.5',
    # install_requires=['torch'],
    # setup_requires=['torch>=1.3.0'],
    packages=packages,
    # package_dir={'sphconv': './sphconv'},
    ext_modules=[
        CMakeExtension('sphconv_cuda'),
        CMakeExtension('sphconv_utils')
        # CUDAExtension(
        #     name='sphconv_cuda',gg
        #     include_dirs=['./', os.getcwd()+'/sphconv/include'],
        #     sources=[
        #         'sphconv/src/all.cpp'
        #     ],
        #     libraries=['make_pytorch'],
        #     library_dirs=['objs'],
        #     # extra_compile_args=['-g']
        #     extra_link_args=[
        #         '-L./sphconv/objs/'
        #     ]
        # )

    ],
    cmdclass={'build_ext': CMakeBuild},
    author='Guo.Xing.',
    author_email='reveriel@hotmail.com',
    description='hh',
    keywords='',
    url='https://github.com/reveriel/sphconv',
    zip_safe=False,
)
