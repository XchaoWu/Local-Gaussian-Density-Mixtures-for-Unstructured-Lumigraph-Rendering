from setuptools import setup, find_packages
import unittest,os 
from typing import List
from glob import glob 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
print(find_packages())
CUDA_FLAGS = []  # type: List[str]



cufiles = glob(os.path.join(os.path.split(os.path.abspath(__file__))[0], "*.cu"))
cppfiles = glob(os.path.join(os.path.split(os.path.abspath(__file__))[0], "*.cpp"))
headers = [os.path.join(os.path.split(os.path.abspath(__file__))[0], 'include')]
# headers += ["/usr/local/include/opencv4", "/usr/local/include"]
headers += ["/usr/local/include"]
library_dir = ["./"]
# library_name = ["opencv_core", "opencv_imgproc","opencv_imgcodecs"]
library_name = ["z"]



ext_modules = [
    CUDAExtension('CUDA_EXT', cufiles + cppfiles,
    library_dirs=library_dir,
    libraries=library_name,
    include_dirs=headers,
    extra_compile_args={"cxx": ["-g", "-O0"]},
    extra_link_args=['-g']),
]


INSTALL_REQUIREMENTS = ['numpy', 'torch']

setup(
    name='CUDA_EXT',
    description='cuda extension operation',
    author='xchao',
    version='0.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
