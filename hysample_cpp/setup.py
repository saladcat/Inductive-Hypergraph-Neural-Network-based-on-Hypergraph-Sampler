from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hysample_cpp',
      ext_modules=[
          cpp_extension.CppExtension('hysample_cpp', ['hysample.cpp'])
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
