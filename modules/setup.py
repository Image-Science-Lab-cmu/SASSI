#!/usr/bin/env python

'''
Description of files goes here.
'''

# System imports
import os
import sys

# Compilation tools
from distutils.core import Extension, setup
from Cython.Build import cythonize

# Scientific computing
import numpy as np

ext_modules = [
    Extension(
        "cassi_cp",
        ["cassi_cp.pyx"],
        extra_compile_args=['-fopenmp', '-march=native', '-O3', '-ffast-math'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name='cassi_cp',
    ext_modules=cythonize(ext_modules)
)
