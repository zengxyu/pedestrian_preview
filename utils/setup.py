#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 8/9/22 8:42 PM 
    @Description    :
        
===========================================
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("icp.pyx")
)
# python setup.py build_ext --inplace