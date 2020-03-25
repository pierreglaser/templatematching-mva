#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup



dist = setup(
    name='templatematching',
    version='0.0.1dev0',
    description='Classifers based on template matching',
    author='Pierre Glaser',
    author_email='pierreglaser@msn.com',
    license='BSD 3-Clause License',
    packages=['cloudpickle'],
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    python_requires='>=3.6'
)
