#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup

sys.path.insert(0, "stella")
from version import __version__


long_description = \
    """
stella is a python package to identify and characterize flares in 
TESS short-cadence data using a convolutional neural network. In its
simplest form, stella takes an array of light curves and predicts where
flares are using the models provided in Feinstein et al. submitted and 
returns predictions. 

Read the documentation at https://adina.feinste.in/stella

Changes to v0.1.0 (2020-05-18):
*  
"""

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='stella',
    version=__version__,
    license='MIT',
    author='Adina D. Feinstein',
    author_email='adina.d.feinstein@gmail.com',
    packages=[
        'stella',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/stella',
    description='For finding flares in TESS 2-min data with a CNN',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        ],
    )
