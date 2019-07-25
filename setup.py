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
stella is a python package to characterize flares for stars in the 
TESS Full Frame Images.
"""


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
    description='For characterizing flares in the TESS FFIs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=[
        'tqdm', 'lightkurve', 'astropy',
        'astroquery', 'exoplanet',
        'setuptools>=41.0.0', 'wotan',
        'tensorflow', 'vaneska', 'beautifulsoup4>=4.6.0', 'tess-point'],


    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
