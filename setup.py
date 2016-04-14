#!/usr/bin/env python
"""
setup.py
This is the setup file for the RadarDataSim python package

@author: John Swoboda
"""
import os, inspect
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'An ISR data simulator',
    'author': 'John Swoboda',
    'url': '',
    'download_url': 'https://github.com/jswoboda/RadarDataSim.git',
    'author_email': 'swoboj@bu.edu',
    'version': '1',
    'install_requires': ['numpy', 'scipy', 'tables','ISRSpectrum','numba','lmfit'],
    'packages': ['RadarDataSim','beamtools','radarsystools'],
    'scripts': [],
    'name': 'RadarDataSim'
}

curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testpath = os.path.join(curpath,'Testdata')
if not os.path.exists(testpath):
    os.mkdir(testpath)
    print "Making a path for testdata at "+testpath
setup(**config)
