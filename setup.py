#!/usr/bin/env python
"""
setup.py
This is the setup file for the SimISR python package

@author: John Swoboda
"""
req = ['ISRSpectrum','lmfit',
'nose','six','numpy','scipy','tables','matplotlib','seaborn','pyyaml','pandas']

import os
from setuptools import setup, find_packages

config = {
    'description': 'An ISR data simulator',
    'author': 'John Swoboda',
    'url': 'https://github.com/jswoboda/SimISR.git',
    'version': '1.0.0',
    'install_requires': req,
    'python_requires': '>=2.7',
    'dependency_links': ['https://github.com/jswoboda/ISRSpectrum/tarball/master#egg=ISRSpectrum-999.0'],
    'packages': find_packages(),
    'name': 'SimISR'
}

curpath = os.path.dirname(__file__)
testpath = os.path.join(curpath,'Testdata')
try:
    os.mkdir(testpath)
except OSError:
    pass
print("created {}".format(testpath))

setup(**config)
