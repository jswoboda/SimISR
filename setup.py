#!/usr/bin/env python
"""
setup.py
This is the setup file for the SimISR python package

@author: John Swoboda
"""
req = ['ISRSpectrum', 'nose','six','numpy','scipy','matplotlib','seaborn','pyyaml','pandas','digital_rf']

import os
from setuptools import setup, find_packages

config = {
    'description': 'An ISR data simulator',
    'author': 'John Swoboda',
    'url': 'https://github.com/jswoboda/SimISR.git',
    'version': '2.0.0b',
    'install_requires': req,
    'python_requires': '>=3.6',
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
