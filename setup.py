#!/usr/bin/env python
"""
setup.py
This is the setup file for the SimISR python package

@author: John Swoboda
"""
req = ['ISRSpectrum', "mitarspysigproc",'nose', 'six', 'numpy', 'scipy', 'matplotlib', 'pyyaml', 'pandas', 'digital_rf', "xarray"]

from setuptools import setup, find_packages

config = {
    'description': 'An ISR data simulator',
    'author': 'John Swoboda',
    'url': 'https://github.com/jswoboda/SimISR.git',
    'version': '2.0.0b',
    'install_requires': req,
    'python_requires': '>=3.6',
    'dependency_links': ['https://github.com/jswoboda/ISRSpectrum/tarball/main#egg=ISRSpectrum-999.0','https://github.com/MIT-Adaptive-Radio-Science/sigprocpython/tarball/main#egg=repo-1.0.0'],
    'packages': find_packages(),
    'name': 'SimISR'
}

setup(**config)
