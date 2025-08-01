#!/usr/bin/env python
"""
setup.py
This is the setup file for the SimISR python package

@author: John Swoboda
"""
with open("requirements.txt") as f:
    req = f.read().splitlines()


from setuptools import setup, find_packages

config = {
    "description": "An ISR data simulator",
    "author": "John Swoboda",
    "url": "https://github.com/jswoboda/SimISR.git",
    "version": "2.0.0b",
    "install_requires": req,
    "setup_requires": req,
    "python_requires": ">=3.6",
    "packages": find_packages(),
    "name": "SimISR",
}

setup(**config)
