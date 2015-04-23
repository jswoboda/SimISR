#!/usr/bin/env python
"""
Created on Thu Apr 23 14:42:30 2015

This script will run the unit tests for each of the classes in one script as a test to
see if the module has been installed correctly.
@author: John Swoboda
"""
import time
import makeConfigFiles
import IonoContainer
import radarData
import fitterMethodGen

def main():
    print('Making config files')
    time.sleep(3.0)
    makeConfigFiles.main()
    print('Making example input data')
    time.sleep(3.0)
    IonoContainer.main()
    print('Applying the radar platform to the data and creating lags')
    time.sleep(3.0)
    radarData.main()
    print('Now fitting data orm the lags.')
    time.sleep(3.0)
    fitterMethodGen.main()

if __name__== '__main__':


   main()