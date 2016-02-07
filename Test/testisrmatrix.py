#!/usr/bin/env python
"""
Created on Tue Oct 20 13:20:27 2015

@author: John Swoboda
"""
import os, inspect,glob
import scipy as sp
import shutil
from RadarDataSim.utilFunctions import makedefaultfile
from RadarDataSim.operatorstuff import makematPA
from RadarDataSim.IonoContainer import MakeTestIonoclass
import RadarDataSim.runsim as runsim



def main():

    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Testdata','MatrixTest')
    origparamsdir = os.path.join(testpath,'Origparams')
    if not os.path.exists(testpath):
        os.mkdir(testpath)
        print "Making a path for matrix test at "+testpath
    if not os.path.exists(origparamsdir):
        os.mkdir(origparamsdir)
        print "Making a path for testdata at "+origparamsdir

    # clear everything out
    folderlist = ['Origparams','Spectrums','Radardata','ACF','Fitted']
    for ifl in folderlist:
        flist = glob.glob(os.path.join(testpath,ifl,'*.h5'))
        for ifile in flist:
            os.remove(ifile)
    # Make Config file
    configname = os.path.join(testpath,'config.ini')
    
    if ~os.path.isfile(configname):
        srcfile =os.path.join( os.path.split(curpath)[0],'RadarDataSim','default.ini')
        shutil.copy(srcfile,configname)

    # make the coordinates
    xvec = sp.zeros((1))
    yvec = sp.zeros((1))
    zvec = sp.arange(50.0,900.0,2.0)
    # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
    xx,zz,yy = sp.meshgrid(xvec,zvec,yvec)
    coords = sp.zeros((xx.size,3))
    coords[:,0] = xx.flatten()
    coords[:,1] = yy.flatten()
    coords[:,2] = zz.flatten()

    Icont1 = MakeTestIonoclass(testv=True,testtemp=True,coords=coords)
    Icont1.saveh5(os.path.join(origparamsdir,'0 testiono.h5'))
    Icont1.saveh5(os.path.join(testpath,'startdata.h5'))
    funcnamelist=['spectrums','applymat','fittingmat']
    runsim.main(funcnamelist,testpath,configname,True)

if __name__== '__main__':

    main()