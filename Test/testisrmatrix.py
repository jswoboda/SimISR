#!/usr/bin/env python
"""
Created on Tue Oct 20 13:20:27 2015

@author: John Swoboda
"""
from RadarDataSim import Path
import scipy as sp
import shutil
from RadarDataSim.utilFunctions import makedefaultfile
from RadarDataSim.operators import makematPA
from RadarDataSim.IonoContainer import MakeTestIonoclass
from RadarDataSim.analysisplots import plotbeamparametersv2
import RadarDataSim.runsim as runsim



def main():

    curpath = Path(__file__).expanduser().parents[1]
    print(curpath)
    testpath = curpath/'Testdata'/'MatrixTest'
    origparamsdir = testpath/'Origparams'
    
    testpath.mkdir(exist_ok=True,parents=True)

    origparamsdir.mkdir(exist_ok=True,parents=True)

    # clear everything out
    folderlist = ['Origparams','Spectrums','Radardata','ACF','Fitted']
    for ifl in folderlist:
        flist = (testpath/ifl).glob('*.h5')
        for ifile in flist:
            ifile.unlink()
    # Make Config file
    configname = testpath/'config.ini'
    
    if not configname.is_file():
        srcfile = curpath/'RadarDataSim'/'default.ini'
        shutil.copy(str(srcfile),str(configname))

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
    Z_0 = 250.
    H_0=30.
    N_0=6.5e11
    Icont1 = MakeTestIonoclass(testv=True,testtemp=True,N_0=N_0,z_0=Z_0,H_0=H_0,coords=coords)
    Icont1.saveh5(origparamsdir/'0 testiono.h5')
    Icont1.saveh5(testpath/'startdata.h5')
    funcnamelist=['spectrums','applymat','fittingmat']
    runsim.main(funcnamelist,testpath,configname,True)
    
    plotdir = testpath/'AnalysisPlots'
    
    plotdir.mkdir(exist_ok=True,parents=True)
    
    f_templ = str(plotdir/'params')
    
    plotbeamparametersv2([0.],str(configname),str(testpath),fitdir = 'FittedMat',params=['Ne','Ti','Te'],filetemplate=f_templ,
                         suptitle = 'With Mat',werrors=False,nelog=False)
                         
if __name__== '__main__':

    main()
