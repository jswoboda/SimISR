#!/usr/bin/env python
"""
Created on Fri Apr 15 15:00:12 2016
This script will be used to see if any changes will prevent the simulator from running.
@author: John Swoboda
"""

import os,inspect,glob
import scipy as sp
import scipy.fftpack as scfft
import scipy.interpolate as spinterp
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from RadarDataSim.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf,makeconfigfile
from RadarDataSim.IonoContainer import IonoContainer
from  RadarDataSim.runsim import main as runsim 
from RadarDataSim.analysisplots import analysisdump


def configfilesetup(testpath,npulses):
    """ """
    
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    defcon = os.path.join(curloc,'statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = 10*tint
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(os.path.join(testpath,'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
def makedata(testpath):
    """ """
    finalpath = os.path.join(testpath,'Origparams')
    if not os.path.isdir(finalpath):
        os.mkdir(finalpath)
    data = sp.array([1e12,2500.])
    z = sp.linspace(50.,1e3,50)
    nz = len(z)
    params = sp.tile(data[sp.newaxis,sp.newaxis,sp.newaxis,:],(nz,1,2,1))
    coords = sp.column_stack((sp.ones(nz),sp.ones(nz),z))
    species=['O+','e-']
    times = sp.array([[0,1e3]])
    vel = sp.zeros((nz,1,3))
    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
        
    finalfile = os.path.join(finalpath,'0 stats.h5')
    Icont1.saveh5(finalfile)
    Icont1.saveh5(os.path.join(testpath,'startfile.h5'))
    

def main(npulse = 100 ,functlist = ['spectrums','radardata','fitting','analysis']):

    
        
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curloc)[0],'Testdata','BasicTest')
    
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
        makedata(testpath)
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()

    
    configfilesetup(testpath,npulse)
    config = os.path.join(testpath,'stats.ini')
    (sensdict,simparams) = readconfigfile(config)
   
    if check_run :
        runsim(functlist_red,testpath,config,True)
    if 'analysis' in functlist:
        analysisdump(testpath,config)

if __name__== '__main__':
    from argparse import ArgumentParser
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    p = ArgumentParser(description=descr)
    
    p.add_argument("-p", "--npulses",help='Number of pulses.',type=int,default=100)
    p.add_argument('-f','--funclist',help='Functions to be uses',nargs='+',default=['spectrums','radardata','fitting','analysis'])#action='append',dest='collection',default=['spectrums','radardata','fitting','analysis'])
    
    p = p.parse_args()
    main(p.npulses,p.funclist)
   