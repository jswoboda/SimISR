#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a set of data 
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
import  RadarDataSim.runsim.main as runsim 
from RadarDataSim.analysisplots import analysisdump
from ISRSpectrum.ISRSpectrum import ISRSpectrum


def configfilesetup(testpath,npulses):
    
    
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    defcon = os.path.join(curloc,'statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = ratio1 * simparams['TimeLim']
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(os.path.join(testpath,'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
def makedata(testpath):
    
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
    
def getinfo(curdir):
    
    origdataname = os.path.join(curdir,'Origparams','0 stats.h5')
    specnam = glob.glob(os.path.join(curdir,'Spectrums','*.h5'))[0]
    measredacf=glob.glob(os.path.join(curdir,'ACF','*.h5'))[0]
    fittedname=glob.glob(os.path.join(curdir,'Fitted','*.h5'))[0]
    
    
def main(plist = None,functlist = ['all']):

    if plist is None:
        sp.array([50,100,200,500,1000,2000,5000])
        
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curloc)[0],'StatsTest')
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
    
    allfolds
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold =os.path.join(testpath,foldname)
        allfolds.append(curfold)
        if not os.path.isdir(curfold):
            os.mkdir(curfold)
            makedata(curfold)
            configfilesetup(curfold,ip)
        runsim(functlist,curfold,os.path.join(curfold,'stats.ini'),True)
        analysisdump(curfold,os.path.join(curfold,'stats.ini'))
        
    for 