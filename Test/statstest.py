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
from  RadarDataSim.runsim import main as runsim 
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
    fittedname=glob.glob(os.path.join(curdir,'Fitted','*.h5'))[0]
    
    # Get Parameters
    Origparam = IonoContainer.readh5(origdataname)
    fittedparam = IonoContainer.readh5(fittedname)
    
    Ne_real = Origparam.Param_List[0,0,1,0]
    Te_real = Origparam.Param_List[0,0,1,1]
    Ti_real = Origparam.Param_List[0,0,0,1]
    
    realdict = {'Ne':Ne_real,'Te':Te_real,'Ti':Ti_real,'Nepow':Ne_real}
    paramnames = fittedparam.Param_Names
    params = ['Ne','Te','Ti','Nepow']
    params_loc = [sp.argwhere(i==sp.array(paramnames))[0][0] for i in params]
    
    datadict = {i:fittedparam.Param_List[:,:,j] for i,j in zip(params,params_loc)}
    datavars = {i:sp.nanvar(datadict[i],axis=1) for i in params}
    dataerror = {i:sp.nanmean((datadict[i]-realdict[i])**2,axis=1) for i in params}
    
    return (datadict,datavars,dataerror, realdict)
def main(plist = None,functlist = ['spectrums','radardata','fitting']):

    if plist is None:
        plist = sp.array([50,100,200,500,1000,2000,5000])
        
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curloc)[0],'Testdata','StatsTest')
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()
    allfolds = []
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold =os.path.join(testpath,foldname)
        allfolds.append(curfold)
        if not os.path.isdir(curfold):
            os.mkdir(curfold)
            makedata(curfold)
            configfilesetup(curfold,ip)
        if check_run :
            runsim(functlist_red,curfold,os.path.join(curfold,'stats.ini'),True)
        if 'analysis' in functlist:
            analysisdump(curfold,os.path.join(curfold,'stats.ini'))
        if 'stats' in functlist:
            datadict,datavars,dataerror,realdict = getinfo(curfold)
        
    