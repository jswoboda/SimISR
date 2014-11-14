# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:20:31 2014

@author: John Swoboda
"""
import os, inspect
import numpy as np
from matplotlib import rc
import matplotlib.pylab as plt
import scipy.io as sio

from RadarDataSim.utilFunctions import *
from RadarDataSim.IonoContainer import IonoContainer
import RadarDataSim.const.sensorConstants as sensconst
from RadarDataSim.fitterMethods import default_fit_func
from const.physConstants import v_C_0, v_Boltz
from beamtools.bcotools import getangles
if __name__== '__main__':
    #%% Import Data
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Test')
    binpath = os.path.join(os.path.split(curpath)[0],'bin')
    DataLags = sio.loadmat(os.path.join(testpath,'ACFdata.mat'))
    NoiseLags =  sio.loadmat(os.path.join(testpath,'Noisedata.mat'))
    Icont1 = IonoContainer.readmat(os.path.join(testpath,'patchewtemp.mat'))
    Icont2 = IonoContainer.readmat(os.path.join(testpath,'patchwtempfit.mat'))
    
    #%% settings
    IPP = .0087
    angles = getangles(os.path.join(binpath, 'spcorbco.txt'))
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    Npulses = 200
    t_int = Npulses*IPP*len(angles)
    pulse = np.ones(14)
    rng_lims = [150,500]
    sensdict = sensconst.getConst('risr',ang_data)
    sensdict['Tsys']=0.001#reduce noise
    
    rm1 = 16
    rm2 = 1
    p2 = 0
    npnts = 128    
    
    #%% Get the closest point in time and space
    descoord = np.array([10,10,400])
    (paramout1,sphereout1,cartout1,dist1) = Icont1.getclosest(descoord)
    (paramout2,sphereout2,cartout2,dist2) = Icont2.getclosest(descoord)
    
    ibeam = np.where([np.allclose(iang,sphereout2[1:]) for iang in ang_data])[0]

    #determine times
    time_loc = (480+350)/2
    timesnotproc = np.argmin(np.abs(Icont1.Time_Vector-time_loc))
    itime = np.argmin(np.abs(DataLags['Time'][:,0]-(time_loc-t_int/2.0)))

    # set up a time vector
    time_lim = 2000.0
    timevec = sp.linspace(0.0,time_lim,num=220)    
    
    timearr = sp.linspace(0.0,time_lim,num=220)
    curint_time = t_int
    curint_time2 = 10.0*IPP*len(angles)
    
    rng_gates = np.arange(rng_lims[0],rng_lims[1],sensdict['t_s']*v_C_0*1e-3)
    sensdict['RG'] = rng_gates
    sumrule = np.array([[-2,-3,-3,-4,-4,-5,-5,-6,-6,-7,-7,-8,-8,-9],[1,1,2,2,3,3,4,4,5,5,6,6,7,7]])
    ambdict = make_amb(sensdict['fs'],30,sensdict['t_s']*len(pulse),len(pulse))
    #%% Set up lags to for normalization
    # get the data nd noise lags
    lagsData= DataLags['ACF']
    (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
    pulses = np.tile(DataLags['Pulses'][:,:,np.newaxis,np.newaxis],(1,1,Nrng,Nlags))
    
    # average by the number of pulses
    lagsData = lagsData/pulses
    lagsNoise=NoiseLags['ACF']
    lagsNoise = np.mean(lagsNoise,axis=2)
    pulsesnoise = np.tile(NoiseLags['Pulses'][:,:,np.newaxis],(1,1,Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = np.tile(lagsNoise[:,:,np.newaxis,:],(1,1,Nrng,1))
    # subtract out noise lags
    lagsData = lagsData-lagsNoise 
    
    # normalized out parameters
    pulsewidth = sensdict['taurg']*sensdict['t_s']
    txpower = sensdict['Pt']
    rng_vec = sensdict['RG']*1e3
    rng3d = np.tile(rng_vec[np.newaxis,np.newaxis,:,np.newaxis],(Nt,Nbeams,1,Nlags))
    Ksysvec = sensdict['Ksys']
    ksys3d = np.tile(Ksysvec[np.newaxis,:,np.newaxis,np.newaxis],(Nt,1,Nrng,Nlags))        
    lagsData = lagsData*rng3d*rng3d/(pulsewidth*txpower*ksys3d)
    minrg = -np.min(sumrule[0])
    maxrg = len(sensdict['RG'])-np.max(sumrule[1])
    Nrng2 = maxrg-minrg;

    curlag = np.zeros(Nlags)
    
    
    rngvec2new =  np.arange(minrg,maxrg)
    
    rngfinal = np.array([ np.mean(sensdict['RG'][i+sumrule[0,0]:i+sumrule[1,0]+1]) for i in rngvec2new])
    irng = rngvec2new[np.argmin(np.abs(rngfinal-sphereout2[0]))]
    
    curlag = np.array([np.mean(lagsData[itime,ibeam,irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag]) for ilag in np.arange(Nlags)])
    #%% Make the fit surface
    Ti_vec = np.arange(100.0,1500.0,50.0)
    Te_vec = np.arange(100.0,1500.0,50.0)
    Ne_vec = np.linspace(1e10,1e11,20)
    params = [Ti_vec,Te_vec,Ne_vec]
    numtype=np.complex128
    dinput = (curlag, pulse,ambdict,sensdict,rm1,rm2,p2,npnts,numtype)
    denom = (np.abs(np.fft.fft(curlag,n = npnts-1))**2).sum()
    cursurf = fitsurface(default_fit_func,params,dinput)
    cursurf = cursurf/denom
    
    indx = np.argmin(cursurf)
    minars = np.unravel_index(indx, cursurf.shape)
    print "Ti = "+str(Ti_vec[minars[0]]) + " Te = " +str(Te_vec[minars[1]])+" Ne {:.2e} ".format(Ne_vec[minars[2]])
    
    mylims = [np.min(np.log10(cursurf)),np.max(np.log10(cursurf))]
    for nenum,ine in enumerate(Ne_vec):
        plt.figure()
        imgplot = plt.imshow(np.log10(cursurf[:,:,nenum]),origin='lower',extent=[Ti_vec.min(),Ti_vec.max(),Te_vec.min(),Te_vec.max()])
        imgplot.set_clim(mylims)
        cbar = plt.colorbar()
        cbar.set_label(r'$log_{10}$ of normed error')#, labelpad=-40)
        plt.xlabel(r'$T_e$ in $^\circ$K')
        plt.ylabel(r'$T_i$ in $^\circ$K')
        plt.title(r'Fit surface For Test Data with $N_e$ = {:.2e}m'.format(ine))
        plt.show(False)
        
        plt.savefig("Fitsurface {:d}.png".format(nenum))
    
   