#!/usr/bin/env python
"""
Created on Thu Oct 30 16:01:11 2014

@author: John Swoboda
This script will make a plasma patch with a tempreture enhancement along with
"""

import os, inspect
import numpy as np
import scipy as sp

# Import Main classes
from RadarDataSim.IonoContainer import IonoContainer
from RadarDataSim.radarData import RadarData
from RadarDataSim.fitterMethods import FitterBasic
# import utilities and constants
import RadarDataSim.const.sensorConstants as sensconst
from RadarDataSim.utilFunctions import make_amb, Chapmanfunc, TempProfile
from beamtools.bcotools import getangles


if __name__== '__main__':

    #%% set up paths
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Test')
    #%% setup the ionosphere container.
    xvec = sp.arange(-150.0,150.0,20.0)
    yvec = sp.arange(-150.0,150.0,20.0)
    zvec = sp.arange(100.0,600.0,10.0)
    # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
    xx,zz,yy = sp.meshgrid(xvec,zvec,yvec)
    coordVecs = {'x':xvec,'y':yvec,'z':zvec}

    # Create a background with a basic chapman profile.
    H_0 = 40 #km
    z_0 = 230 #km
    N_0 = 1e11
    Ne_profile = Chapmanfunc(zz.flatten() ,H_0,z_0,N_0)
    # make Te and Ti fixed to a ratio of 1.
    (Te,Ti)= TempProfile(zz.flatten())
    # Make the coordinates array.
    coords = sp.zeros((xx.size,3))
    coords[:,0] = xx.flatten()
    coords[:,1] = yy.flatten()
    coords[:,2] = zz.flatten()
    # set up a time vector
    time_lim = 2000.0
    timevec = sp.linspace(0.0,time_lim,num=220)

    # Make the ball of plasma travel through the background
    centerstart = sp.array([0,-200.0,400])
    vel = sp.array([0,.5,0]) #km/s
    rad = 35.0
    val = 5e10
    params = sp.zeros((Ne_profile.size,timevec.size,7),dtype=np.float)
    Vi_all = np.zeros(Ne_profile.size,dtype=np.float)
    for it,t in enumerate(timevec):
        centloc = centerstart+vel*t
        centlocrep = np.repeat(centloc[None,:],len(coords),axis=0)
        auglist = np.where(((coords-centlocrep)**2).sum(axis=1)<rad**2)[0]
        Titemp = Ti.copy()
        Tetemp = Te.copy()

        curcoords = coords[auglist]
        denom = np.tile(np.sqrt(np.sum(curcoords**2,1))[:,None],(1,3))
        unit_coords = curcoords/denom
        Vi = (np.tile(vel[None,:],(len(auglist),1))*unit_coords).sum(1)
        Vi_all[auglist] = Vi

        curprofile = Ne_profile.copy()
        ratiodiff = curprofile[auglist]/val
        Titemp[auglist] = Titemp[auglist]*ratiodiff
        Tetemp[auglist] = Tetemp[auglist]*ratiodiff
        curprofile[auglist] = val
        params[:,it,0] = Titemp
        params[:,it,1] = Tetemp/Titemp
        params[:,it,2] = sp.log10(curprofile)
        params[:,it,3] = 16 # ion weight
        params[:,it,4] = 1 # ion weight
        params[:,it,5] = 0
        params[:,it,6] =Vi_all

    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = timevec,coordvecs=coordVecs)
    Icont1.savemat(os.path.join(testpath,'pathwtempdoppler.mat'))

    sensdict = sensconst.getConst('risr')

    Icont2 = Icont1.makespectruminstance(sensdict,128)
    Icont2.savemat(os.path.join(testpath,'pathwtempdopplerspecs.mat'))
