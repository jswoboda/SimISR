#!/usr/bin/env python
"""
patchexamp.py 
This script will create a simulation an ISR system with a small patch of plasma
going through the field of view of a radar with a background plasma density of
a simple Chapman function and Ti=Te=1000k.

@author: John Swoboda
"""

import numpy as np
import scipy as sp
from beamtools.bcotools import getangles

from IonoContainer import IonoContainer, Chapmanfunc
import const.sensorConstants as sensconst
from radarData import RadarData
from fitterMethods import FitterBasic, make_amb
if __name__== '__main__':
    
    # setup the ionosphere container.
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
    (Te,Ti)= (2000.0,2000.0)
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
    speed = sp.array([0,.5,0]) #km/s
    rad = 35.0
    val = 5e10
    params = sp.zeros((Ne_profile.size,timevec.size,6))
    
    for it,t in enumerate(timevec):
        centloc = centerstart+speed*t
        centlocrep = np.repeat(centloc[np.newaxis,:],len(coords),axis=0) 
        auglist = np.where(((coords-centlocrep)**2).sum(axis=1)<rad**2)[0]
        
        curprofile = Ne_profile.copy()
        curprofile[auglist] = val
        params[:,it,0] = Ti
        params[:,it,1] = Te/Ti
        params[:,it,2] = sp.log10(curprofile)
        params[:,it,3] = 16 # ion weight 
        params[:,it,4] = 1 # ion weight
        params[:,it,5] = 0
    
    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = timevec,coordvecs=coordVecs)
    
    IPP = .0087
    angles = getangles('spcorbco.txt')
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    Npulses = 200
    t_int = Npulses*IPP*len(angles)
    pulse = np.ones(14)
    rng_lims = [150,500]
    sensdict = sensconst.getConst('risr',ang_data)
    sensdict['Tsys']=0.001#reduce noise
    radardata = RadarData(Icont1,sensdict,angles,IPP,t_int,time_lim,pulse,rng_lims)

    timearr = sp.linspace(0.0,time_lim,num=220)
    curint_time = t_int
    (DataLags,NoiseLags) = radardata.processdata(timearr,curint_time)
    
    simparams = radardata.simparams.copy()
    
    simparams['SUMRULE'] = np.array([[-2,-3,-3,-4,-4,-5,-5,-6,-6,-7,-7,-8,-8,-9],[1,1,2,2,3,3,4,4,5,5,6,6,7,7]])
    simparams['amb_dict'] = make_amb(sensdict['fs'],30,sensdict['t_s']*len(pulse),len(pulse))
    curfitter =  FitterBasic(DataLags,NoiseLags,radardata.sensdict,simparams)   
    Ne = curfitter.fitNE()
    range_vec = sensdict['RG']
    ang_rep =  np.tile(ang_data,(len(range_vec),1))
    rangemat = np.repeat(range_vec[:,np.newaxis],ang_data.shape[0],axis=1)
    rangevecall = rangemat.flatten()
    Ne_trans = Ne.transpose()
    timearr= timearr[:Ne_trans.shape[2]]
    Nemat =np.zeros((len(rangevecall),len(timearr)))
    
    for irng in range(len(range_vec)):
        for iang in range(ang_data.shape[0]):
            Nemat[irng*ang_data.shape[0]+iang] =Ne_trans[irng,iang]
    coordvecsr = {'r':range_vec,'theta':ang_data[:,0],'phi':ang_data[:,1]}
    
    Icont2 = IonoContainer(coordlist=np.column_stack((rangevecall,ang_rep)),paramlist=Nemat,times=timearr,coordvecs=coordvecsr,ver=1)
    Icont1.savemat('/Users/Bodangles/Documents/MATLAB/ec717image/project/testcases/patchexv4.mat')
    Icont2.savemat('/Users/Bodangles/Documents/MATLAB/ec717image/project/testcases/patchexNEv4.mat')