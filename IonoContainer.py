#!/usr/bin/env python
"""
Created on Thu Dec 12 12:40:46 2013

@author: Bodangles
"""

import numpy as np
import scipy as sp
from ISSpectrum import ISSpectrum
import pdb

from physConstants import *
from sensorConstants import *
class IonoContainer(object):
    """Holds the coordinates and parameters to create the ISR data.  Also will 
    make the spectrums for each point."""
    def __init__(self,coordlist,paramlist,times = np.arange(1),sensor_loc = [0,0,0]):
        """ """
        
        X_vec = coordlist[:,0]
        Y_vec = coordlist[:,1]
        Z_vec = coordlist[:,2]
        
        R_vec = sp.sqrt(X_vec**2+Y_vec**2+Z_vec**2)
        Az_vec = sp.arctan2(Y_vec,X_vec)*180/pi     
        El_vec = sp.arcsin(Z_vec/R_vec)*180/pi
        
        
        self.Cart_Coords = coordlist        
        self.Sphere_Coords = sp.array([R_vec,Az_vec,El_vec]).transpose()
        self.Param_List = paramlist
        self.Time_Vector = times
    def makespectrums(self,range_gates,centangles,beamwidths,sensdict):
        """ Creates a spectrum for each range gate, it will be assumed that the 
        spectrums for each range will be averaged at the spectrum level"""
        
        az_limits = [centangles[0]-beamwidths[0]/2.0,centangles[0]+beamwidths[0]/2]
        el_limits = [centangles[1]-beamwidths[1]/2.0,centangles[1]+beamwidths[1]/2]
        # filter in az and el
        az_cond = (self.Sphere_Coords[:,1]>az_limits[0]) & (self.Sphere_Coords[:,1]<az_limits[1])
        el_cond = (self.Sphere_Coords[:,2]>el_limits[0]) & (self.Sphere_Coords[:,2]<el_limits[1])
        
       
        rng_len = sensdict['t_s']*v_C_0/1000.0
        # Reduce range dimesion and parameters to a single beam
        rho = self.Sphere_Coords[:,0]
        # Go through each range gate until 
        rng_dict = dict()
        param_dict = dict()
        #pdb.set_trace()        
        for rng in range_gates:
            rnglims = [rng-rng_len/2.0,rng+rng_len/2.0]
            rng_cond = (rho>rnglims[0]) & (rho<rnglims[1])
            cur_cond  = rng_cond & az_cond & el_cond
            if cur_cond.sum()==0:                
                #pdb.set_trace()
                x = rng*np.cos(centangles[1]*np.pi/180)*np.cos(centangles[0]*np.pi/180)
                y = rng*np.cos(centangles[1]*np.pi/180)*np.sin(centangles[0]*np.pi/180)
                z = rng*np.sin(centangles[1]*np.pi/180)
                checkmat = np.tile(np.array([x,y,z]),(len(cur_cond),1))
                error = checkmat-self.Cart_Coords
                argmin_dist = ((error**2).sum(1)).argmin()
                cur_cond[argmin_dist] = True
            (omeg,specs,params) = self.__getspectrums2__(cur_cond,sensdict)
            rng_dict[rng] = specs
            param_dict[rng] = params
        return (omeg,rng_dict,param_dict)
    def __getspectrums__(self,conditions,sensdict,weights=None):
        """ This will get a spectrum for a specific point in range and angle space.
        It will take all of the spectrums in an area and average them all together
        to get a single spectrum.  """        
        params_red = self.Param_List[conditions]
        num_true = conditions.sum()
        
        npts = 128
        myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
        
        
        datashape = params_red.shape
        #pdb.set_trace()
        if weights == None:
            weights = sp.ones(datashape[0])
        #normalize the weights
        weights = weights/weights.sum()
        spec_dict = dict()
        
        # case 1 have both a time and space dimension
        if params_red.ndim==3:
            # get the number of times
            num_times = datashape[1]
            for i_time in np.arange(num_times):
                first_thing = True
                for i_pos in np.arange(datashape[0]):
                    
                    cur_params = params_red[i_pos,i_time,:]
                    (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])
                    cur_spec = cur_spec*weights[i_pos]
                    
                    if first_thing ==True:
                        spec_out = cur_spec*weights[i_pos]
                    else:
                        spec_out = cur_spec*weights[i_pos]+spec_out
                
                spec_dict[i_time] =spec_out 
                
        #case2 have only  a time dimension, only one space elment
        elif num_true ==1 and params_red.ndim==2:
            # get the number of times
            num_times = datashape[0]
            for i_time in np.arange(num_times):
                    
                cur_params = params_red[i_time,:]
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                
                spec_dict[i_time] =cur_spec
        #case 3 have a space but no time
        elif num_true !=1 and params_red.ndim==2:
            
            
            first_thing = True
            
            for i_pos in np.arange(datashape[0]):
                
                cur_params = params_red[i_pos,:]
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                cur_spec = cur_spec*weights[i_pos]
                if first_thing:
                    spec_out = cur_spec
                    first_thing= False
                else:
                    spec_out = cur_spec+spec_out
            spec_dict[0] =spec_out 
        # case 4
        else:
            cur_params = params_red
            (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                cur_params[3], cur_params[4], cur_params[5])
                
            spec_dict[0] = cur_spec
        return (omeg,spec_dict)
        
    def __getspectrums2__(self,conditions,sensdict,weights=None):
        """ This will get a spectrum for a specific point in range and angle space.
        It will take all of the spectrums and put them into a dictionary.  The 
        dictionary will be a set where each number is   """        
        params_red = self.Param_List[conditions]
        num_true = conditions.sum()
        
        npts = 128
        #npts = 64
        myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
        
        datashape = params_red.shape
        
        # case 1 have both a time and space dimension
        if params_red.ndim==3:
            # get the number of times
            num_times = datashape[1]
            num_locs = datashape[0]
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,6))
            for i_time in np.arange(num_times):
                for i_pos in np.arange(datashape[0]):
                    
                    cur_params = params_red[i_pos,i_time,:]
                    (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])
                    
                    
                    spec_ar[i_pos,i_time] =cur_spec
                    params_ar[i_pos,i_time] = cur_params
                
        #case2 have only  a time dimension, only one space elment
        elif num_true ==1 and params_red.ndim==2:
            # get the number of times
            num_times = datashape[0]
            num_locs = 1
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,6))
            for i_time in np.arange(num_times):
                    
                cur_params = params_red[i_time,:]
                (omeg,spec_out) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                
                spec_ar[0,i_time] =spec_out
                params_ar[0,i_time] = cur_params
                
        #case 3 have a space but no time
        elif num_true !=1 and params_red.ndim==2:
            # get the number of times
            num_times = 1
            num_locs = datashape[0]
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,6))
            for i_pos in np.arange(num_locs):
                
                cur_params = params_red[i_pos,:]
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                               
                spec_ar[i_pos,0] =cur_spec 
                params_ar[i_pos,0] =cur_params
        # case 4
        else:
            num_times = 1
            num_locs =1
            
            num_times = datashape[1]
            num_locs = datashape[0]
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,6))
            
            cur_params = params_red
            (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                cur_params[3], cur_params[4], cur_params[5])
                
            spec_ar[0,0] = cur_spec
            params_ar[0,0] = cur_params
        
        if 'omeg' not in locals():
            pdb.set_trace()
        return (omeg,spec_ar,params_ar)
        
def Chapmanfunc(z,H_0,Z_0,N_0):
    """This function will return the Chapman function for a given altitude 
    vector z.  All of the height values are assumed km.
    Inputs 
    z: An array of z values in km.
    H_0: A single float of the height in km.
    Z_0: The peak density location.
    N_0: The peak electron density.
    """    
    z1 = (z-Z_0)/H_0
    Ne = N_0*sp.exp(0.5*(1-z1-sp.exp(-z1)))
    return Ne
    
def TempProfile(z):
    """This function creates a tempreture profile that is pretty much made up"""
    Ti_val = 1000.0
    Ti = Ti_val*sp.ones(z.shape)
    Te = 2*Ti_val*sp.ones(z.shape)
    Te_sep = sp.where(z>300)
    Te[Te_sep] =1.4*Ti_val
    return (Te,Ti)
    
def MakeTestIonoclass():

    xvec = sp.arange(-100,100,4)
    yvec = sp.arange(-100,100,4)
    zvec = sp.arange(200,500,2)
 
    xx,yy,zz = sp.meshgrid(xvec,yvec,zvec)
    H_0 = 40 #km
    z_0 = 300 #km
    N_0 = 10**11
 
    Ne_profile = Chapmanfunc(zz,H_0,z_0,N_0)
    (Te,Ti)= TempProfile(zz)
 
    coords = sp.zeros((xx.size,3))
    coords[:,0] = xx.flatten()
    coords[:,1] = yy.flatten()
    coords[:,2] = zz.flatten()
    
    params = sp.zeros((Ne_profile.size,6))
    params[:,0] = Ti.flatten()
    params[:,1] = Te.flatten()/Ti.flatten()
    params[:,2] = sp.log10(Ne_profile.flatten())
    params[:,3] = 16 # ion weight 
    params[:,4] = 1 # ion weight
    params[:,5] = 0
    
    Icont1 = IonoContainer(coordlist=coords,paramlist=params)
    return Icont1
if __name__== '__main__':
    
    Icont1 = MakeTestIonoclass()
    range_gates = np.arange(250.0,500.0,AMISR['t_s']*v_C_0/1000)
    centangles = [5,85]
    beamwidths = [2,2]
    (omeg,mydict,myparams) = Icont1.makespectrums(range_gates,centangles,beamwidths,AMISR)