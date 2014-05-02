#!/usr/bin/env python
"""
Holds the IonoContainer class that contains the ionospheric parameters.
@author: John Swoboda
"""

import numpy as np
import scipy as sp
import scipy.io as sio
# From my 
from ISSpectrum import ISSpectrum

from const.physConstants import *
from const.sensorConstants import *
import matplotlib.pylab as plt
class IonoContainer(object):
    """Holds the coordinates and parameters to create the ISR data.  Also will 
    make the spectrums for each point."""
    def __init__(self,coordlist,paramlist,times = None,sensor_loc = [0,0,0],ver =0,coordvecs = None):
        """ This constructor function will use create an instance of the IonoContainer class
        using either cartisian or spherical coordinates depending on which ever the user prefers.
        Inputs:
        coordlist - Nx3 Numpy array where N is the number of coordinates.
        paramlist - NxTxP Numpy array where T is the number of times and P is the number of parameters
                    alternatively it could be NxP if there is only one time instance.
        times - A T length numpy array where T is the number of times.  This is
                optional input, if not given then its just a numpy array of 0-T
        sensor_loc - A numpy array of length 3 that gives the sensor location.
                    The default value is [0,0,0] in cartisian space.
        ver - (Optional) If 0 the coordlist is in Cartisian coordinates if 1 then 
        coordlist is a spherical coordinates.
        coordvecs - (Optional) A dictionary that holds the individual coordinate vectors.
        if sphereical coordinates keys are 'r','theta','phi' if cartisian 'x','y','z'.
        """
        r2d = 180.0/np.pi
        d2r = np.pi/180.0
        # Set up the size for the time vector if its not given.
        Ndims = paramlist.ndim
        psizetup = paramlist.shape
        if times==None:
            if Ndims==3:
                times = np.arange(psizetup[1])
            else:
                times = np.arange(1)
        # Assume that the 
        if ver==0:
        
            X_vec = coordlist[:,0]
            Y_vec = coordlist[:,1]
            Z_vec = coordlist[:,2]
        
            R_vec = sp.sqrt(X_vec**2+Y_vec**2+Z_vec**2)
            Az_vec = sp.arctan2(Y_vec,X_vec)*r2d     
            El_vec = sp.arcsin(Z_vec/R_vec)*r2d
            
            self.Cart_Coords = coordlist        
            self.Sphere_Coords = sp.array([R_vec,Az_vec,El_vec]).transpose()
            if coordvecs!= None:
                if set(coordvecs.keys())!={'x','y','z'}:
                    raise NameError("Keys for coordvecs need to be 'x','y','z' ")
            
        elif ver==1:
            R_vec = coordlist[:,0]
            Az_vec = coordlist[:,1]
            El_vec = coordlist[:,2]
            
            X_vec = R_vec*np.cos(Az_vec*d2r)*np.cos(El_vec*d2r)
            Y_vec = R_vec*np.sin(Az_vec*d2r)*np.cos(El_vec*d2r)
            Z_vec = R_vec*np.sin(El_vec*d2r)
            
            self.Cart_Coords = sp.array([X_vec,Y_vec,Z_vec]).transpose()        
            self.Sphere_Coords = coordlist
            if coordvecs!= None:
                if set(coordvecs.keys())!={'r','theta','phi'}:
                    raise NameError("Keys for coordvecs need to be 'r','theta','phi' ")
            
        self.Param_List = paramlist
        self.Time_Vector = times
        self.Coord_Vecs = coordvecs
    def savemat(self,filename):
        """ This method will write out a structured mat file and save information
        from the class.
        inputs
        filename - A string for the file name.
        """
        outdict = {'Cart_Coords':self.Cart_Coords,'Sphere_Coords':self.Sphere_Coords,\
            'Param_List':self.Param_List,'Time_Vector':self.Time_Vector}
        if self.Coord_Vecs!=None:    
            #fancy way of combining dictionaries
            outdict = dict(outdict.items()+self.Coord_Vecs.items())
            
        sio.savemat(filename,mdict=outdict)
    def makespectrums(self,range_gates,centangles,beamwidths,sensdict):
        """ Creates a spectrum for each range gate, it will be assumed that the 
        spectrums for each range will be averaged by adding the noisy signals
        Inputs:
        range_gates: Numpy array for each range sample.
        centangles: The center angle of each beam in a 2xNb numpy array.
        beamwidths: The beam width of the beams az and el.
        sensdict: The dictionary that holds the sensor parameters.
        Outputs:
        omeg: A numpy array of frequency samples.
        rng_dict: A dictionary which the keys are the range gate values in km which hold the spectrums
        params: A dictionary with keys of the range gate values in km that hold the parameters"""
        
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
        It will take all of the spectrums and put them into a dictionary.  
        inputs:
        conditions: An array the same size as the param array that is full of bools which 
        determine what parameters are used.
        sensdict: The dictionary of sensor parameters
        weights: Weighting to the different spectrums that will be taken.
        Outputs: 
        omeg: A numpy array of frequency samples.
        spec_ar: A numpy array that holds all of the spectrums that are in the 
        present time and space point selected.
        params_ar: A numpy array that holds all of the parameters in the present 
        time and space."""        
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
#    Ti_val = 1000.0
#    Ti = Ti_val*sp.ones(z.shape)
#    Te = 2*Ti_val*sp.ones(z.shape)
#    Te_sep = sp.where(z>300)
#    Te[Te_sep] =1.4*Ti_val
    
    Te = ((45.0/500.0)*(z-200.0))**2+1000.0
    Ti = ((20.0/500.0)*(z-200.0))**2+1000.0
    return (Te,Ti)
    
def MakeTestIonoclass():

    xvec = sp.arange(-250.0,250.0,6.0)
    yvec = sp.arange(-250.0,250.0,6.0)
    zvec = sp.arange(200.0,500.0,3.0)
    # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
    xx,zz,yy = sp.meshgrid(xvec,zvec,yvec)
    H_0 = 40 #km
    z_0 = 300 #km
    N_0 = 10**11
 
    Ne_profile = Chapmanfunc(zz,H_0,z_0,N_0)
    (Te,Ti)= TempProfile(zz)
    Te = np.ones_like(zz)*1000.0
    Ti = np.ones_like(zz)*1000.0
 
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
    Icont1.savemat('test.mat')