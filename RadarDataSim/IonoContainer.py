#!/usr/bin/env python
"""
Holds the IonoContainer class that contains the ionospheric parameters.
@author: John Swoboda
"""

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.interpolate
import pdb
# From my 
from ISSpectrum import ISSpectrum
from const.physConstants import v_C_0, v_Boltz, v_epsilon0
import const.sensorConstants as sensconst
from utilFunctions import Chapmanfunc, TempProfile

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
        if Ndims==2:
            paramlist = paramlist[:,np.newaxis,:]
            
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
    def getclosestsphere(self,coords):
        d2r = np.pi/180.0
        (R,Az,El) = coords
        x_coord = R*np.cos(Az*d2r)*np.cos(El*d2r)
        y_coord = R*np.sin(Az*d2r)*np.cos(El*d2r)
        z_coord= R*np.sin(El*d2r)
        cartcoord = np.array([x_coord,y_coord,z_coord])
        return self.getclosest(cartcoord)
    def getclosest(self,coords):
        """"""
        X_vec = self.Cart_Coords[:,0]
        Y_vec = self.Cart_Coords[:,1]
        Z_vec = self.Cart_Coords[:,2]
        
        xdiff = X_vec -coords[0]
        ydiff = Y_vec -coords[1]
        zdiff = Z_vec -coords[2]
        distall = xdiff**2+ydiff**2+zdiff**2
        minidx = np.argmin(distall)
        paramout = self.Param_List[minidx]
        sphereout = self.Sphere_Coords[minidx]
        cartout = self.Cart_Coords[minidx]
        return (paramout,sphereout,cartout,np.sqrt(distall[minidx]))
        
        
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
    @staticmethod
    def readmat(filename):
         indata = sio.loadmat(filename)
         if "sensor_loc" in indata.keys():
             return IonoContainer(indata['Cart_Coords'],indata['Param_List'],indata['Time_Vector'],indata['Param_List'])
         else:
             return IonoContainer(indata['Cart_Coords'],indata['Param_List'],indata['Time_Vector'])
             
    def makeallspectrums(self,sensdict,npts):
        
        #npts is going to be lowered by one because of this.        
        if np.mod(npts,2)==0:
            npts = npts-1
        specobj = ISSpectrum(nspec = npts,sampfreq=sensdict['fs'])
        
        paramshape = self.Param_List.shape
        if len(paramshape)==3:
            outspecs = np.zeros((paramshape[0],paramshape[1],npts))
            full_grid = True
        elif len(paramshape)==2:
            outspecs = np.zeros((paramshape[0],1,npts))
            full_grid = False
        (N_x,N_t) = outspecs.shape[:2]
        #pdb.set_trace()
        first_spec = True
        for i_x in np.arange(N_x):
            for i_t in np.arange(N_t):
                if full_grid:
                    cur_params = self.Param_List[i_x,i_t]
                else:
                    cur_params = self.Param_List[i_x]
                Ti = cur_params[0]
                Tr = cur_params[1]
                Te = Ti*Tr
                N_e = 10**cur_params[2]
                debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
                if first_spec:
                    (omeg,cur_spec) = specobj.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])
                    fillspot = np.argmax(omeg)
                else:
                    cur_spec = specobj.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])[0]
                cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                # Ion velocity                    
                if len(cur_params)>6:
                    Vi = cur_params[-1]
                    Fd = -2.0*Vi/sensdict['lamb']
                    omegnew = omeg-Fd
                    fillval = cur_spec_weighted[fillspot]
                    cur_spec_weighted =scipy.interpolate.interp1d(omeg,cur_spec_weighted,bounds_error=0,fill_value=fillval)(omegnew) 
        
                outspecs[i_x,i_t] = cur_spec_weighted
                
        return (omeg,outspecs,npts)
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
        
        d2r = np.pi/180.0
        rng_len = sensdict['t_s']*v_C_0/1000.0
        # Reduce range dimesion and parameters to a single beam
        rho = self.Sphere_Coords[:,0]
        # Go through each range gate until 
        rng_dict = dict()
        param_dict = dict()
        #pdb.set_trace()      
        centanglesr = [iang*d2r for iang in centangles]
        for rng in range_gates:
            rnglims = [rng-rng_len/2.0,rng+rng_len/2.0]
            rng_cond = (rho>rnglims[0]) & (rho<rnglims[1])
            cur_cond  = rng_cond & az_cond & el_cond
            if cur_cond.sum()==0:                
                #pdb.set_trace()
                x = rng*np.cos(centanglesr[1])*np.cos(centanglesr[0])
                y = rng*np.cos(centanglesr[1])*np.sin(centanglesr[0])
                z = rng*np.sin(centanglesr[1])
                checkmat = np.tile(np.array([x,y,z]),(len(cur_cond),1))
                error = checkmat-self.Cart_Coords
                argmin_dist = ((error**2).sum(1)).argmin()
                cur_cond[argmin_dist] = True
            (omeg,specs,params) = self.__getspectrums2__(cur_cond,sensdict)
            rng_dict[rng] = specs
            param_dict[rng] = params
        return (omeg,rng_dict,param_dict)
    def __getspectrums__(self,conditions,sensdict,weights=None,npts=128):
        """ This will get a spectrum for a specific point in range and angle space.
        It will take all of the spectrums in an area and average them all together
        to get a single spectrum.  """        
        params_red = self.Param_List[conditions]
        num_true = conditions.sum()
        
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
                    
                     # get the plasma parameters
                    Ti = cur_params[0]
                    Tr = cur_params[1]
                    Te = Ti*Tr
                    N_e = 10**cur_params[2]
                    debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                    rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
                    (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])
                    cur_spec = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                    cur_spec = cur_spec*weights[i_pos]
                    # Ion velocity                    
                    if len(cur_params)>6:
                        Vi = cur_params[-1]
                        Fd = -2.0*Vi/sensdict['lamb']
                        omegnew = omeg-Fd
                        cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0)(omegnew) 
                    
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
                # Ion velocity                    
                if len(cur_params)>6:
                    Vi = cur_params[-1]
                    Fd = -2.0*Vi/sensdict['lamb']
                    omegnew = omeg-Fd
                    cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0)(omegnew) 
                spec_dict[i_time] =cur_spec
        #case 3 have a space but no time
        elif num_true !=1 and params_red.ndim==2:
            
            
            first_thing = True
            
            for i_pos in np.arange(datashape[0]):
                
                cur_params = params_red[i_pos,:]
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                cur_spec = cur_spec*weights[i_pos]
                # Ion velocity                    
                if len(cur_params)>6:
                    Vi = cur_params[-1]
                    Fd = -2.0*Vi/sensdict['lamb']
                    omegnew = omeg-Fd
                    cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0)(omegnew) 
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
            # Ion velocity                    
            if len(cur_params)>6:
                Vi = cur_params[-1]
                Fd = -2.0*Vi/sensdict['lamb']
                omegnew = omeg-Fd
                cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0)(omegnew) 
            spec_dict[0] = cur_spec
        return (omeg,spec_dict)
        
    def __getspectrums2__(self,conditions,sensdict,weights=None,npts=128):
        """ This will get a spectrum for a specific point in range and angle space.
        It will take all of the spectrums and put them into a dictionary. The spectrums
        will be scaled so that power based off of the parameters will be included. 
        Specifically the sum of the spectrum will equal N^2*a where N is the length
        of the spectrum and a is the power derived from the parameters of 
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
        
        myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
        
        datashape = params_red.shape
        nparams = datashape[-1]
#        pdb.set_trace()

        # case 1 have both a time and space dimension
        if params_red.ndim==3:
            # get the number of times
            num_times = datashape[1]
            num_locs = datashape[0]
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,nparams))
            for i_time in np.arange(num_times):
                for i_pos in np.arange(datashape[0]):
                    
                    cur_params = params_red[i_pos,i_time,:]
                    
                    # get the plasma parameters
                    Ti = cur_params[0]
                    Tr = cur_params[1]
                    Te = Ti*Tr
                    N_e = 10**cur_params[2]
                    # Make the the scaling for the power
                    debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                    rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
                    # Make the spectrum
                    (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                        cur_params[3], cur_params[4], cur_params[5])
                    
                    cur_spec = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                    # Ion velocity                    
                    if len(cur_params)>6:
                        Vi = cur_params[-1]
                        Fd = -2.0*Vi/sensdict['lamb']
                        omegnew = omeg-Fd
                        fillspot = np.argmax(omeg)
                        fillval = cur_spec[fillspot]
                        cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0,fill_value=fillval)(omegnew)                 
                        
                    spec_ar[i_pos,i_time] =cur_spec
                    params_ar[i_pos,i_time] = cur_params
                
        #case2 have only  a time dimension, only one space elment
        elif num_true ==1 and params_red.ndim==2:
            # get the number of times
            num_times = datashape[0]
            num_locs = 1
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,nparams))
            for i_time in np.arange(num_times):
                    
                cur_params = params_red[i_time,:]
                # get the plasma parameters
                Ti = cur_params[0]
                Tr = cur_params[1]
                Te = Ti*Tr

                N_e = 10**cur_params[2]
                # Make the the scaling for the power
                debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
                # Make the spectrum
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                
                cur_spec = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                # Ion velocity    
#                pdb.set_trace()                
                if len(cur_params)>6:
                    Vi = cur_params[-1]
                    Fd = -2.0*Vi/sensdict['lamb']
                    omegnew = omeg-Fd
                    fillspot = np.argmax(omeg)
                    fillval = cur_spec[fillspot]
                    cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0,fill_value=fillval)(omegnew) 
                spec_ar[0,i_time] =cur_spec
                params_ar[0,i_time] = cur_params
                
        #case 3 have a space but no time
        elif num_true !=1 and params_red.ndim==2:
            # get the number of times
            num_times = 1
            num_locs = datashape[0]
            spec_ar = sp.zeros((num_locs,num_times,npts-1))
            # need to change this if moving to a new spectrum
            params_ar = sp.zeros((num_locs,num_times,nparams))
            for i_pos in np.arange(num_locs):
                
                cur_params = params_red[i_pos,:]
                # get the plasma parameters
                Ti = cur_params[0]
                Tr = cur_params[1]
                Te = Ti*Tr
                N_e = 10**cur_params[2]
                
                # Make the the scaling for the power
                debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
                # Make the spectrum
                (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                    cur_params[3], cur_params[4], cur_params[5])
                
                cur_spec = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
                # Ion velocity                    
                if len(cur_params)>6:
                    Vi = cur_params[-1]
                    Fd = -2.0*Vi/sensdict['lamb']
                    omegnew = omeg-Fd
                    fillspot = np.argmax(omeg)
                    fillval = cur_spec[fillspot]
                    cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0,fill_value=fillval)(omegnew) 
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
            params_ar = sp.zeros((num_locs,num_times,nparams))
            
            cur_params = params_red
            # get the plasma parameters
            Ti = cur_params[0]
            Tr = cur_params[1]
            Te = Ti*Tr
            N_e = 10**cur_params[2]
            
            # Make the the scaling for the power
            debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
            rcs = N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))# based of new way of calculating
            # Make the spectrum
            (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
                cur_params[3], cur_params[4], cur_params[5])
                
            cur_spec = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()    
            # Ion velocity                    
            if len(cur_params)>6:
                Vi = cur_params[-1]
                Fd = -2.0*Vi/sensdict['lamb']
                omegnew = omeg-Fd
                fillspot = np.argmax(omeg)
                fillval = cur_spec[fillspot]
                cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0,fill_value=fillval)(omegnew) 
            spec_ar[0,0] = cur_spec
            params_ar[0,0] = cur_params
        
        if 'omeg' not in locals():
            pdb.set_trace()
        return (omeg,spec_ar,params_ar)

# utility functions
            
    
def MakeTestIonoclass(testv=False,testtemp=False):
    """ This function will create a test ionoclass with an electron density that
    follows a chapman function"""
    xvec = sp.arange(-250.0,250.0,20.0)
    yvec = sp.arange(-250.0,250.0,20.0)
    zvec = sp.arange(200.0,500.0,3.0)
    # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
    xx,zz,yy = sp.meshgrid(xvec,zvec,yvec)
    coords = sp.zeros((xx.size,3))
    coords[:,0] = xx.flatten()
    coords[:,1] = yy.flatten()
    coords[:,2] = zz.flatten()    
    
    H_0 = 40 #km scale height
    z_0 = 300 #km
    N_0 = 10**11
    
    # Make electron density
    Ne_profile = Chapmanfunc(zz,H_0,z_0,N_0)
    # Make temperture background
    if testtemp:
        (Te,Ti)= TempProfile(zz)
    else:
        Te = np.ones_like(zz)*1000.0
        Ti = np.ones_like(zz)*1000.0
        
    # set up the velocity
    vel = sp.zeros(coords.shape)
    if testv:
        vel[:,2] = zz.flatten()/5.0
    
    denom = np.tile(np.sqrt(np.sum(coords**2,1))[:,np.newaxis],(1,3))
    unit_coords = coords/denom
    Vi = (vel*unit_coords).sum(1)
    # put the parameters in order    
    params = sp.zeros((Ne_profile.size,7))
    params[:,0] = Ti.flatten()
    params[:,1] = Te.flatten()/Ti.flatten()
    params[:,2] = sp.log10(Ne_profile.flatten())
    params[:,3] = 16 # ion weight 
    params[:,4] = 1 # ion weight
    params[:,5] = 0
    params[:,6] = Vi
    
    Icont1 = IonoContainer(coordlist=coords,paramlist=params)
    return Icont1
if __name__== '__main__':
    
    Icont1 = MakeTestIonoclass()
    angles = [(90,85)]
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    sensdict = sensconst.getConst('risr',ang_data)

    range_gates = np.arange(250.0,500.0,sensdict['t_s']*v_C_0/1000)
    (omeg,mydict,myparams) = Icont1.makespectrums(range_gates,angles[0],sensdict['BeamWidth'],sensdict)
    Icont1.savemat('test.mat')