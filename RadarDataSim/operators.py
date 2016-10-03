#!/usr/bin/env python
"""
Created on Tue Dec 29 15:49:01 2015

@author: John Swoboda
"""

import tables
from const.physConstants import v_C_0
from utilFunctions import readconfigfile
import scipy.fftpack as scfft
from IonoContainer import IonoContainer,makeionocombined
from RadarDataSim.utilFunctions import spect2acf
import scipy as sp
import pdb
class RadarSpaceTimeOperator(object):
    """ This is a class to hold the operator methods for the ISR simulator.
    
        Variables:
           Cart_Cords_In - Ntbegx3 array of cartisian coordinates for the input space.
           Sphere_Cords_In - Ntbegx3 array of spherical coordinates for the input space.
           Cart_Cords_Out - Ntoutx3 array of cartisian coordinates for the output space.
           Sphere_Cords_Out - Ntoutx3 array of spherical coordinates for the output space.
           Time_In - A Ntbegx2 numpy array with the start and stop times of the input data.
           Time_Out - A Ntoutx2 numpy array with the start and stop times of the output data.
           RSTMat - A list of matricies or a single matrix that is the forward between physical space
               to the discrete samples space of the radar.
           blocks - A tuple that holds the number of block matricies in overall forward operator.
           blocksize - A tuple that holds the shape of the outmatrix size.
           blockloc - An Ntout x Ntbeg array that holds the corresponding spatial forward model.
    """
    def __init__(self,ionoin,configfile,timein=None,mattype='matrix'):
        """ This will create the RadarSpaceTimeOperator object.
            Inputs
                ionoin - The input ionocontainer. This can be either an string that is a ionocontainer file,
                    a list of ionocontainer objects or a list a strings to ionocontainer files
                config  - The ini file that used to set up the simulation.
                timein - A Ntx2 numpy array of times.
                RSTOPinv - The inverse operator object.
                invmat - The inverse matrix to the original operator.
        """
            
        d2r = sp.pi/180.0
        (sensdict,simparams) = readconfigfile(configfile)
        # determine if the input ionocontainer is a string, a list of strings or a list of ionocontainers.
        ionoin=makeionocombined(ionoin)
        #Input location
        self.Cart_Coords_In = ionoin.Cart_Coords
        self.Sphere_Coords_In = ionoin.Sphere_Coords

        # Set the input times
        if timein is None:
            self.Time_In = ionoin.Time_Vector
        else:
            self.Time_In = timein

        #Create an array of output location based off of the inputs
        rng_vec2 = simparams['Rangegatesfinal']
        nrgout = len(rng_vec2)

        angles = simparams['angles']
        nang =len(angles)

        ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
        rng_all = sp.repeat(rng_vec2,(nang),axis=0)
        ang_all = sp.tile(ang_data,(nrgout,1))
        self.Sphere_Coords_Out = sp.column_stack((rng_all,ang_all))
        (R_vec,Az_vec,El_vec) = (self.Sphere_Coords_Out[:,0],self.Sphere_Coords_Out[:,1],
            self.Sphere_Coords_Out[:,2])
        xvecmult = sp.sin(Az_vec*d2r)*sp.cos(El_vec*d2r)
        yvecmult = sp.cos(Az_vec*d2r)*sp.cos(El_vec*d2r)
        zvecmult = sp.sin(El_vec*d2r)
        X_vec = R_vec*xvecmult
        Y_vec = R_vec*yvecmult
        Z_vec = R_vec*zvecmult
        
        self.Cart_Coords_Out = sp.column_stack((X_vec,Y_vec,Z_vec))
        self.Time_Out = sp.column_stack((simparams['Timevec'],simparams['Timevec']+simparams['Tint']))+self.Time_In[0,0]
        self.simparams=simparams
        self.sensdict=sensdict
        self.lagmat = self.simparams['amb_dict']['WttMatrix']
        # create the matrix
        (self.RSTMat,self.overlaps,self.blocklocs) = makematPA(ionoin.Sphere_Coords,ionoin.Cart_Coords,ionoin.Time_Vector,configfile,ionoin.Velocity,mattype)


    def mult_iono(self,ionoin_list):
        """ 
            This will apply the forward model to the contents of an ionocontainer object. It is assuming that 
            this is an ionocontainer holding the spectra.
        """

        ntout = self.Time_Out.shape[0]
        nlout = self.Cart_Coords_Out.shape[0]
        blist_in,blist_out = self.blocklocs
        amb_dict = self.simparams['amb_dict']
        ambmat = amb_dict['WttMatrix']
        overlaps = self.overlaps
        
        t_s = self.sensdict['t_s']
        
        if isinstance(ionoin_list,list)or isinstance(ionoin_list,str):
            
            Iono_in = makeionocombined(ionoin_list)
        else:
            Iono_in=ionoin_list
        
        
        ionocart = Iono_in.Cart_Coords
      
        if self.simparams['numpoints']==Iono_in.Param_List.shape[-1]:
            tau,acf=spect2acf(Iono_in.Param_Names,Iono_in.Param_List)
            np = ambmat.shape[0]
        else:
            acf=Iono_in.Param_List
            np = acf.shape[-1]
        np_in =acf.shape[-1]
        tau_out = t_s*sp.arange(np)
        outdata = sp.zeros((nlout,ntout,np),dtype=acf.dtype)
        assert sp.allclose(ionocart,self.Cart_Coords_In), "Spatial Coordinates need to be the same"

        for it_out in range(ntout):
            
            overlists = overlaps[it_out]
            irows = blist_out[it_out]
            curintimes = [i[0] for i in overlists]
            curintratio=[i[1] for i in overlists]
            cur_outmat = self.RSTMat[irows[0]:irows[1],:]
            icols=    blist_in[it_out]
            cur_mat = cur_outmat[:,icols[0]:icols[1]]
            
            for i_it,it_in in enumerate(curintimes):
                tempdata=sp.zeros((np_in,nlout),dtype=acf.dtype)
                for iparam in range(np_in):
                   tempdata[iparam]=cur_mat.dot(acf[:,it_in,iparam])
                if self.simparams['numpoints']==Iono_in.Param_List.shape[-1]:
                    tempdata=sp.dot(ambmat,tempdata)
                
                outdata[:,it_out] = sp.transpose(tempdata)*curintratio[i_it] + outdata[:,it_out]

        outiono = IonoContainer(self.Sphere_Coords_Out,outdata,times=self.Time_Out,sensor_loc=Iono_in.Sensor_loc,
                               ver=1,coordvecs = ['r','theta','phi'],paramnames=tau_out)
        return outiono

    

def makematPA(Sphere_Coords,Cart_Coords,timein,configfile,vel=None,mattype='matrix'):
    """Make a Ntimeout*Nbeam*Nrng x Ntime*Nloc sparse matrix for the space time operator. 
       The output space will have range repeated first, then beams then time. 
       The coordinates will be [t0,b0,r0],[t0,b0,r1],[t0,b0,r2],...
       [t0,b1,r0],[t0,b1,r1], ... [t1,b0,r0],[t1,b0,r1],...[t1,b1,r0]...
       Inputs
           Sphere_Coords - A Nlocx3 array of the spherical coordinates of the input data.
           timein - A Ntbegx2 numpy array with the start and stop times of the input data.
           configfile - The ini file used for the simulation configuration.
           vel - A NlocxNtx3 numpy array of velocity.
       Outputs
           outmat - A list of matricies or a single matrix that is the forward between physical space
               to the discrete samples space of the radar.
           blocks - A tuple that holds the number of block matricies in overall forward operator.
           blocksize - A tuple that holds the shape of the outmatrix size.
           blockloc - An Ntout x Ntbeg array that holds the corresponding spatial forward model.
    """
    #
    (sensdict,simparams) = readconfigfile(configfile)
    timeout = simparams['Timevec']
    Tint = simparams['Tint']
    timeout = sp.column_stack((timeout,timeout+Tint)) +timein[0,0]

    rng_bin=sensdict['t_s']*v_C_0*1e-3/2.
    
    angles = simparams['angles']
    Nbeams = len(angles)

    
    rng_vec2 = simparams['Rangegatesfinal']
    nrgout = len(rng_vec2)
    pulse=simparams['Pulse']
    p_samps = pulse.shape[0]
    rng_len=p_samps*rng_bin
    Nlocbeg = Cart_Coords.shape[0]
    Nlocout= Nbeams*nrgout
    Ntbeg = len(timein)
    Ntout = len(timeout)
    if vel is None:
        vel=sp.zeros((Nlocbeg,Ntbeg,3))
    # set up blocks
    blocksize = (Ntout*Nbeams*nrgout,Nlocbeg*Ntout)

    
    # make the matrix
    outmat = sp.sparse.lil_matrix(blocksize,dtype =sp.float64)
    # determine the overlaps
    # change
    blockloc_in = [[i*Nlocbeg,(i+1)*Nlocbeg] for i in range(Ntout)]
    blockloc_out = [[i*Nlocout,(i+1)*Nlocout] for i in range(Ntout)]
    blockloc = [blockloc_in,blockloc_out]
    overlaps={}
    if mattype=='real':
        cor_ratio=.5
    else:
        cor_ratio=1.
    for iton,ito in enumerate(timeout):
        overlaps[iton]=[]
        firstone=True
        for ix, x in enumerate(timein):
            if ( x[0]+1<ito[1] or ix==(Ntbeg-1)) and x[1]-1>ito[0]:
                
                # find the end of the data
                if ix ==Ntbeg-1:
                    enp=ito[1]
                else:
                    enp = sp.minimum(ito[1],x[1])
                stp = sp.maximum(x[0],ito[0])
                if firstone:
                    firstone=False
                    t_0 = stp.copy()
                    curvel=vel[:,ix]*1e-3
                    curdiff=sp.zeros_like(curvel)
                    curdiff2=curdiff+curvel*enp
                else:
                    T_1=float(x[0]-t_0)   
                    t_0=stp.copy()
                    curdiff= curdiff+T_1*curvel
                    curvel=vel[:,ix]*1e-3
                    curdiff2=curdiff+enp*curvel
                #find amount of time for overlap
                ratio = float(enp-stp)/Tint
                # set up new coordinate system
                # The thee types of coordinates are as follows
                # The matrix type assumes that the matrix will be applied to 
                if mattype=='matrix':
                    newcoorsds1 = cart2sphere(Cart_Coords)
                    newcoorsds2 = cart2sphere(Cart_Coords)
                elif mattype=='real':
                    newcoorsds1 = cart2sphere(Cart_Coords-curdiff)
                    newcoorsds2 = cart2sphere(Cart_Coords-curdiff2)
                else:
                    newcoorsds1 = cart2sphere(Cart_Coords-curdiff)
                    newcoorsds2 = cart2sphere(Cart_Coords-curdiff)
                overlaps[iton].append([ix,ratio,newcoorsds1,newcoorsds2])
    # make the matrix
    for iton,ito in enumerate(timeout):
        cur_over = overlaps[iton]
        if mattype=='matrix':
            cur_over=[cur_over[0]]
            cur_over[0][1]=1.
        for it_in,it_info in enumerate(cur_over):
            
            cur_it,cur_ratio,Sp1,Sp2 = it_info
            rho1 = Sp1[:,0]
            Az1 = Sp1[:,1]
            El1 = Sp1[:,2]
            rho2 = Sp2[:,0]
            Az2 = Sp2[:,1]
            El2 = Sp2[:,2]
            # get the weights
            weights1 = {ibn:sensdict['ArrayFunc'](Az1,El1,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}
            weights2 = {ibn:sensdict['ArrayFunc'](Az2,El2,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}
    
            for ibn in range(Nbeams):
                

                print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
                weight1 = weights1[ibn]
                weight2 = weights2[ibn]
                for isamp in range(nrgout):
                    # make the row
                    irow = ibn + isamp*Nbeams + Nbeams*nrgout*iton
                    range_g = rng_vec2[isamp]
                    rnglims = [range_g-rng_len/2.,range_g+rng_len/2.]
                    
                    # assume centered lag product.
                    rangelog = ((rho1>=rnglims[0])&(rho1<rnglims[1]))
                    # This is a nearest neighbors interpolation for the spectrums in the range domain
                    if sp.sum(rangelog)==0:
                        minrng = sp.argmin(sp.absolute(range_g-rho1))
                        rangelog[minrng] = True
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight1[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()
                    icols = sp.where(rangelog)[0] + Nlocbeg*iton
#                    icols = sp.where(rangelog)[0] + Nlocbeg*cur_it

                    weights_final = weight_cur*range_g**2/rho1[rangelog]**2
                    outmat[irow,icols] = weights_final*cur_ratio*cor_ratio+outmat[irow,icols]
                    
                    if mattype=='real':
                        # assume centered lag product.
                        rangelog = ((rho2>=rnglims[0])&(rho2<rnglims[1]))
                        # This is a nearest neighbors interpolation for the spectrums in the range domain
                        if sp.sum(rangelog)==0:
                            minrng = sp.argmin(sp.absolute(range_g-rho2))
                            rangelog[minrng] = True
                        #create the weights and weight location based on the beams pattern.
                        weight_cur =weight2[rangelog]
                        weight_cur = weight_cur/weight_cur.sum()
    #                    icols = sp.where(rangelog)[0]+ Nlocbeg*cur_it
                        icols = sp.where(rangelog)[0]+ Nlocbeg*iton
                        weights_final = weight_cur*range_g**2/rho2[rangelog]**2
                        outmat[irow,icols] = weights_final*cur_ratio*cor_ratio+outmat[irow,icols]
                
                



    return(outmat,overlaps,blockloc)


def saveoutmat(filename,Sphere_Coords,timein,configfile):
    """ This will save the outmatrix into an h5 file.
    """
    outmat = makematPA(Sphere_Coords,timein,configfile)
    h5file = tables.openFile(filename, mode = "w", title = "Radar Matrix out.")
    h5file.createArray('/','RadarMatrix',outmat,'Static array')
    h5file.close()

def readradarmat(filename):
    """ This will read in a precomuted matrix form an h5 file.
    """
    h5file=tables.openFile(filename)
    outmat = h5file.root.RadarMatrix.read()
    h5file.close()
    return outmat
#%% extra functions
def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

#%% Math Functions

def cart2sphere(coordlist):
    r2d = 180.0/sp.pi
    d2r = sp.pi/180.0
    
    X_vec = coordlist[:,0]
    Y_vec = coordlist[:,1]
    Z_vec = coordlist[:,2]
    R_vec = sp.sqrt(X_vec**2+Y_vec**2+Z_vec**2)
    Az_vec = sp.arctan2(X_vec,Y_vec)*r2d
    El_vec = sp.arcsin(Z_vec/R_vec)*r2d
    sp_coords = sp.array([R_vec,Az_vec,El_vec]).transpose()
    return sp_coords
    
    
    
