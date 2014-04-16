#!/usr/bin/env python
"""
Created on Fri Jan 10 12:09:44 2014

@author: Bodangles
"""
# Imported modules
import numpy as np
import scipy as sp
import scipy.optimize
import time
import os
from matplotlib import rc, rcParams
import matplotlib.pylab as plt

# debub
import pdb
# My modules
from IonoContainer import IonoContainer, MakeTestIonoclass
from radarData import RadarData
from ISSpectrum import ISSpectrum
from const.physConstants import *
from const.sensorConstants import *



class FitterBasic(object):
    """ This is a basic fitter to take data created by the RadarData Class and 
    make fitted Data"""
    def init(self,fit_func=default_fit_func,init_vals=Init_vales):
        """ The init funciton for the fitter will determine the type of  """
        
        self.fit_func = fit_func
        self.init_func = init_vals
    def fitdata(self,icode):
        """This function will fit data for a single radar beam and output a numpy
        array.  The instance self will be adjusted.
        Inputs:
            self - An instance of the RadarData class.
            icode - This is simply an integer to select which beam will have
                data created for it.
        """
        fit_func = self.fit_func
        init_vals_func = self.init_func
        
        angles = self.simparams['angles']
        n_beams = len(angles)
        rawIQ = self.datadict[icode]
        IPP = self.simparams['IPP']
        timelim = self.simparams['TimeLim']
        PIPP = n_beams*IPP
        beg_time = icode*IPP
        Pulse_shape =  self.simparams['Pulse']
        plen = len(Pulse_shape)
        sensdict = self.sensdict
        npnts = 64
        # need to change this should not reference assumptions
        rm1 = 16# atomic weight of species 1
        rm2 = 1# atomic weight os species 2
        p2 =0 #
        
        # loop to figure out all of the pulses in a time period
        pulse_times = np.arange(beg_time,timelim,PIPP)
        
        int_timesbeg = self.simparams['Timevec']
        int_times = np.append(int_timesbeg,timelim)
        int_timesend = int_times[1:]
        numpast = 0
        Pulselims = np.zeros((len(int_timesbeg),2))
        N_pulses = np.zeros_like(int_timesbeg)
        #make noise lags
        noisedata = self.noisedata[icode]
        noiselags = CenteredLagProduct(noisedata,plen)
        NNp = self.simparams['Noisepulses']
        NNs = noiselags.shape[0]
        final_noise_lags = noiselags.sum(0)
        
        # loop to determine what pulses are kep for each period         
        for itime in np.arange(len(int_timesbeg)):
            self.noiselag[itime,icode] = final_noise_lags
            Pulselims[itime,0] = numpast
            keep_pulses = (pulse_times>=int_timesbeg[itime]) & (pulse_times<int_timesend[itime])
            N_pulses[itime] = keep_pulses.sum()
            numpast+= N_pulses[itime]
            Pulselims[itime,1] = numpast
            # loop for fitting
            cur_raw = rawIQ[:,Pulselims[itime,0]:Pulselims[itime,1]]
            
            out_lags = CenteredLagProduct(cur_raw,plen)
            (N_gates,N_lags) = out_lags.shape
            self.lagarray[itime,icode] = out_lags
            for irng in np.arange(N_gates):
                rangem=sensdict['RG'][irng]*1e3
                curlag = out_lags[irng]                
                noise_denom = final_noise_lags*(N_pulses[itime]/(NNp*NNs))
                normedlag = curlag-noise_denom
                d_func = (normedlag, Pulse_shape,rm1,rm2,p2,sensdict,rangem,npnts,N_pulses[itime]) 
                x_0 = init_vals_func()
                #pdb.set_trace()
                try:                
                    (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fit_func,x0=x_0,args=d_func,full_output=True)
                                
                    self.fittedarray[itime,icode,irng] = x
                    if cov_x == None:
                        self.fittederror[itime,icode,irng] = np.ones((len(x_0),len(x_0)))*float('nan')
                    else:
                        self.fittederror[itime,icode,irng] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                except TypeError:
                    pdb.set_trace()
                
    def fitalldata(self):
        """This method will fun the fitdata method on all of the beams."""
        print('Data Now being fit.\n\n')
        for ibeam in np.arange(len(self.simparams['angles'])):
        
            self.fitdata(ibeam)
            print('Data from beam {0:d} fitted'.format(ibeam))

def CenteredLagProduct(rawbeams,N =14):
    """ This function will create a centered lag product for each range using the
    raw IQ given to it.  It will form each lag for each pulse and then integrate 
    all of the pulses.
    Inputs: 
        rawbeams - This is a NsxNpu complex numpy array where Ns is number of 
        samples per pulse and Npu is number of pulses
        N - The number of lags that will be created, default is 14
    Output:
        acf_cent - This is a NrxNl complex numpy array where Nr is number of 
        range gate and Nl is number of lags.
    """
    # It will be assumed the data will be range vs pulses    
    (Nr,Np) = rawbeams.shape    
    
    # Make masks for each piece of data
    arex = np.arange(0,N/2.0,0.5);
    arback = np.array([-np.int(np.floor(k)) for k in arex]);
    arfor = np.array([np.int(np.ceil(k)) for k in arex]) ;
    
    # figure out how much range space will be kept
    sp = np.max(abs(arback));
    ep = Nr- np.max(arfor);
    rng_ar_all = np.arange(sp,ep);
    #acf_cent = np.zeros((ep-sp,N))*(1+1j)
    acf_cent = np.zeros((ep-sp,N),dtype=np.complex128)
    for irng in  np.arange(len(rng_ar_all)):
        rng_ar1 =np.int(rng_ar_all[irng]) + arback
        rng_ar2 = np.int(rng_ar_all[irng]) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = np.conj(rawbeams[rng_ar1,:])*rawbeams[rng_ar2,:]
        acf_ave = np.sum(acf_tmp,1)
        acf_cent[irng,:] = acf_ave# might need to transpose this
    return acf_cent                
def Init_vales():
    return np.array([1000.0,1500.0,10**11])    
def default_fit_func(x,y_acf,amb_func,rm1,rm2,p2,sensdict,range,npts,Npulses):
    """ This is the default fit function used by the least squares command."""
    ti = x[0]
    te = x[1]
    Ne = x[2]    
    if te<0:
        te=-te
    if ti <0:
        ti=-ti
    if Ne<=0:
        Ne=-Ne
    po = np.log10(Ne)
    tr = te/ti
    myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
    (omeg,cur_spec) = myspec.getSpectrum(ti, tr, po, rm1, rm2, p2)    
    
    # form the acf
    guessacf = np.fft.ifft(np.fft.ifftshift(cur_spec))
    L_amb = len(amb_func)
    full_amb = np.concatenate((amb_func,np.zeros(len(guessacf)-L_amb)))
    acf_mult = guessacf*full_amb
    # Add the change in power from the sensor
    pow_num = sensdict['Pt']*sensdict['G']*v_C_0*sensdict['lamb']**2*Npulses*sensdict['taurg']
    pow_den = 2*16*np.pi**2*range**2
    rcs = v_electron_rcs*Ne/((1+tr))    
    # multiply the power
    acf_mult = (acf_mult/np.abs(acf_mult[0]))*rcs*pow_num/pow_den
    spec_interm = np.fft.fft(acf_mult)
    spec_final = spec_interm.real
    y_interm = np.fft.fft(y_acf,n=len(spec_final))
    y = y_interm.real    
    #y = y/np.sum(y)
    return y-spec_final
    
if __name__== '__main__':
    """ This is a test for the fitter class"""
