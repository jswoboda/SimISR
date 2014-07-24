#!/usr/bin/env python
"""
Created on Tue Jul 22 16:18:21 2014

@author: Bodangles
"""

import numpy as np
import scipy as sp
from physConstants import v_C_0

# utility functions
def make_amb(Fsorg,m_up,plen,nlags):
    """ Make the ambiguity function dictionary that holds the lag ambiguity and 
    range ambiguity. Uses a sinc function weighted by a blackman window. Currently
    only set up for an uncoded pulse.
    Inputs:
    Fsorg: A scalar, the original sampling frequency in Hertz.
    m_up: The upsampled ratio between the original sampling rate and the rate of 
    the ambiguity function up sampling.
    plen: The length of the pulse in samples at the original sampling frequency.
    nlags: The number of lags used.
    Outputs:
    Wttdict: A dictionary with the keys 'WttAll' which is the full ambiguity function
    for each lag, 'Wtt' is the max for each lag for plotting, 'Wrange' is the 
    ambiguity in the range with the lag dimension summed, 'Wlag' The ambiguity 
    for the lag, 'Delay' the numpy array for the lag sampling, 'Range' the array 
    for the range sampling.
    """
    
    # make the sinc
    nsamps = np.floor(8.5*m_up)
    nsamps = nsamps-(1-np.mod(nsamps,2))
    
    nvec = np.arange(-np.floor(nsamps/2.0),np.floor(nsamps/2.0)+1)
    outsinc = np.blackman(nsamps)*np.sinc(nvec/m_up)
    outsinc = outsinc/np.sum(outsinc)
    dt = 1/(Fsorg*m_up)
    Delay = np.arange(-(len(nvec)-1),m_up*(nlags+5))*dt
    t_rng = np.arange(0,1.5*plen,dt)
    numdiff = len(Delay)-len(outsinc)
    outsincpad  = np.pad(outsinc,(0,numdiff),mode='constant',constant_values=(0.0,0.0))
    (srng,d2d)=np.meshgrid(t_rng,Delay)
    # envelop function    
    envfunc = np.zeros(d2d.shape)
    envfunc[(d2d-srng+plen-Delay.min()>=0)&(d2d-srng+plen-Delay.min()<=plen)]=1
    #pdb.set_trace()
    envfunc = envfunc/np.sqrt(envfunc.sum(axis=0).max())
    #create the ambiguity function for everything
    Wtt = np.zeros((nlags,d2d.shape[0],d2d.shape[1]))
    cursincrep = np.tile(outsincpad[:,np.newaxis],(1,d2d.shape[1]))
    Wt0 = Wta = cursincrep*envfunc
    Wt0fft = np.fft.fft(Wt0,axis=0)
    for ilag in np.arange(nlags):
        cursinc = np.roll(outsincpad,ilag*m_up)
        cursincrep = np.tile(cursinc[:,np.newaxis],(1,d2d.shape[1]))
        Wta = cursincrep*envfunc
        #do fft based convolution, probably best method given sizes
        Wtafft = np.fft.fft(Wta,axis=0)
        if ilag==0:
            nmove = len(nvec)-1
        else:
            nmove = len(nvec)
        Wtt[ilag] = np.roll(np.fft.ifft(Wtafft*np.conj(Wt0fft),axis=0).real,nmove,axis=0)
    Wttdict = {'WttAll':Wtt,'Wtt':Wtt.max(axis=0),'Wrange':Wtt.sum(axis=1),'Wlag':Wtt.sum(axis=2),'Delay':Delay,'Range':v_C_0*t_rng/2.0}
    return Wttdict
    
def spect2acf(omeg,spec):
    """ Creates acf and time array associated with the given frequency vector and spectrum
    Inputs:
    omeg: The frequency sampling vector
    spec: The spectrum array.
    Output:
    tau: The time sampling array.
    acf: The acf from the original spectrum.""" 
    padnum = np.floor(len(spec)/2)
    df = omeg[1]-omeg[0]
    
    specpadd = np.pad(spec,(padnum,padnum),mode='constant',constant_values=(0.0,0.0))
    acf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(specpadd)))
    dt = 1/(df*len(specpadd))
    tau = np.arange(-np.floor(len(acf)/2),np.floor(len(acf)/2+1))*dt
    return tau, acf
    
    
def MakePulseData(pulse_shape, filt_freq, delay=16):
    """ This function will create a pulse width of data shaped by the filter that who's frequency
        response is passed as the parameter filt_freq.  The pulse shape is delayed by the parameter
        delay into the data. The noise vector that will be multiplied by the filter's frequency
        response will be zero mean complex white Gaussian noise with a power of 1. The user
        then will need to scale their filter to get the desired power out.
        Inputs:
            pulse_shape: A numpy array that holds the shape of the single pulse.
            filt_freq - a numpy array that holds the complex frequency response of the filter
            that will be used to shape the noise data.
            delay - The number of samples that the pulse will be delayed into the 
            array of noise data to avoid any problems with filter overlap.
    """
    npts = len(filt_freq)
    
    noise_vec = (np.random.randn(npts)+1j*np.random.randn(npts))/np.sqrt(2.0)# make a noise vector
    mult_freq = filt_freq*noise_vec
    data = np.fft.ifft(mult_freq)
    data_out = pulse_shape*data[delay:(delay+len(pulse_shape))]
    return data_out

def CenteredLagProduct(rawbeams,N =14,numtype=np.complex128):
    """ This function will create a centered lag product for each range using the
    raw IQ given to it.  It will form each lag for each pulse and then integrate 
    all of the pulses.
    Inputs: 
        rawbeams - This is a NsxNpu complex numpy array where Ns is number of 
        samples per pulse and Npu is number of pulses
        N - The number of lags that will be created, default is 14.
        numtype - The type of numbers used to create the data. Default is np.complex128
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
    ap = np.max(abs(arback));
    ep = Nr- np.max(arfor);
    rng_ar_all = np.arange(ap,ep);
    #acf_cent = np.zeros((ep-ap,N))*(1+1j)
    acf_cent = np.zeros((ep-ap,N),dtype=numtype)
    for irng in  np.arange(len(rng_ar_all)):
        rng_ar1 =np.int(rng_ar_all[irng]) + arback
        rng_ar2 = np.int(rng_ar_all[irng]) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = np.conj(rawbeams[rng_ar1,:])*rawbeams[rng_ar2,:]
        acf_ave = np.sum(acf_tmp,1)
        acf_cent[irng,:] = acf_ave# might need to transpose this
    return acf_cent    
    
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
# Test functions    
def TempProfile(z):
    """This function creates a tempreture profile for test purposes."""
    
    Te = ((45.0/500.0)*(z-200.0))**2+1000.0
    Ti = ((20.0/500.0)*(z-200.0))**2+1000.0
    return (Te,Ti)