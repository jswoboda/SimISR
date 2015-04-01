#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""

#imported basic modules
import os, inspect, time
import pdb
# Imported scipy and matplotlib modules
import scipy as sp
import scipy.fftpack as scfft
import scipy.optimize,scipy.interpolate
from matplotlib import rc
import matplotlib.pylab as plt
# My modules
from IonoContainer import MakeTestIonoclass
from radarData import RadarData
from ISSpectrum import ISSpectrum
import const.sensorConstants as sensconst
from utilFunctions import make_amb, spect2acf

class Fitterionoconainer(object):
    def __init__(self,DataLags,NoiseLags,sensdict,simparams):
        """ The init function for the fitter take the inputs for the fitter programs.

            Inputs:
            DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            sensdict: The dictionary that holds the sensor info.
            simparams: The dictionary that hold the specific simulation params"""

        self.DataDict = DataLags
        self.NoiseDict = NoiseLags
        self.sensdict = sensdict
        self.simparams = simparams
    def fitNE(self,Tratio = 1):
        """ This funtction will fit electron density assuming Te/Ti is constant
        thus only the zero lag will be needed.
        Inputs:
        Tratio: Optional  a scaler for the Te/Ti.
        Outputs:
        Ne: A numpy array that is NtxNbxNrg, Nt is number of times, Nb is number
        of beams and Nrg is number of range gates."""
        pulsewidth = self.sensdict['taurg']*self.sensdict['t_s']
        txpower = self.sensdict['Pt']
        DataLags = self.DataDict
        (Nt,Nbeams,Nrng) = DataLags['Pow'].shape
        power = DataLags['Pow']
        pulses = sp.repeat(DataLags['Pulses'][:,:,sp.newaxis],Nrng,axis=-1)
        power = power/pulses
        rng_vec = self.sensdict['RG']*1e3
        rng3d = sp.tile(rng_vec,(Nt,Nbeams,1))
        Ksysvec = self.sensdict['Ksys'] # Beam shape and physcial constants
        ksys3d = sp.tile(Ksysvec[sp.newaxis,:,sp.newaxis],(Nt,1,Nrng))

        Ne = power*rng3d*rng3d/(pulsewidth*txpower*ksys3d)*2.0
        return Ne
    def fitdata2(self,npts=64,numtype = sp.complex128,startvalfunc=defstart,d_funcfunc = default_fit_func2,fitfunc=deffitfunc2):
        """ """

        # get intial guess for NE
        Ne_start =self.fitNE()

        sumrule = self.simparams['SUMRULE']
        minrg = -sp.min(sumrule[0])
        maxrg = len(self.sensdict['RG'])-sp.max(sumrule[1])
        Nrng2 = maxrg-minrg;
        # get the data nd noise lags
        lagsData= self.DataDict['ACF']
        (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
        pulses = sp.tile(self.DataDict['Pulses'][:,:,sp.newaxis,sp.newaxis],(1,1,Nrng,Nlags))

        # average by the number of pulses
        lagsData = lagsData/pulses
        lagsNoise=self.NoiseDict['ACF']
        lagsNoise = sp.mean(lagsNoise,axis=2)
        pulsesnoise = sp.tile(self.NoiseDict['Pulses'][:,:,sp.newaxis],(1,1,Nlags))
        lagsNoise = lagsNoise/pulsesnoise
        lagsNoise = sp.tile(lagsNoise[:,:,sp.newaxis,:],(1,1,Nrng,1))
        # subtract out noise lags
        lagsData = lagsData-lagsNoise

        # normalized out parameters
        pulsewidth = self.sensdict['taurg']*self.sensdict['t_s']
        txpower = self.sensdict['Pt']
        rng_vec = self.sensdict['RG']*1e3
        rng3d = sp.tile(rng_vec[sp.newaxis,sp.newaxis,:,sp.newaxis],(Nt,Nbeams,1,Nlags))
        Ksysvec = self.sensdict['Ksys']
        ksys3d = sp.tile(Ksysvec[sp.newaxis,:,sp.newaxis,sp.newaxis],(Nt,1,Nrng,Nlags))
        lagsData = lagsData*rng3d*rng3d/(pulsewidth*txpower*ksys3d)
        Pulse_shape = self.simparams['Pulse']
        fittedarray = sp.zeros((Nt,Nbeams,Nrng2,nparams))
        fittederror = sp.zeros((Nt,Nbeams,Nrng2,nparams,nparams))
        #self.simparams['Rangegatesfinal'] = sp.zeros(Nrng2)
        self.simparams['Rangegatesfinal'] = sp.array([ sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1]) for irng in sp.arange(minrg,maxrg)])
        print('\nData Now being fit.')
        for itime in sp.arange(Nt):
            print('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt))
            for ibeam in sp.arange(Nbeams):
                for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):

                   # self.simparams['Rangegatesfinal'][irngnew] = sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1])
                    curlag = sp.array([sp.mean(lagsData[itime,ibeam,irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag]) for ilag in sp.arange(Nlags)])#/sumreg
                    d_func = d_funcfunc(curlag, Pulse_shape,self.simparams['amb_dict'],self.sensdict,numtype)
                    x_0 = startvalfunc(Ne_start[itime,ibeam,irng])

                    try:
                        (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,x0=x_0,args=d_func,full_output=True)

                        fittedarray[itime,ibeam,irngnew] = x
                        if cov_x == None:
                            fittederror[itime,ibeam,irngnew] = sp.ones((len(x_0),len(x_0)))*float('nan')
                        else:
                            fittederror[itime,ibeam,irngnew] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                    except TypeError:
                        pdb.set_trace()

                print('\t\tData for Beam {0:d} of {1:d} fitted.'.format(ibeam,Nbeams))
        return(fittedarray,fittederror)