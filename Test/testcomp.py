#!/usr/bin/env python
"""
Created on Wed Jul  1 10:29:20 2015

@author: John Swoboda
"""
from SimISR import Path
import scipy as sp
import scipy.fftpack as scfft
import scipy.interpolate as spinterp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")

from SimISR.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf
from ISRSpectrum.ISRSpectrum import ISRSpectrum

inifile= "/Users/Bodangles/Documents/Python/SimISR/Testdata/PFISRExample.pickle"

def main():

    (sensdict,simparams) = readconfigfile(inifile)
    simdtype = simparams['dtype']
    sumrule = simparams['SUMRULE']
    npts = simparams['numpoints']
    amb_dict = simparams['amb_dict']
    # for spectrum
    ISS2 = ISRSpectrum(centerFrequency = 440.2*1e6, bMag = 0.4e-4, nspec=npts, sampfreq=sensdict['fs'],dFlag=True)
    ti = 2e3
    te = 2e3
    Ne = 1e11
    Ni = 1e11


    datablock90 = sp.array([[Ni,ti],[Ne,te]])
    species = simparams['species']

    (omega,specorig,rcs) = ISS2.getspecsep(datablock90, species,rcsflag = True)

    cur_filt = sp.sqrt(scfft.ifftshift(specorig*npts*npts*rcs/specorig.sum()))
    #for data
    Nrep = 10000
    pulse = sp.ones(14)
    lp_pnts = len(pulse)
    N_samps = 100
    minrg = -sumrule[0].min()
    maxrg = N_samps+lp_pnts-sumrule[1].max()
    Nrng2 = maxrg-minrg;
    out_data = sp.zeros((Nrep,N_samps+lp_pnts),dtype=simdtype)
    samp_num = sp.arange(lp_pnts)
    for isamp in range(N_samps):
        cur_pnts = samp_num+isamp
        cur_pulse_data = MakePulseDataRep(pulse,cur_filt,rep=Nrep)
        out_data[:,cur_pnts] = cur_pulse_data+out_data[:,cur_pnts]

    lagsData = CenteredLagProduct(out_data,numtype=simdtype,pulse =pulse)
    lagsData=lagsData/Nrep # divide out the number of pulses summed
    Nlags = lagsData.shape[-1]
    lagsDatasum = sp.zeros((Nrng2,Nlags),dtype=lagsData.dtype)
    for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
        for ilag in range(Nlags):
            lagsDatasum[irngnew,ilag] = lagsData[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)
    lagsDatasum=lagsDatasum/lp_pnts # divide out the pulse length
    (tau,acf) = spect2acf(omega,specorig)

    # apply ambiguity function
    tauint = amb_dict['Delay']
    acfinterp = sp.zeros(len(tauint),dtype=simdtype)

    acfinterp.real =spinterp.interp1d(tau,acf.real,bounds_error=0)(tauint)
    acfinterp.imag =spinterp.interp1d(tau,acf.imag,bounds_error=0)(tauint)
    # Apply the lag ambiguity function to the data
    guess_acf = sp.zeros(amb_dict['Wlag'].shape[0],dtype=sp.complex128)
    for i in range(amb_dict['Wlag'].shape[0]):
        guess_acf[i] = sp.sum(acfinterp*amb_dict['Wlag'][i])

#    pdb.set_trace()
    guess_acf = guess_acf*rcs/guess_acf[0].real

    # fit to spectrums
    spec_interm = scfft.fftshift(scfft.fft(guess_acf,n=npts))
    spec_final = spec_interm.real
    allspecs = scfft.fftshift(scfft.fft(lagsDatasum,n=len(spec_final),axis=-1),axes=-1)
#    allspecs = scfft.fftshift(scfft.fft(lagsDatasum,n=npts,axis=-1),axes=-1)
    fig = plt.figure()
    plt.plot(omega,spec_final.real,label='In',linewidth=5)
    plt.hold(True)
    plt.plot(omega,allspecs[40].real,label='Out',linewidth=5)
    plt.axis((omega.min(),omega.max(),0.0,2e11))
    plt.show(False)
if __name__== '__main__':

    main()
