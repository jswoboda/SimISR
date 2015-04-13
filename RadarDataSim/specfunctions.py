#!/usr/bin/env python
"""
Created on Mon Mar 16 19:36:27 2015

@author: John Swoboda
"""
import scipy as sp
import scipy.interpolate as spinterp
import scipy.fftpack as scfft
from ISRSpectrum.ISRSpectrum import ISRSpectrum
from utilFunctions import  spect2acf


def ISRSspecmake(ionocont,sensdict,npts):
    Vi = ionocont.getDoppler()
    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])

    paramshape = ionocont.Param_List.shape
    if ionocont.Time_Vector is None:
        outspecs = sp.zeros((paramshape[0],1,npts))
        full_grid = False
    else:
        outspecs = sp.zeros((paramshape[0],paramshape[1],npts))
        full_grid = True

    (N_x,N_t) = outspecs.shape[:2]
    #pdb.set_trace()
    for i_x in sp.arange(N_x):
        for i_t in sp.arange(N_t):
            if full_grid:
                cur_params = ionocont.Param_List[i_x,i_t]
                cur_vel = Vi[i_x,i_t]
            else:
                cur_params = ionocont.Param_List[i_x]
            (omeg,cur_spec,rcs) = specobj.getspecsep(cur_params,ionocont.Species,cur_vel,rcsflag=True)
            cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
            outspecs[i_x,i_t] = cur_spec_weighted

    return (omeg,outspecs,npts)

def ISRSfitfunction(x,y_acf,amb_func,amb_dict,sensdict,npts,numtype):

    specs = sensdict['species']
    nspecs = len(specs)
    datablock = sp.zeros((nspecs,2),dtype=numtype)
    datablock[:,0] = x[sp.arange(0,nspecs*2,2)]
    datablock[:,1] = x[sp.arange(1,nspecs*2,2)]
    v_i = x[-1]

    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])
    (omeg,cur_spec,rcs) = specobj.getspecsep(datablock,specs,v_i,rcsflag=True)
    # Create spectrum guess
    (tau,acf) = spect2acf(omeg,cur_spec)

    # apply ambiguity function
    tauint = amb_dict['Delay']
    acfinterp = sp.zeros(len(tauint),dtype=numtype)
    acfinterp.real =spinterp.interp1d(tau,acf.real,bounds_error=0)(tauint)
    acfinterp.imag =spinterp.interp1d(tau,acf.imag,bounds_error=0)(tauint)
    # Apply the lag ambiguity function to the data
    guess_acf = sp.zeros(amb_dict['Wlag'].shape[0],dtype=sp.complex128)
    for i in range(amb_dict['Wlag'].shape[0]):
        guess_acf[i] = sp.sum(acfinterp*amb_dict['Wlag'][i])

    guess_acf = guess_acf*rcs/guess_acf[0].real
    # fit to spectrums
    spec_interm = scfft.fft(guess_acf,n=len(cur_spec))
    spec_final = spec_interm.real
    y_interm = scfft.fft(y_acf,n=len(spec_final))
    y = y_interm.real
    return y-spec_final