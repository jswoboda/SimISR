#!/usr/bin/env python
"""
specfunctions.py
This module holds the functions that deal with the spectrum formation functions like
fitting and making spectrums.
@author: John Swoboda
"""
import scipy as sp
import pdb
import scipy.fftpack as scfft
from ISRSpectrum.ISRSpectrum import ISRSpectrum
from utilFunctions import  spect2acf


def ISRSspecmake(ionocont,sensdict,npts):
    """This function will take an ionocontainer instance of plasma parameters and create
    ISR spectra for each object.

    Inputs
    ionocont - An instance of the ionocontainer class with plasma parameters. Its param list
    must an array of [Nl,Nt,Ni,2]."""
    Vi = ionocont.getDoppler()
    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])

    if ionocont.Time_Vector is None:
        N_x = ionocont.Param_List.shape[0]
        N_t = 1
        outspecs = sp.zeros((N_x,1,npts))
        full_grid = False
    else:
        (N_x,N_t) = ionocont.Param_List.shape[:2]
        outspecs = sp.zeros((N_x,N_t,npts))
        full_grid = True

    (N_x,N_t) = outspecs.shape[:2]
    outspecsorig = sp.zeros_like(outspecs)
    outrcs = sp.zeros((N_x,N_t))
    #pdb.set_trace()
    for i_x in sp.arange(N_x):
        for i_t in sp.arange(N_t):
            print('\t Time:{0:d} of {1:d} Location:{2:d} of {3:d}, now making spectrum.'.format(i_t,N_t,i_x,N_x))

            if full_grid:
                cur_params = ionocont.Param_List[i_x,i_t]
                cur_vel = Vi[i_x,i_t]
            else:
                cur_params = ionocont.Param_List[i_x]
            (omeg,cur_spec,rcs) = specobj.getspecsep(cur_params,ionocont.Species,cur_vel,rcsflag=True)
            specsum = sp.absolute(cur_spec).sum()
            cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/specsum
            outspecsorig[i_x,i_t] = cur_spec
            outrcs[i_x,i_t] = rcs
            outspecs[i_x,i_t] = cur_spec_weighted
    return (omeg,outspecs,npts)

def ISRspecmakeout(paramvals,fc,fs,species,npts):


    if paramvals.ndim==2:
        paramvals=paramvals[sp.newaxis]

    (N_x,N_t) = paramvals.shape[:2]
    Nsp = len(species)
    Vi = paramvals[:,:,2*Nsp]
    Parammat = paramvals[:,:,:2*Nsp].reshape((N_x,N_t,Nsp,2))
    outspecs=sp.zeros((N_x,N_t,npts))
    specobj = ISRSpectrum(centerFrequency =fc,nspec = npts,sampfreq=fs)
    outspecsorig = sp.zeros_like(outspecs)
    outrcs = sp.zeros((N_x,N_t))
    for i_x in sp.arange(N_x):
        for i_t in sp.arange(N_t):
            cur_params = Parammat[i_x,i_t]
            cur_vel = Vi[i_x,i_t]
            (omeg,cur_spec,rcs) = specobj.getspecsep(cur_params,species,cur_vel,rcsflag=True)
            specsum = sp.absolute(cur_spec).sum()
            cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/specsum
            outspecsorig[i_x,i_t] = cur_spec
            outrcs[i_x,i_t] = rcs
            outspecs[i_x,i_t] = cur_spec_weighted
    return (omeg,outspecs)
def ISRSfitfunction(x,y_acf,sensdict,simparams):
    """ """
    npts = simparams['numpoints']
    specs = simparams['species']
    amb_dict = simparams['amb_dict']
    numtype = simparams['dtype']
    if 'FitType' in simparams.keys():
        fitspec = simparams['FitType']
    else:
        fitspec ='Spectrum'
    nspecs = len(specs)
    datablock = sp.zeros((nspecs,2),dtype=x.dtype)
    datablock[:,0] = x[sp.arange(0,nspecs*2,2)]
    datablock[:,1] = x[sp.arange(1,nspecs*2,2)]
    v_i = x[-1]

    # determine if you've gone beyond the bounds
    #penalty for being less then zero
    grt0 = sp.exp(-datablock)
    pentsum = sp.zeros(grt0.size+1)
    pentsum[:-1] = grt0.flatten()

    #penalties for densities not being equal
    nis = datablock[:-1,0]
    ne = datablock[-1,0]
    nisum = nis.sum()
    pentsum[-1] = sp.exp(-sp.absolute(ne-nisum))

    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])
    (omeg,cur_spec,rcs) = specobj.getspecsep(datablock,specs,v_i,rcsflag=True)
    cur_spec.astype(numtype)
    # Create spectrum guess
    (tau,acf) = spect2acf(omeg,cur_spec)

    if amb_dict['WttMatrix'].shape[-1]!=acf.shape[0]:
        pdb.set_trace()
    guess_acf = sp.dot(amb_dict['WttMatrix'],acf)
    # apply ambiguity function

    guess_acf = guess_acf*rcs/guess_acf[0].real
    if fitspec.lower()=='spectrum':
        # fit to spectrums
        spec_interm = scfft.fft(guess_acf,n=len(cur_spec))
        spec_final = spec_interm.real
        y_interm = scfft.fft(y_acf,n=len(spec_final))
        y = y_interm.real
        yout = (y-spec_final)
    elif fitspec.lower() =='acf':
        yout = y_acf-guess_acf

    # Cannot make the output a complex array! To avoid this problem simply double
    # the size of the array and place the real and imaginary parts in alternating spots.
    if sp.iscomplexobj(yout):
        youttmp=yout.copy()
        yout=sp.zeros(2*len(youttmp)).astype(youttmp.real.dtype)
        yout[::2]=youttmp.real
        yout[1::2] = youttmp.imag

    penadd = sp.sqrt(sp.power(sp.absolute(yout),2).sum())*pentsum.sum()
    return yout+penadd

def makefitsurf(xarrs,y_acf,sensdict,simparams):
    youtsize = [len(x) for x in xarrs]
    ytprod = 1
    for xl in youtsize:
        ytprod = ytprod*xl

    yout = sp.zeros(youtsize,dtype=sp.float128)

    for iparam in range(ytprod):
        curind = sp.unravel_index(iparam,youtsize)
        curx = sp.array([x[curind[ix]] for ix, x in enumerate(xarrs)])

        yout[curind[:]] = sp.power(sp.absolute(ISRSfitfunction(curx,y_acf,sensdict,simparams)),2).sum()
    return(yout)