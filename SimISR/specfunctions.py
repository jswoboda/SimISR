#!/usr/bin/env python
"""
specfunctions.py
This module holds the functions that deal with the spectrum formation functions like
fitting and making spectrums.
@author: John Swoboda
"""
import numpy as np
import scipy.fftpack as scfft
import pdb
#
from ISRSpectrum.ISRSpectrum import ISRSpectrum
from SimISR.utilFunctions import spect2acf, update_progress


def ISRSspecmake(ionocont,sensdict, simparams, npts,ifile=0.,nfiles=1.,print_line=True):
    """ This function will take an ionocontainer instance of plasma parameters and create
        ISR spectra for each object.

        Inputs
            ionocont - An instance of the ionocontainer class with plasma parameters. Its param list
                must an array of [Nl,Nt,Ni,2].
            sensdict - A dictionary with sensort information.
            npts - The number of points for the spectra.
        Outputs
            omeg - The frequency vector in Hz.
            outspects - The spectra which have been weighted using the RCS. The
                weighting is npts^2 *rcs.
    """
    v_i = ionocont.getDoppler()
    sample_rate_numerator = simparams['fsnum']
    sample_rate_denominator = simparams['fsden']
    declist = simparams['declist']
    d_fac = np.prod(declist)
    f_s = float(sample_rate_numerator)/sample_rate_denominator/d_fac
    npts = simparams['numpoints']/d_fac
    specobj = ISRSpectrum(centerFrequency=sensdict['fc'], nspec=npts, sampfreq=f_s)


    (n_x, n_t) = ionocont.Param_List.shape[:2]
    outspecs = np.zeros((n_x, n_t, npts))
    outspecsorig = np.zeros_like(outspecs)
    outrcs = np.zeros((n_x, n_t))
    #pdb.set_trace()
    for i_x in np.arange(n_x):
        for i_t in np.arange(n_t):
            if print_line:
                curnum = ifile/nfiles + float(i_x)/n_x/nfiles+float(i_t)/n_t/n_x/nfiles
                outstr = 'Time:{0:d} of {1:d} Location:{2:d} of {3:d}, now making spectrum.'
                outstr = outstr.format(i_t, n_t, i_x, n_x)
                update_progress(curnum, outstr)


            cur_params = ionocont.Param_List[i_x, i_t]
            cur_vel = v_i[i_x, i_t]

            (omeg, cur_spec, rcs) = specobj.getspecsep(cur_params, ionocont.Species,
                                                       cur_vel, rcsflag=True)
            specsum = np.absolute(cur_spec).sum()
            cur_spec_weighted = len(cur_spec)*cur_spec*rcs/specsum
            outspecsorig[i_x, i_t] = cur_spec
            outrcs[i_x, i_t] = rcs
            outspecs[i_x, i_t] = cur_spec_weighted
    return (omeg, outspecs)

def ISRspecmakeout(paramvals,fc,fs,species,npts):
    """ This will make a spectra for a set a param values. This is mainly used
        in the plotting functions to get spectra for given parameters.
        Input
            paramvals - A N_x x N_t x 2Nsp+1 numpy array that holds the parameter
                values. Nx is number of spatial locations, N_t is number of
                times and Nsp is number of ion and electron species.
            fc - The carrier frequency of the ISR.
            fs - The sampling frequency of the ISR.
            species - A list of species.
            npts - The number of points for each spectrum
        Output
            omeg - Frequency vector in Hz.
            outspecs - the spectra to be output."""

    if paramvals.ndim == 2:
        paramvals = paramvals[np.newaxis]

    (N_x, N_t) = paramvals.shape[:2]
    Nsp = len(species)
    Vi = paramvals[:, :, 2*Nsp]
    Parammat = paramvals[:, :, :2*Nsp].reshape((N_x, N_t, Nsp, 2))
    outspecs = np.zeros((N_x, N_t, npts))
    specobj = ISRSpectrum(centerFrequency=fc, nspec=npts, sampfreq=fs)
    outspecsorig = np.zeros_like(outspecs)
    outrcs = np.zeros((N_x, N_t))
    for i_x in np.arange(N_x):
        for i_t in np.arange(N_t):
            cur_params = Parammat[i_x, i_t]
            cur_vel = Vi[i_x, i_t]
            (omeg, cur_spec, rcs) = specobj.getspecsep(cur_params, species, cur_vel, rcsflag=True)
            specsum = np.absolute(cur_spec).sum()
            cur_spec_weighted = 0.5*np.pi*len(cur_spec)**2*cur_spec*rcs/specsum
            outspecsorig[i_x, i_t] = cur_spec
            outrcs[i_x, i_t] = rcs
            outspecs[i_x, i_t] = cur_spec_weighted

    return (omeg, outspecs)


def ISRSfitfunction(x, y_acf, sensdict, simparams, Niratios,  y_err=None ):
    """
    This is the fit fucntion that is used with scipy.optimize.leastsquares. It will
    take a set parameter values construct a spectrum/acf based on those values, apply
    the ambiguity function and take the difference between the two. Since the ACFs are
    complex the arrays split up and the size doubled as it is output.
    Inputs
    x - A Np array of parameter values used
    y_acf - This is the esitmated ACF/spectrum represented as a complex numpy array
    sensdict - This is a dictionary that holds many of the sensor parameters.
    simparams - This is a dictionary that holds info on the simulation parameters.
    y_err -  default None - A numpy array of size Nd that holds the standard deviations of the data.
    fitmethod - default 0 - A number representing the input parameters
    Output
    y_diff - A Nd or 2Nd array if input data is complex that is the difference
    between the data and the fitted model"""
    npts = simparams['numpoints']
    specs = simparams['species']
    amb_dict = simparams['amb_dict']
    numtype = simparams['dtype']
    if 'FitType' in simparams.keys():
        fitspec = simparams['FitType']
    else:
        fitspec = 'Spectrum'
    nspecs = len(specs)
    if not 'fitmode' in simparams.keys():
        (Ti, Ne, Te, v_i) = x
    elif simparams['fitmode'] == 0:
        (Ti, Ne, Te, v_i) = x
    elif simparams['fitmode'] == 1:
        (Ti, Ne, TeoTi, v_i) = x
        Te = TeoTi*Ti
    datablock = np.zeros((nspecs, 2), dtype=x.dtype)
    datablock[:-1, 0] = Ne*Niratios
    datablock[:-1, 1] = Ti
    datablock[-1, 0] = Ne
    datablock[-1, 1] = Te

    # determine if you've gone beyond the bounds
    # penalty for being less then zero
    grt0 = np.exp(-datablock)
    pentsum = np.zeros(grt0.size+1)
    pentsum[:-1] = grt0.flatten()
    ds_fac = int(np.prod(simparams['declist']))

    f_s = float(simparams['fsnum'])/simparams['fsden']
    sampfreq = f_s/ds_fac
    npts = int(npts/ds_fac)
    specobj = ISRSpectrum(centerFrequency=sensdict['fc'], nspec=npts, sampfreq=sampfreq)
    (omeg, cur_spec, rcs) = specobj.getspecsep(datablock, specs, v_i, rcsflag=True)
    cur_spec.astype(numtype)
    # Create spectrum guess
    (_, acf) = spect2acf(omeg, cur_spec)

    if amb_dict['WttMatrix'].shape[-1] != acf.shape[0]:
        pdb.set_trace()
    guess_acf = np.dot(amb_dict['WttMatrix'], acf)
    # apply ambiguity function

    guess_acf = guess_acf*rcs/guess_acf[0].real
    if fitspec.lower() == 'spectrum':
        # fit to spectrums
        spec_interm = scfft.fft(guess_acf, n=len(cur_spec))
        spec_final = spec_interm.real
        y_interm = scfft.fft(y_acf, n=len(spec_final))
        y_spec = y_interm.real
        yout = y_spec-spec_final
    elif fitspec.lower() == 'acf':
        yout = y_acf-guess_acf

    if y_err is not None:
        yout = yout*1./y_err
    # Cannot make the output a complex array! To avoid this problem simply double
    # the size of the array and place the real and imaginary parts in alternating spots.
    if np.iscomplexobj(yout):
        youttmp = yout.copy()
        yout = np.zeros(2*len(youttmp)).astype(youttmp.real.dtype)
        yout[::2] = youttmp.real
        yout[1::2] = youttmp.imag

    penadd = np.sqrt(np.power(np.absolute(yout), 2).sum())*pentsum.sum()

    return yout+penadd


def fitsurface(errfunc,paramlists,inputs):
    """This function will create a fit surface using an error function given by the user
    and an N length list of parameter value lists. The output will be a N-dimensional array
    where each dimension is the size of the array given for each of the parameters. Arrays of
    one element are not represented in the returned fit surface array.
    Inputs:
        errfunc - The function used to determine the error between the given data and
        the theoretical function
        paramlists - An N length list of arrays for each of the parameters.
        inputs - A tuple of the rest of the inputs for error function."""
    paramsizlist = np.array([len(i) for i in paramlists])
    outsize = np.where(paramsizlist!=1)[0]
    #  make the fit surface and flatten it
    fit_surface = np.zeros(paramsizlist[outsize])
    fit_surface = fit_surface.flatten()

    for inum in range(np.prod(paramsizlist)):
        numcopy = inum
        curnum = np.zeros_like(paramsizlist)
        # TODO: Replace with np.unravel_index
        # determine current parameters
        for i, iparam in enumerate(reversed(paramsizlist)):
            curnum[i] = np.mod(numcopy,iparam)
            numcopy = np.floor(numcopy/iparam)
        curnum = curnum[::-1]
        cur_x = np.array([ip[curnum[num_p]] for num_p ,ip in enumerate(paramlists)])
        diffthing = errfunc(cur_x,*inputs)
        fit_surface[inum]=(np.absolute(diffthing)**2).sum()
        # return the fitsurace after its been de flattened
    return fit_surface.reshape(paramsizlist[outsize]).copy()



def makefitsurf(xarrs,y_acf,sensdict,simparams,yerr=None):


    youtsize = [len(x) for x in xarrs]
    ytprod = 1
    for xl in youtsize:
        ytprod = ytprod*xl

    yout = np.zeros(youtsize,dtype=np.float128)

    for iparam in range(ytprod):
        curind = np.unravel_index(iparam,youtsize)
        curx = np.array([x[curind[ix]] for ix, x in enumerate(xarrs)])

        yout[curind[:]] = np.power(np.absolute(ISRSfitfunction(curx,y_acf,sensdict,simparams,yerr)),2).sum()
    return(yout)


def makefitsurfv2(xarrs,y_acf,sensdict,simparams,yerr=None):


    youtsize = [len(x) for x in xarrs]
    ytprod = 1
    for xl in youtsize:
        ytprod = ytprod*xl

    yout = np.zeros(youtsize,dtype=np.float128)

    for iparam in range(ytprod):
        curind = np.unravel_index(iparam,youtsize)
        curx = xarrs[curind[0]][curind[1]]

        yout[curind[:]] = np.power(np.absolute(ISRSfitfunction(curx,y_acf,sensdict,simparams,yerr)),2).sum()
    return(yout)
