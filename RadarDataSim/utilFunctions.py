#!/usr/bin/env python
"""
Created on Tue Jul 22 16:18:21 2014

@author: Bodangles
"""
import os
import pickle
import scipy as sp
import scipy.fftpack as scfft
from const.physConstants import v_C_0
import const.sensorConstants as sensconst
from beamtools.bcotools import getangles

import tables
import pdb
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
    nsamps = sp.floor(8.5*m_up)
    nsamps = nsamps-(1-sp.mod(nsamps,2))

    nvec = sp.arange(-sp.floor(nsamps/2.0),sp.floor(nsamps/2.0)+1)
    outsinc = sp.blackman(nsamps)*sp.sinc(nvec/m_up)
    outsinc = outsinc/sp.sum(outsinc)
    dt = 1/(Fsorg*m_up)
    Delay = sp.arange(-(len(nvec)-1),m_up*(nlags+5))*dt
    t_rng = sp.arange(0,1.5*plen,dt)
    numdiff = len(Delay)-len(outsinc)
    outsincpad  = sp.pad(outsinc,(0,numdiff),mode='constant',constant_values=(0.0,0.0))
    (srng,d2d)=sp.meshgrid(t_rng,Delay)
    # envelop function
    envfunc = sp.zeros(d2d.shape)
    envfunc[(d2d-srng+plen-Delay.min()>=0)&(d2d-srng+plen-Delay.min()<=plen)]=1
    envfunc = envfunc/sp.sqrt(envfunc.sum(axis=0).max())
    #create the ambiguity function for everything
    Wtt = sp.zeros((nlags,d2d.shape[0],d2d.shape[1]))
    cursincrep = sp.tile(outsincpad[:,sp.newaxis],(1,d2d.shape[1]))
    Wt0 = Wta = cursincrep*envfunc
    Wt0fft = sp.fft(Wt0,axis=0)
    for ilag in sp.arange(nlags):
        cursinc = sp.roll(outsincpad,ilag*m_up)
        cursincrep = sp.tile(cursinc[:,sp.newaxis],(1,d2d.shape[1]))
        Wta = cursincrep*envfunc
        #do fft based convolution, probably best method given sizes
        Wtafft = scfft.fft(Wta,axis=0)
        if ilag==0:
            nmove = len(nvec)-1
        else:
            nmove = len(nvec)
        Wtt[ilag] = sp.roll(scfft.ifft(Wtafft*sp.conj(Wt0fft),axis=0).real,nmove,axis=0)
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
    padnum = sp.floor(len(spec)/2)
    df = omeg[1]-omeg[0]

    specpadd = sp.pad(spec,(padnum,padnum),mode='constant',constant_values=(0.0,0.0))
    acf = scfft.fftshift(scfft.ifft(scfft.ifftshift(specpadd)))
    dt = 1/(df*len(specpadd))
    tau = sp.arange(-sp.ceil((len(acf)-1.0)/2),sp.floor((len(acf)-1.0)/2+1))*dt
    return tau, acf


#%% making pulse data
def MakePulseData(pulse_shape, filt_freq, delay=16,numtype = sp.complex128):
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

    noise_vec = (sp.random.randn(npts).astype(numtype)+1j*sp.random.randn(npts).astype(numtype))/sp.sqrt(2.0)# make a noise vector
    mult_freq = filt_freq.astype(numtype)*noise_vec
    data = scfft.ifft(mult_freq)
    data_out = pulse_shape*data[delay:(delay+len(pulse_shape))]
    return data_out

def MakePulseDataRep(pulse_shape, filt_freq, delay=16,rep=1,numtype = sp.complex128):
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
    filt_tile = sp.tile(filt_freq[sp.newaxis,:],(rep,1))
    shaperep = sp.tile(pulse_shape[sp.newaxis,:],(rep,1))
    noise_vec = (sp.random.randn(rep,npts).astype(numtype)+1j*sp.random.randn(rep,npts).astype(numtype))/sp.sqrt(2.0)# make a noise vector
    mult_freq = filt_tile.astype(numtype)*noise_vec
    data = scfft.ifft(mult_freq,axis=-1)
    data_out = shaperep*data[:,delay:(delay+len(pulse_shape))]
    return data_out
#%% Pulse shapes
def GenBarker(blen):
    bdict = {1:[-1], 2:[-1, 1], 3:[-1, -1, 1], 4:[-1, -1, 1, -1], 5:[-1, -1, -1, 1, -1],
             7:[-1, -1, -1, 1, 1, -1, 1], 11:[-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
            13:[-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]}
    outar = sp.array(bdict[blen])
    outar.astype(sp.float64)
    return outar
#%% Lag Functions
def CenteredLagProduct(rawbeams,numtype=sp.complex128,pulse =sp.ones(14)):
    """ This function will create a centered lag product for each range using the
    raw IQ given to it.  It will form each lag for each pulse and then integrate
    all of the pulses.
    Inputs:
        rawbeams - This is a NpxNs complex numpy array where Ns is number of
        samples per pulse and Npu is number of pulses
        N - The number of lags that will be created, default is 14.
        numtype - The type of numbers used to create the data. Default is sp.complex128
    Output:
        acf_cent - This is a NrxNl complex numpy array where Nr is number of
        range gate and Nl is number of lags.
    """
    N=len(pulse)
    # It will be assumed the data will be pulses vs rangne
    rawbeams = rawbeams.transpose()
    (Nr,Np) = rawbeams.shape

    # Make masks for each piece of data
    arex = sp.arange(0,N/2.0,0.5);
    arback = sp.array([-sp.int_(sp.floor(k)) for k in arex]);
    arfor = sp.array([sp.int_(sp.ceil(k)) for k in arex]) ;

    # figure out how much range space will be kept
    ap = sp.nanmax(abs(arback));
    ep = Nr- sp.nanmax(arfor);
    rng_ar_all = sp.arange(ap,ep);
    #acf_cent = sp.zeros((ep-ap,N))*(1+1j)
    acf_cent = sp.zeros((ep-ap,N),dtype=numtype)
    for irng in  sp.arange(len(rng_ar_all)):
        rng_ar1 =sp.int_(rng_ar_all[irng]) + arback
        rng_ar2 = sp.int_(rng_ar_all[irng]) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = sp.conj(rawbeams[rng_ar1,:])*rawbeams[rng_ar2,:]
        acf_ave = sp.sum(acf_tmp,1)
        acf_cent[irng,:] = acf_ave# might need to transpose this
    return acf_cent


def BarkerLag(rawbeams,numtype=sp.complex128,pulse=GenBarker(13)):
    """ """
     # It will be assumed the data will be pulses vs rangne
    rawbeams = rawbeams.transpose()
    (Nr,Np) = rawbeams.shape
    # Make matched filter
    filt = sp.fft(pulse[::-1],n=Nr)
    filtmat = sp.repeat(filt[:,sp.newaxis],Np,axis=1)
    rawfreq = sp.fft(rawbeams,axis=0)
    outdata = sp.ifft(filtmat*rawfreq,axis=0)
    outdata = sp.sum(outdata,axis=-1)
    #increase the number of axes
    return outdata[len(pulse)-1:,sp.newaxis]


#%% dictionary file
def dict2h5(filename,dictin):
# Main function test
    h5file = tables.openFile(filename, mode = "w", title = "RadarDataFile out.")
    try:
        # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
        # Make group for each dictionary
        for cvar in dictin.keys():

            h5file.createArray('/',cvar,dictin[cvar],'Static array')
        h5file.close()
    except Exception as inst:
        print type(inst)
        print inst.args
        print inst
        h5file.close()
        raise NameError('Failed to write to h5 file.')
        #%% Test functions
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
    """This function creates a tempreture profile for test purposes."""

    Te = ((45.0/500.0)*(z-200.0))**2+1000.0
    Ti = ((20.0/500.0)*(z-200.0))**2+1000.0
    Te[z<=200.0]=1000.0
    Ti[z<=200.0]=1000.0
    return (Te,Ti)


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
    paramsizlist = sp.array([len(i) for i in paramlists])
    outsize = sp.where(paramsizlist!=1)[0]
    #  make the fit surface and flatten it
    fit_surface = sp.zeros(paramsizlist[outsize])
    fit_surface = fit_surface.flatten()

    for inum in range(sp.prod(paramsizlist)):
        numcopy = inum
        curnum = sp.zeros_like(paramsizlist)
        # TODO: Replace with sp.unravel_index
        # determine current parameters
        for i, iparam in enumerate(reversed(paramsizlist)):
            curnum[i] = sp.mod(numcopy,iparam)
            numcopy = sp.floor(numcopy/iparam)
        curnum = curnum[::-1]
        cur_x = sp.array([ip[curnum[num_p]] for num_p ,ip in enumerate(paramlists)])
        diffthing = errfunc(cur_x,*inputs)
        fit_surface[inum]=(sp.absolute(diffthing)**2).sum()
        # return the fitsurace after its been de flattened
    return fit_surface.reshape(paramsizlist[outsize]).copy()
#%% Config files

def makepicklefile(fname,beamlist,radarname,simparams):
    pickleFile = open(fname, 'wb')
    pickle.dump([{'beamlist':beamlist,'radarname':radarname},simparams],pickleFile)
    pickleFile.close()
def readconfigfile(fname):

    ftype = os.path.splitext(fname)[-1]
    if ftype=='.pickle':
        pickleFile = open(fname, 'rb')
        dictlist = pickle.load(pickleFile)
        pickleFile.close()
        angles = getangles(dictlist[0]['beamlist'],dictlist[0]['radarname'])
        ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
        sensdict = sensconst.getConst(dictlist[0]['radarname'],ang_data)

        simparams = dictlist[1]
        time_lim = simparams['TimeLim']
        pulse  = simparams['Pulse']
        simparams['amb_dict'] = make_amb(sensdict['fs'],simparams['ambupsamp'],
            sensdict['t_s']*len(pulse),len(pulse))
        simparams['angles']=angles
        rng_lims = simparams['RangeLims']
        rng_gates = sp.arange(rng_lims[0],rng_lims[1],sensdict['t_s']*v_C_0*1e-3)
        simparams['Timevec']=sp.arange(0,time_lim,simparams['Fitinter'])
        simparams['Rangegates']=rng_gates

    return(sensdict,simparams)


#def makexample(npts,sensdict,cur_params,pulse,npulses):
#    """This will create a set centered lag products as if it were collected from ISR
#    data with the parameter values in cur_params. The lag products will have the
#    the number of pulses found in npulses using evelope found in pulse.
#    Inputs
#    npts - The length of the spectrum, this will be reduced by 1 if its an even number.
#    sensdict - This is a sensor dictionary that can be created from one of the functions in
#    the sensorconst file.
#    cur_params - The parameters in the order seen for the spectrum method being used.
#    pulse - This is an array that hold the pulse shape from the envelope.
#    npulses - The number of pulses that will be integrated."""
#
#
#    Nrg = 3*len(pulse)
#    N_samps = Nrg +len(pulse)-1
#    #TODO: Make this able to handle any spectrum input.
#    myspec = ISSpectrum(nspec = npts,sampfreq=sensdict['fs'])
#    (omeg,cur_spec) = myspec.getSpectrum(cur_params[0], cur_params[1], cur_params[2], \
#                    cur_params[3], cur_params[4], cur_params[5])
#    Ne = 10**cur_params[2]
#    tr =  cur_params[1]
#    # Set the power for the spectrum
#    cur_spec =  len(cur_spec)**2*Ne/(1+tr)*cur_spec/sp.sum(cur_spec)
#    # Change the spectrum filter kernal for the fft based filtering
#    cur_filt = sp.sqrt(scfft.ifftshift(cur_spec))
#    outdata = sp.zeros((npulses,N_samps),dtype=sp.complex128)
#    samp_num = sp.arange(len(pulse))
#    for ipulse in range(npulses):
#        for isamp in range(Nrg):
#            curpnts =  samp_num+isamp
#            curpulse = MakePulseData(pulse,cur_filt,delay=len(pulse))
#            outdata[ipulse,curpnts] = curpulse +outdata[ipulse,curpnts]
#    # Perform a centered lag product.
#    lags = CenteredLagProduct(outdata,N =len(pulse))
#    return lags
#
