#!/usr/bin/env python
"""
Created on Tue Jul 22 16:18:21 2014

@author: Bodangles
"""
from __future__ import print_function
import sys
import warnings
import pickle
import yaml
from six.moves.configparser import ConfigParser
import tables
import scipy as sp
import scipy.fftpack as scfft
import scipy.signal as scisig
import scipy.interpolate as spinterp
#
from isrutilities.physConstants import v_C_0
import isrutilities.sensorConstants as sensconst
from isrutilities import Path

# utility functions

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress, extstr=""):
    """
        This will make a progress bar in the command line

        Args:
            progress (:obj:`float`): Proportion of progress on scale of 0 to one.
            extstr (:obj:'str'): Extra string added to the progress bar.
    """
    bar_length = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0.
        status = "error: progress var must be float\r\n"
    if progress < 0.:
        progress = 0.
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1.
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    if  extstr != "":
        status = status + extstr + '\n'
    text = "\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(bar_length-block),
                                              progress*100, status)
    print(text,end="")
    sys.stdout.flush()


def make_amb(Fsorg, ds_list, pulse, nspec=128, winname='boxcar'):
    """
        Make the ambiguity function dictionary that holds the lag ambiguity and
        range ambiguity. Uses a sinc function weighted by a blackman window. Currently
        only set up for an uncoded pulse.

        Args:
            Fsorg (:obj:`float`): A scalar, the original sampling frequency in Hertz.
            m_up (:obj:`int`): The upsampled ratio between the original sampling rate and the rate of
            the ambiguity function up sampling.
            nlags (:obj:`int`): The number of lags used.

        Returns:
            Wttdict (:obj:`dict`): A dictionary with the keys 'WttAll' which is the full ambiguity function
            for each lag, 'Wtt' is the max for each lag for plotting, 'Wrange' is the
            ambiguity in the range with the lag dimension summed, 'Wlag' The ambiguity
            for the lag, 'Delay' the numpy array for the lag sampling, 'Range' the array
            for the range sampling and 'WttMatrix' for a matrix that will impart the ambiguity
            function on a pulses.
    """


    # come up with frequency response of filter from signal.decimate.
    # For default case use chebychev 1 order 8 and take the power of the
    # original frequency response of each decmination filter because it is run as
    # a zero phase filter using filtfilt.
    curfreq = sp.ones(nspec)
    curds = 1
    m_up = sp.prod(ds_list)
    for i_ds in ds_list:
        # taken from decimate command
        dtisys1 =  scisig.dlti(*sp.signal.cheby1(8, 0.05, 0.8 / i_ds))
        w,hf = scisig.freqz(dtisys1.num,dtisys1.den,nspec/curds,True)
        Hf = sp.absolute(hf)**2
        f_1 = int(sp.ceil(float(nspec/curds)/2))
        curfreq[:f_1] = curfreq[:f_1]*Hf[:f_1]
        curfreq[-f_1:] = curfreq[-f_1:]*Hf[-f_1:]
        curds *= i_ds
    curh = sp.ifft(curfreq)
    nspec = int(nspec)
    plen = len(pulse)
    nlags = plen/m_up

    # make the sinc
    nsamps = sp.floor(23.*m_up)
    nsamps = int(nsamps-(1-sp.mod(nsamps, 2)))

    # need to incorporate summation rule
    vol = 1.
    nvec = sp.arange(-sp.floor(nsamps/2.0), sp.floor(nsamps/2.0)+1).astype(int)
    pos_windows = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
                   'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                   'barthann']
    curwin = scisig.get_window(winname, nsamps)
    # Apply window to the sinc function. This will act as the impulse respons of the filter
    impres = curwin*sp.sinc(nvec/m_up)
    impres = impres/sp.sum(impres)
    impres = curh[nvec]
    d_t = 1./Fsorg
    #make delay vector
    Delay_num = sp.arange(-(len(nvec)-1), m_up*(nlags+5))
    Delay = Delay_num*d_t

    t_rng = sp.arange(0, 1.5*plen)*d_t
    if len(t_rng) > 2e4:
        raise ValueError('The time array is way too large. plen should be in seconds.')
    numdiff = len(Delay)-len(impres)
    numback = int(nvec.min()/m_up-Delay_num.min())
    numfront = numdiff-numback
#    imprespad  = sp.pad(impres,(0,numdiff),mode='constant',constant_values=(0.0,0.0))
    imprespad  = sp.pad(impres, (numback, numfront), mode='constant',
                        constant_values=(0.0, 0.0))
    (d2d, srng) = sp.meshgrid(Delay, t_rng)
    # envelop function
    t_p = sp.arange(plen)/Fsorg

    envfunc = sp.interp(sp.ravel(srng-d2d), t_p, pulse, left=0., right=0.).reshape(d2d.shape)
#    envfunc = sp.zeros(d2d.shape)
#    envfunc[(d2d-srng+plen-Delay.min()>=0)&(d2d-srng+plen-Delay.min()<=plen)]=1
    envfunc = envfunc/sp.sqrt(envfunc.sum(axis=0).max())
    #create the ambiguity function for everything
    Wtt = sp.zeros((nlags,d2d.shape[0],d2d.shape[1]))
    cursincrep = sp.tile(imprespad[sp.newaxis,:],(len(t_rng),1))
    Wt0 = cursincrep*envfunc
    Wt0fft = sp.fft(Wt0,axis=1)
    for ilag in sp.arange(nlags):
        cursinc = sp.roll(imprespad, ilag*m_up)
        cursincrep = sp.tile(cursinc[sp.newaxis, :], (len(t_rng), 1))
        Wta = cursincrep*envfunc
        #do fft based convolution, probably best method given sizes
        Wtafft = scfft.fft(Wta, axis=1)

        nmove = len(nvec)-1
        Wtt[ilag] = sp.roll(scfft.ifft(Wtafft*sp.conj(Wt0fft), axis=1).real, nmove, axis=1)

    # make matrix to take
    imat = sp.eye(nspec)
    tau = sp.arange(-sp.floor(nspec/2.), sp.ceil(nspec/2.))/Fsorg
    tauint = Delay
    interpmat = spinterp.interp1d(tau,imat, bounds_error=0, axis=0)(tauint)
    lagmat = sp.dot(Wtt.sum(axis=1),interpmat)
    W0 = lagmat[0].sum()
    for ilag in range(nlags):
       lagmat[ilag] = ((vol+ilag)/(vol*W0))*lagmat[ilag]

    Wttdict = {'WttAll':Wtt,'Wtt':Wtt.max(axis=0),'Wrange':Wtt.sum(axis=1),'Wlag':Wtt.sum(axis=2),
               'Delay':Delay,'Range':v_C_0*t_rng/2.0,'WttMatrix':lagmat}
    return Wttdict

def spect2acf(omeg,spec,n=None):
    """ Creates acf and time array associated with the given frequency vector and spectrum
    Inputs:
    omeg: The frequency sampling vector
    spec: The spectrum array.
    n: optional, default len(spec), Length of output spectrum
    Output:
    tau: The time sampling array.
    acf: The acf from the original spectrum."""
    if n is None:
        n=float(spec.shape[-1])
#    padnum = sp.floor(len(spec)/2)
    df = omeg[1]-omeg[0]

#    specpadd = sp.pad(spec,(padnum,padnum),mode='constant',constant_values=(0.0,0.0))
    acf = scfft.fftshift(scfft.ifft(scfft.ifftshift(spec,axes=-1),n,axis=-1),axes=-1)
    acf = acf/n
    d_t = 1/(df*n)
    tau = sp.arange(-sp.ceil(float(n-1)/2.),sp.floor(float(n-1)/2.)+1)*d_t
    return tau, acf

def acf2spect(tau,acf,n=None,initshift = False):
    """ Creates spectrum and frequency vector associated with the given time array and acf.
    Inputs:
    tau: The time sampling array.
    acf: The acf from the original spectrum.
    n: optional, default len(acf), Length of output spectrum
    Output:
    omeg: The frequency sampling vector
    spec: The spectrum array.
    """

    if n is None:
        n=float(acf.shape[-1])
    d_t = tau[1]-tau[0]

    if initshift:
        acf = scfft.iffthsift(acf,axes=-1)
    spec = scfft.fftshift(scfft.fft(acf,n=n,axis=-1),axes=-1)
    fs = 1/d_t
    omeg = sp.arange(-sp.ceil(n/2.),sp.floor(n/2.)+1)*fs
    return omeg, spec
#%% making pulse data
def MakePulseDataRep(pulse_shape, filt_freq, delay=16, rep=1, numtype=sp.complex128):
    """ This function will create a repxLp numpy array, where rep is number of independent
        repeats and Lp is number of pulses, of noise shaped by the filter who's frequency
        response is passed as the parameter filt_freq. The pulse shape is delayed by the parameter
        delay into the data. The noise vector that will be multiplied by the filter's frequency
        response will be zero mean complex white Gaussian noise with a power of 1. The user
        then will need to multiply the filter by its size to get the desired power from using
        the function.
        Inputs:
            pulse_shape: A numpy array that holds the shape of the single pulse.
            filt_freq - a numpy array that holds the complex frequency response of the filter
                that will be used to shape the noise data. It is assumed that the
                filter has been correctly scaled so the noise will have the
                desired energy/variance.
            delay - The number of samples that the pulse will be delayed into the
                array of noise data to avoid any problems with filter overlap.
            rep - Number of independent samples/pulses shaped by the filter.
            numtype - The type of numbers used for the output.
        Output
            data_out - A repxLp of data that has been shaped by the filter. Points along
            The first axis are independent of each other while samples along the second
            axis are colored using the filter and multiplied by the pulse shape.
    """
    npts = len(filt_freq)
    multforimag = sp.ones_like(filt_freq)
    hpnt = int(sp.ceil(npts/2.))
    multforimag[hpnt:] = -1
    tmp = scfft.ifft(filt_freq)
    tmp[hpnt:] = 0.
    #comp_filt = scfft.fft(tmp)*sp.sqrt(2.)
    filt_tile = sp.tile(filt_freq[sp.newaxis, :], (rep, 1))
    shaperep = sp.tile(pulse_shape[sp.newaxis, :], (rep, 1))
    noisereal = sp.random.randn(rep, npts).astype(numtype)
    noiseimag = sp.random.randn(rep, npts).astype(numtype)
    noise_vec = (noisereal+1j*noiseimag)/sp.sqrt(2.0)
#    noise_vec = noisereal
    mult_freq = filt_tile.astype(numtype)*noise_vec
    data = scfft.ifft(mult_freq, axis=-1)
    data_out = shaperep*data[:, delay:(delay+len(pulse_shape))]
    return data_out

def MakePulseDataRepLPC(pulse, spec, N, rep1, numtype=sp.complex128):
    """ This will make data by assuming the data is an autoregressive process.
        Inputs
            spec - The properly weighted spectrum.
            N - The size of the ar process used to model the filter.
            pulse - The pulse shape.
            rep1 - The number of repeats of the process.
        Outputs
            outdata - A numpy Array with the shape of the """

    lp = len(pulse)
    r1 = scfft.ifft(scfft.ifftshift(spec))
    rp1 = r1[:N]
    rp2 = r1[1:N+1]
    # Use Levinson recursion to find the coefs for the data
    xr1 = sp.linalg.solve_toeplitz(rp1, rp2)
    lpc = sp.r_[sp.ones(1), -xr1]
    # The Gain  term.
    G = sp.sqrt(sp.sum(sp.conjugate(r1[:N+1])*lpc))
    Gvec = sp.r_[G, sp.zeros(N)]
    Npnt = (N+1)*3+lp
    nfft = scfft.next_fast_len(Npnt)

    _, h_filt = sp.signal.freqz(Gvec, lpc, worN=nfft, whole=True)
    h_tile = sp.tile(h_filt[sp.newaxis, :], (rep1, 1))
    # Create the noise vector and normalize
    xin = sp.random.randn(rep1, nfft)+1j*sp.random.randn(rep1, nfft)
    xinsum = sp.tile(sp.sqrt(sp.sum(xin.real**2+xin.imag**2, axis=1))[:, sp.newaxis],(1, nfft))
    xinsum = xinsum/sp.sqrt(nfft)
    xin = sp.sqrt(nfft)*xin/xinsum
    outdata = sp.ifft(h_tile*xin, axis=1)
    #outdata = sp.signal.lfilter(Gvec, lpc, xin, axis=1)
    outpulse = sp.tile(pulse[sp.newaxis],(rep1, 1))
    outdata = outpulse*outdata[:,N:N+lp]
    return outdata
#%% Pulse shapes
def GenBarker(blen):
    """This function will output a barker code pulse.
    Inputs
        blen -An integer for number of bauds in barker code.
    Output
        outar - A blen length numpy array.
    """
    bdict = {1:[-1], 2:[-1, 1], 3:[-1, -1, 1], 4:[-1, -1, 1, -1], 5:[-1, -1, -1, 1, -1],
             7:[-1, -1, -1, 1, 1, -1, 1], 11:[-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
            13:[-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]}
    outar = sp.array(bdict[blen])
    outar.astype(sp.float64)
    return outar
#%% Lag Functions
def CenteredLagProduct(rawbeams,numtype=sp.complex128,pulse =sp.ones(14),lagtype='centered'):
    """ This function will create a centered lag product for each range using the
    raw IQ given to it.  It will form each lag for each pulse and then integrate
    all of the pulses.
    Inputs:
        rawbeams - This is a NpxNs complex numpy array where Ns is number of
        samples per pulse and Npu is number of pulses
        N - The number of lags that will be created, default is 14.
        numtype - The type of numbers used to create the data. Default is sp.complex128
        lagtype - Can be centered forward or backward.
    Output:
        acf_cent - This is a NrxNl complex numpy array where Nr is number of
        range gate and Nl is number of lags.
    """
    n_pulse = len(pulse)
    # It will be assumed the data will be pulses vs rangne
    rawbeams = rawbeams.transpose()
    (Nr, Np) = rawbeams.shape

    # Make masks for each piece of data
    if lagtype == 'forward':
        arback = sp.zeros(n_pulse, dtype=int)
        arfor = sp.arange(n_pulse, dtype=int)

    elif lagtype == 'backward':
        arback = sp.arange(n_pulse, dtype=int)
        arfor = sp.zeros(n_pulse, dtype=int)
    else:
       # arex = sp.arange(0,N/2.0,0.5);
        arback = -sp.floor(sp.arange(0, n_pulse/2.0, 0.5)).astype(int)
        arfor = sp.ceil(sp.arange(0, n_pulse/2.0, 0.5)).astype(int)

    # figure out how much range space will be kept
    ap = sp.nanmax(abs(arback))
    ep = Nr- sp.nanmax(arfor)
    rng_ar_all = sp.arange(ap, ep)
#    wearr = (1./(N-sp.tile((arfor-arback)[:,sp.newaxis],(1,Np)))).astype(numtype)
    #acf_cent = sp.zeros((ep-ap,N))*(1+1j)
    acf_cent = sp.zeros((ep-ap, n_pulse), dtype=numtype)
    for irng, curange in  enumerate(rng_ar_all):
        rng_ar1 = int(curange) + arback
        rng_ar2 = int(curange) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = sp.conj(rawbeams[rng_ar1, :])*rawbeams[rng_ar2, :]#*wearr
        acf_ave = sp.sum(acf_tmp, 1)
        acf_cent[irng, :] = acf_ave# might need to transpose this
    return acf_cent


def BarkerLag(rawbeams, numtype=sp.complex128, pulse=GenBarker(13), lagtype=None):
    """This will process barker code data by filtering it with a barker code pulse and
    then sum up the pulses.
    Inputs
        rawbeams - A complex numpy array size NpxNs where Np is the number of pulses and
        Ns is the number of samples.
        numtype - The type of numbers being used for processing.
        pulse - The barkercode pulse.
    Outputs
        outdata- A Nrx1 size numpy array that holds the processed data. Nr is the number
        of range gates  """
     # It will be assumed the data will be pulses vs rangne
    rawbeams = rawbeams.transpose()
    (Nr, Np) = rawbeams.shape
    pulsepow = sp.power(sp.absolute(pulse), 2.0).sum()
    # Make matched filter
    filt = sp.fft(pulse[::-1]/sp.sqrt(pulsepow), n=Nr)
    filtmat = sp.repeat(filt[:, sp.newaxis], Np, axis=1)
    rawfreq = sp.fft(rawbeams, axis=0)
    outdata = sp.ifft(filtmat*rawfreq, axis=0)
    outdata = outdata*outdata.conj()
    outdata = sp.sum(outdata, axis=-1)
    #increase the number of axes
    return outdata[len(pulse)-1:, sp.newaxis]

def makesumrule(ptype, nlags, lagtype='centered'):
    """ This function will return the sum rule.
        Inputs
            ptype - The type of pulse.
            plen - Length of the pulse in seconds.
            ts - Sample time in seconds.
            lagtype -  Can be centered forward or backward.
        Output
            sumrule - A 2 x nlags numpy array that holds the summation rule.
    """
    if ptype.lower() == 'long':
        if lagtype == 'forward':
            arback = -sp.arange(nlags, dtype=int)
            arforward = sp.zeros(nlags, dtype=int)
        elif lagtype == 'backward':
            arback = sp.zeros(nlags, dtype=int)
            arforward = sp.arange(nlags, dtype=int)
        else:
            arback = -sp.ceil(sp.arange(0, nlags/2.0, 0.5)).astype(int)
            arforward = sp.floor(sp.arange(0, nlags/2.0, 0.5)).astype(int)
        sumrule = sp.array([arback, arforward])
    elif ptype.lower() == 'barker':
        sumrule = sp.array([[0], [0]])
    return sumrule

#%% Make pulse
def makepulse(ptype,nsamps,ts,nbauds=13):
    """ This will make the pulse array.
        Inputs
            ptype - The type of pulse used.
            plen - The length of the pulse in seconds.
            ts - The sampling rate of the pulse.
        Output
            pulse - The pulse array that will be used as the window in the data formation.
            plen - The length of the pulse with the sampling time taken into account.
    """
    if ptype.lower() == 'long':
        pulse = sp.ones(nsamps)
        plen = nsamps*ts

    elif ptype.lower() == 'barker':

        blen = sp.array([1, 2, 3, 4, 5, 7, 11, 13])
        nsampsarg = sp.argmin(sp.absolute(blen-nbauds))
        nbauds = blen[nsampsarg]
        pulse = GenBarker(nbauds)
        baudratio = float(nbauds)/nsamps
        pulse_samps = sp.floor(sp.arange(nsamps)*baudratio)
        pulse = pulse[pulse_samps]
        plen = nsamps*ts
#elif ptype.lower()=='ac':
    else:
        raise ValueError('The pulse type %s is not a valide pulse type.' % (ptype))

    return (pulse, plen)


#%% dictionary file
def dict2h5(fn,dictin):
    """A function that will save a dictionary to a h5 file.
    Inputs
        filename - The file name in a string.
        dictin - A dictionary that will be saved out.
    """
    fn = Path(fn).expanduser()
    if fn.is_file():
        fn.unlink()
    with tables.open_file(str(fn), mode = "w", title = "RadarDataFile out.") as f:
        try:
            # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
            # Make group for each dictionary
            for cvar in dictin.keys():
    #            pdb.set_trace()
                if type(dictin[cvar]) is list:
                    f.create_group('/',cvar)
                    lenzeros= len(str(len(dictin[cvar])))-1
                    for inum, datapnts in enumerate(dictin[cvar]):
                        f.create_array('/'+cvar,'Inst{0:0{1:d}d}'.format(inum,lenzeros),datapnts,'Static array')
                elif type(dictin[cvar]) is sp.ndarray:
                    f.create_array('/',cvar,dictin[cvar],'Static array')
                else:
                    raise ValueError('Values in list must be lists or numpy arrays')
            f.close()
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            raise NameError('Failed to write to h5 file.')

def h52dict(filename):
    """This will read in the information from a structure h5 file where it is assumed
    that the base groups are either root or are a group that leads to arrays.
    Input
    filename - A string that holds the name of the h5 file that will be opened.
    Output
    outdict - A dictionary where the keys are the group names and the values are lists
    or numpy arrays."""
    h5file = tables.open_file(filename, mode = "r")
    output ={}
    for group in h5file.walk_groups('/'):
            output[group._v_pathname]={}
            for array in h5file.list_nodes(group, classname = 'Array'):
                output[group._v_pathname][array.name]=array.read()
    h5file.close()

    outdict= {}
    # first get the
    base_arrs = output['/']

    outdict={ikey.strip('/'):base_arrs[ikey] for ikey in base_arrs.keys()}

    del output['/']
    for ikey in output.keys():
        sublist = [output[ikey][l] for l in output[ikey].keys()]
        outdict[ikey.strip('/')] = sublist

    return outdict
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


def TempProfile(z,T0=1000.,z0=100.):
    """
    This function creates a tempreture profile using arc tan functions for test purposes.
    Inputs
        z - The altitude locations in km.
        T0 - The value of the lowest tempretures in K.
        z0 - The middle value of the atan functions along alitutude. In km.
    Outputs
        Te - The electron density profile in K. 1700*(atan((z-z0)2*exp(1)/400-exp(1))+1)/2 +T0
        Ti - The ion density profile in K. 500*(atan((z-z0)2*exp(1)/400-exp(1))+1)/2 +T0
    """
    zall = (z-z0)*2.*sp.exp(1)/400. -sp.exp(1)
    atanshp = (sp.tanh(zall)+1.)/2
    Te = 1700*atanshp+T0
    Ti = 500*atanshp+T0

    return (Te,Ti)



#%% Config files

def makeconfigfile(fname,beamlist,radarname,simparams_orig):
    """This will make the config file based off of the desired input parmeters.
    Inputs
        fname - Name of the file as a string.
        beamlist - A list of beams numbers used by the AMISRS
        radarname - A string that is the name of the radar being simulated.
        simparams_orig - A set of simulation parameters in a dictionary."""
    fname = Path(fname).expanduser()

    curpath = Path(__file__).resolve().parent
    d_file = curpath/'default.ini'
    fext = fname.suffix

    # reduce the number of stuff needed to be saved and avoid problems with writing
    keys2save = ['IPP', 'IPPsamps', 'TimeLim', 'datasamples', 'calsamples', 'noisesamples',
                 'Pulselength', 'Pulsetype', 'fsnum', 'fsden', 'Tint', 'Fitinter',
                 'dtype', 'ambupsamp', 'species', 'numpoints', 'startfile',
                 'FitType', 'beamrate', 'outangles', 'declist']

    if not 'beamrate' in simparams_orig.keys():
        simparams_orig['beamrate'] = 1
    if not 'outangles' in simparams_orig.keys():
        simparams_orig['outangles'] = beamlist
    simparams = {i:simparams_orig[i] for i in keys2save}
    if fext == '.yml':
        with fname.open('w') as f:
            yaml.dump([{'beamlist':beamlist, 'radarname':radarname}, simparams], f)
    elif fext == '.ini':
        defaultparser = ConfigParser()
        defaultparser.read(str(d_file))
#        config = configparser()
#        config.read(fname)
        cfgfile = open(str(fname),'w')
        config = ConfigParser(allow_no_value = True)

        config.add_section('section 1')
        beamstring = ""
        for beam in beamlist:
            beamstring += str(beam)
            beamstring += " "
        config.set('section 1','; beamlist must be list of ints')
        config.set('section 1','beamlist',beamstring)
        config.set('section 1','; radarname can be pfisr, risr, or sondastrom')
        config.set('section 1','radarname',radarname)

        config.add_section('simparams')
        config.add_section('simparamsnames')
        defitems = [i[0] for i in defaultparser.items('simparamsnotes')]
        for param in simparams:
            if param=='Beamlist':
                continue
            if param.lower() in defitems:
                paramnote = defaultparser.get('simparamsnotes',param.lower())
            else:
                paramnote = 'Not in default parameters'
            config.set('simparams','; '+param +' '+paramnote)
            # for the output list of angles
            if param.lower()=='outangles':
                outstr = ''
                beamlistlist = simparams[param]
                if beamlistlist=='':
                    beamlistlist=beamlist
                for ilist in beamlistlist:
                    if isinstance(ilist,list) or isinstance(ilist,sp.ndarray):
                        for inum in ilist:
                            outstr=outstr+str(inum)+' '

                    else:
                        outstr=outstr+str(ilist)
                    outstr=outstr+', '
                outstr=outstr[:-2]
                config.set('simparams',param,outstr)

            elif isinstance(simparams[param],list):
                data = ""
                for a in simparams[param]:
                    data += str(a)
                    data += " "
                config.set('simparams',param,str(data))
            else:  #TODO config.set() is obsolete, undefined behavior!  use mapping protocol instead https://docs.python.org/3/library/configparser.html#mapping-protocol-access
                config.set('simparams',param,str(simparams[param]))
            config.set('simparamsnames',param,param)
        config.write(cfgfile)
        cfgfile.close()
    else:
        raise ValueError('fname needs to have an extension of .pickle or .ini')

def makedefaultfile(fname):
    """
        This function will give you default configuration dictionaries.
    """
    (sensdict, simparams) = getdefualtparams()
    makeconfigfile(fname, simparams['Beamlist'], sensdict['Name'], simparams)

def getdefualtparams():
    """
        This function will copy the default configuration file to whatever file the users
        specifies.
    """
    curpath = Path(__file__).parent
    d_file = curpath / 'default.ini'
    (sensdict, simparams) = readconfigfile(str(d_file))
    return sensdict, simparams

def readconfigfile(fname):
    """
        This funciton will read in the ini or yaml files that are used for configuration.

        Args:
            fname - A string containing the file name and location.

        Returns:
            sensdict - A dictionary that holds the sensor parameters.
            simparams - A dictionary that holds the simulation parameters.
    """

    fname = Path(fname).expanduser()
    if not fname.is_file():
        raise IOError('{} not found'.format(fname))

    ftype = fname.suffix
    curpath = fname.parent
    if ftype == '.yml':
        with fname.open('r') as f:
            dictlist = yaml.load(f)

        angles = sensconst.getangles(dictlist[0]['beamlist'], dictlist[0]['radarname'])
        beamlist = [float(i) for i in dictlist[0]['beamlist']]
        ang_data = sp.array([[iout[0], iout[1]] for iout in angles])
        sensdict = sensconst.getConst(dictlist[0]['radarname'], ang_data)

        simparams = dictlist[1]
    if ftype == '.ini':
        config = ConfigParser()
        config.read(str(fname))
        beamlist = config.get('section 1', 'beamlist').split()
        beamlist = [float(i) for i in beamlist]
        angles = sensconst.getangles(beamlist, config.get('section 1', 'radarname'))
        ang_data = sp.array([[iout[0], iout[1]] for iout in angles])

        sensdict = sensconst.getConst(config.get('section 1', 'radarname'), ang_data)

        simparams = {}
        for param in config.options('simparams'):
            rname = config.get('simparamsnames', param)
            simparams[rname] = config.get('simparams', param)

        for param in simparams:
            if simparams[param] == "<type 'numpy.complex128'>":
                simparams[param] = sp.complex128
            elif simparams[param] == "<type 'numpy.complex64'>":
                simparams[param] = sp.complex64
            elif param == 'outangles':
                outlist1 = simparams[param].split(',')
                simparams[param] = [[float(j) for j in
                                     i.lstrip().rstrip().split(' ')] for i in outlist1]
            else:
                simparams[param] = simparams[param].split(" ")
                if len(simparams[param]) == 1:
                    simparams[param] = simparams[param][0]
                    try:
                        simparams[param] = float(simparams[param])
                    except:
                        pass
                else:
                    for a in range(len(simparams[param])):
                        try:
                            simparams[param][a] = float(simparams[param][a])
                        except:
                            pass



    if 'declist' not in simparams.keys():
        simparams['declist'] = []


    for ikey in sensdict.keys():
        if ikey  in simparams.keys():
            sensdict[ikey] = simparams[ikey]
#            del simparams[ikey]
    ds_fac = int(sp.prod(simparams['declist']))
    simparams['Beamlist'] = beamlist
    time_lim = simparams['TimeLim']
    f_s = float(simparams['fsnum'])/simparams['fsden']
    t_s = float(simparams['fsden'])/simparams['fsnum']
    (pulse, simparams['Pulselength']) = makepulse(simparams['Pulsetype'],
                                                  simparams['Pulselength'],
                                                  t_s)
    simparams['Pulse'] = pulse
    simparams['amb_dict'] = make_amb(f_s/ds_fac, simparams['declist'], pulse,
                                     simparams['numpoints'])
    simparams['angles'] = angles
    d_len = simparams['datasamples']
    rng_gates = sp.arange(d_len[0], d_len[1]-(len(pulse)-1))*t_s*v_C_0*1e-3/2.
    rng_gates_ds = rng_gates[::ds_fac]
    simparams['Timevec'] = sp.arange(0, time_lim, simparams['Fitinter'])
    simparams['Rangegates'] = rng_gates
    if 'lagtype' not in simparams.keys():
        simparams['lagtype'] = 'centered'

    plen_ds = simparams['Pulselength']/ds_fac
    sumrule = makesumrule(simparams['Pulsetype'], plen_ds, simparams['lagtype'])
    simparams['SUMRULE'] = sumrule
    minrg = -sumrule[0].min()
    maxrg = len(rng_gates_ds)-sumrule[1].max()

    simparams['Rangegatesfinal'] = sp.array([sp.mean(rng_gates_ds[irng+sumrule[0, 0]:irng+sumrule[1, 0]+1])
                                             for irng in range(minrg, maxrg)])
    # HACK need to move this to the sensor constants part
    sensdict['CalDiodeTemp'] = 1689.21
    # Set the number of noise samples to IPP

    if ('startfile' in simparams.keys() and simparams['startfile']) and simparams['Pulsetype'].lower() != 'barker':
        relpath = Path(simparams['startfile'])
        if not relpath.is_absolute():
            # Some times the ini files may split the strings of the start
            # file because of white space in file names.
            if type(simparams['startfile'])is list:
                startfile = " ".join(simparams['startfile'])
            else:
                startfile = simparams['startfile']

            fullfilepath = curpath.joinpath(startfile)
            simparams['startfile'] = str(fullfilepath)

        else:
            fullfilepath = simparams['startfile']
        stext = Path(fullfilepath).is_file()
        if not stext:
            warnings.warn('The given start file does not exist', UserWarning)

    elif simparams['Pulsetype'].lower() != 'barker':
        warnings.warn('No start file given', UserWarning)

    return(sensdict, simparams)
