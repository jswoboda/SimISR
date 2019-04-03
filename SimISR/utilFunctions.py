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
    print(text, end="")
    sys.stdout.flush()


def make_amb(Fsorg, ds_list, pulse, nspec=128, sweepid = [300], winname='boxcar'):
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
    # curds = 1
    m_up = sp.prod(ds_list)

    nout = nspec/m_up


    nspec = int(nspec)
    plen = pulse.shape[-1]
    nlags = plen/m_up

    # find alternating codes and long pulse
    sweepu, u_ind = sp.unique(sweepid, return_index=True)
    ac_ind = u_ind[sp.where(sp.logical_and(sweepu >= 1, sweepu <= 32))[0]]
    ac_pulses = pulse[ac_ind]
    n_codes = ac_pulses.shape[0]
    ac_codes = ac_pulses[:,::m_up]

    decode_arr = sp.zeros((n_codes, nlags))
    for ilag in range(nlags):
        if ilag==0:
            decode_arr[:,ilag] = sp.sum(ac_codes*ac_codes,axis=-1)
        else:
            decode_arr[:,ilag] = sp.sum(ac_codes[:,:-ilag]*ac_codes[:,ilag:],axis=-1)
    lp_ind = u_ind[sp.where(sweepu == 300)[0]]
    lp_pulse = pulse[lp_ind].flatten()
    # make the sinc
    nsamps = int(plen-(1-sp.mod(plen, 2)))

    # need to incorporate summation rule
    vol = 1.
    nvec = sp.arange(-sp.floor(nsamps/2.0), sp.floor(nsamps/2.0)+1).astype(int)
    pos_windows = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
                   'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                   'barthann']
    curwin = scisig.get_window(winname, nsamps)
    # Apply window to the sinc function. This will act as the impulse respons of the filter

    impres = curwin*sp.sinc(nvec.astype(float)/m_up)
    impres = impres/sp.sum(impres)
    d_t = 1./Fsorg/m_up
    #make delay vector
    delay_num = sp.arange(-m_up*(nlags+2), m_up*(nlags+2)+1)
    delay = delay_num*d_t

    t_rng = sp.arange(-sp.floor(.5*plen), sp.ceil(1.5*plen))*d_t
    if len(t_rng) > 2e4:
        raise ValueError('The time array is way too large. plen should be in seconds.')
    numdiff = len(delay)-nsamps
    # numback = int(nvec.min()/m_up-delay_num.min())
    # numfront = numdiff-numback
#    imprespad  = sp.pad(impres,(0,numdiff),mode='constant',constant_values=(0.0,0.0))
    imprespad = sp.pad(impres, (numdiff/2, numdiff/2), mode='constant',
                       constant_values=(0.0, 0.0))
    cursincrep = sp.tile(imprespad[sp.newaxis, :], (len(t_rng), 1))

    (d2d, srng) = sp.meshgrid(delay, t_rng)
    # envelop function
    t_p = sp.arange(plen)*d_t

    envfunc = sp.interp(sp.ravel(srng-d2d), t_p, lp_pulse, left=0., right=0.).reshape(d2d.shape)
#    envfunc = sp.zeros(d2d.shape)
#    envfunc[(d2d-srng+plen-Delay.min()>=0)&(d2d-srng+plen-Delay.min()<=plen)]=1
    envfunc = envfunc/sp.sqrt(envfunc.sum(axis=0).max())
    #create the ambiguity function for everything
    Wtt = sp.zeros((nlags, d2d.shape[0], d2d.shape[1]))
    Wt0 = scfft.ifftshift(cursincrep*envfunc, axes=1)
    Wt0fft = scfft.fft(Wt0, axis=1)
    for ilag in sp.arange(nlags):
        Wtafft = sp.roll(Wt0fft, ilag*m_up, axis=0)
        curwt =  scfft.ifftshift(scfft.ifft(Wtafft*sp.conj(Wt0fft), axis=1).real, axes=1)
        Wtt[ilag] = sp.roll(curwt, ilag*m_up, axis=1)


    # make matrix for application of
    imat = sp.eye(nspec)
    tau = sp.arange(-sp.floor(nspec/2.), sp.ceil(nspec/2.))*d_t
    tauint = delay
    interpmat = spinterp.interp1d(tau, imat, bounds_error=0, axis=0)(tauint)
    lagmat = sp.dot(Wtt.sum(axis=1), interpmat)
    W0 = lagmat[0].sum()
    for ilag in range(nlags):
        lagmat[ilag] = ((vol+ilag)/(vol*W0))*lagmat[ilag]
    lagmat = lagmat[:, ::m_up]

    #%% Alt. code ambigutity
    Wttac = sp.zeros((nlags, d2d.shape[0], d2d.shape[1]))

    for icn, i_pulse in enumerate(ac_pulses):
        envfunc = sp.interp(sp.ravel(srng-d2d), t_p, i_pulse, left=0., right=0.).reshape(d2d.shape)
    #    envfunc = sp.zeros(d2d.shape)
    #    envfunc[(d2d-srng+plen-Delay.min()>=0)&(d2d-srng+plen-Delay.min()<=plen)]=1
        envfunc = envfunc/sp.sqrt(sp.absolute(envfunc).sum(axis=0).max())
        #create the ambiguity function for everything

        Wt0 = scfft.ifftshift(cursincrep*envfunc, axes=1)
        Wt0fft = sp.fft(Wt0, axis=1)
        for ilag in range(nlags):
            Wtafft = sp.roll(Wt0fft, ilag*m_up, axis=0)
            curwt =  scfft.ifftshift(scfft.ifft(Wtafft*sp.conj(Wt0fft), axis=1).real, axes=1)
            curwt = sp.roll(curwt, ilag*m_up, axis=1)
            Wttac[ilag] = Wttac[ilag] + curwt*decode_arr[icn, ilag]

    Wttac[0] = Wttac[0]/nlags
    Wttac = Wttac/ac_pulses.shape[0]
    lagmatac = sp.dot(Wttac.sum(axis=1), interpmat)
    W0ac = lagmatac[0].sum()
    for ilag in range(nlags):
        lagmatac[ilag] = ((vol+ilag)/(vol*W0ac))*lagmatac[ilag]
    lagmatac = lagmatac[:, ::m_up]
    wttdict = {'WttAll':Wtt, 'Wtt':Wtt.max(axis=0), 'Wrange':Wtt.sum(axis=1),
               'Wlag':Wtt.sum(axis=2), 'Delay':delay, 'Range':v_C_0*t_rng/2.0,
               'WttMatrix':lagmat, 'WttAllac':Wttac, 'Wttac':Wttac.max(axis=0),
               'Wrangeac':Wttac.sum(axis=1), 'Wlagac':Wttac.sum(axis=2),
               'WttMatrixac':lagmatac}
    return wttdict

def spect2acf(omeg, spec, n_s=None):
    """ Creates acf and time array associated with the given frequency vector and spectrum
    Inputs:
    omeg: The frequency sampling vector
    spec: The spectrum array.
    n: optional, default len(spec), Length of output spectrum
    Output:
    tau: The time sampling array.
    acf: The acf from the original spectrum."""
    if n_s is None:
        n_s = float(spec.shape[-1])
#    padnum = sp.floor(len(spec)/2)
    d_f = omeg[1]-omeg[0]

#    specpadd = sp.pad(spec,(padnum,padnum),mode='constant',constant_values=(0.0,0.0))
    acf = scfft.fftshift(scfft.ifft(scfft.ifftshift(spec, axes=-1), n_s, axis=-1), axes=-1)
    d_t = 1/(d_f*n_s)
    tau = sp.arange(-sp.ceil(float(n_s-1)/2.), sp.floor(float(n_s-1)/2.)+1)*d_t
    return tau, acf

def acf2spect(tau, acf, n_s=None, initshift=False):
    """ Creates spectrum and frequency vector associated with the given time array and acf.
    Inputs:
    tau: The time sampling array.
    acf: The acf from the original spectrum.
    n: optional, default len(acf), Length of output spectrum
    Output:
    omeg: The frequency sampling vector
    spec: The spectrum array.
    """

    if n_s is None:
        n_s = float(acf.shape[-1])
    d_t = tau[1]-tau[0]

    if initshift:
        acf = scfft.ifftshift(acf, axes=-1)
    spec = scfft.fftshift(scfft.fft(acf, n=n_s, axis=-1), axes=-1)
    fs = 1/d_t
    omeg = sp.arange(-sp.ceil(n_s/2.), sp.floor(n_s/2.)+1)*fs
    return omeg, spec
#%% making pulse data

def MakePulseDataRepLPC(pulse, spec, nlpc, p_ind, numtype=sp.complex128):
    """
        This will make shaped noise data using linear predictive coding windowed
        by the pulse function.
        Args:
            spec (:obj:'ndarray'): The properly weighted spectrum.
            N (:obj:'int'): The size of the ar process used to model the filter.
            p_ind (:obj:'ndarray'): The pulse shape. The pulse is flipped in this so the first sample
                    will be the leading in range.
            p_ind (:obj:'int'): The index of the pulses in the pulse array that will be used.
        Returns:
            outdata (:obj:'ndarray'): A numpy array with the shape of the rep1xlen(pulse)
    """

    npulse, lp = pulse.shape
    # repeat the pulse pattern
    r1 = scfft.ifft(scfft.ifftshift(spec))
    rp1 = r1[:nlpc]
    rp2 = r1[1:nlpc+1]
    # rcs is encoded in the spectrum
    # Use Levinson recursion to find the coefs for the data
    xr1 = sp.linalg.solve_toeplitz(rp1, rp2)
    lpc = sp.r_[sp.ones(1), -xr1]
    # The Gain  term.
    G = sp.sqrt(sp.sum(sp.conjugate(r1[:nlpc+1])*lpc))
    Gvec = sp.r_[G, sp.zeros(nlpc)]
    n_pnt = (nlpc+1)*3+lp
    rep1 = len(p_ind)
    if n_pnt >= 200:
        nfft = scfft.next_fast_len(n_pnt)
        _, h_filt = sp.signal.freqz(Gvec, lpc, worN=nfft, whole=True)
        h_tile = sp.tile(h_filt[sp.newaxis, :], (rep1, 1))
        # Create the noise vector and normalize
        xin = sp.random.randn(rep1, nfft)+1j*sp.random.randn(rep1, nfft)
        x_vec = sp.mean(xin.real**2+xin.imag**2, axis=1)
        xinsum = sp.tile(sp.sqrt(x_vec)[:, sp.newaxis], (1, nfft))
        xinsum = xinsum
        xin = sp.sqrt(nfft)*xin/xinsum
        outdata = sp.ifft(h_tile*xin, axis=1)
        # Flipping pulse
        outpulse = pulse[p_ind,::-1]
        outdata = outpulse*outdata[:, nlpc:nlpc+lp]
    else:
        xin = sp.random.randn(rep1, n_pnt)+1j*sp.random.randn(rep1, n_pnt)
        x_vec = sp.mean(xin.real**2+xin.imag**2, axis=1)
        xinsum = sp.tile(sp.sqrt(x_vec)[:, sp.newaxis], (1, n_pnt))
        xin = xin/xinsum
        outdata = sp.signal.lfilter(Gvec, lpc, xin, axis=1)
        # Flipping pulse
        outpulse = pulse[p_ind,::-1]
        outdata = outpulse*outdata[:, nlpc:nlpc+lp]
    #outdata = sp.sqrt(rcs)*outdata/sp.sqrt(sp.mean(outdata.var(axis=1)))
    return outdata
#%% Pulse shapes
def GenBarker(blen):
    """This function will output a barker code pulse.
    Inputs
        blen - An integer for number of bauds in barker code.
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
def CenteredLagProduct(rawbeams, numtype=sp.complex128, pulse=sp.ones(14), lagtype='centered'):
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
    if ptype.lower() == 'long' or ptype.lower() == 'interleaved':
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
def makepulse(ptype, nsamps, t_s, nbauds=16):
    """ This will make the pulse array.
        Inputs
            ptype - The type of pulse used.
            plen - The length of the pulse in seconds.
            ts - The sampling rate of the pulse.
        Output
            pulse - The pulse array that will be used as the window in the data formation.
            plen - The length of the pulse with the sampling time taken into account.
    """
    plen = nsamps*t_s
    if ptype.lower() == 'long':
        pulse = sp.ones(nsamps)[sp.newaxis]
        sweepid = sp.array([300])
        sweepnum = sp.array([0])
    elif ptype.lower() == 'barker':
        blen = sp.array([1, 2, 3, 4, 5, 7, 11, 13])
        nsampsarg = sp.argmin(sp.absolute(blen-nbauds))
        nbauds = blen[nsampsarg]
        pulse = GenBarker(nbauds)
        baudratio = float(nbauds)/nsamps
        pulse_samps = sp.floor(sp.arange(nsamps)*baudratio)
        pulse = pulse[pulse_samps]
        plen = nsamps*ts
        sweepid = sp.array([400])
        sweepnum = sp.array([0])
    elif ptype.lower() == 'ac':
        pulse, sweepid, sweepnum = gen_ac(nsamps, nbauds)

    elif ptype.lower() == 'interleaved':
        pulse, acsweepid, acsweepnum = gen_ac(nsamps, nbauds)
        pulse = pulse.repeat(3, axis=0)
        pulse[2::3, :] = 1
        sweepid = acsweepid.repeat(3)
        sweepnum = acsweepnum.repeat(3)
        sweepid[2::3] = 300
        sweepnum[2::3] = 0
    else:
        raise ValueError('The pulse type %s is not a valide pulse type.' % (ptype))

    return (pulse, plen, sweepid, sweepnum)

def gen_ac(nsamps, nbauds):
    """
        This will generate a set of alternating code pulses and their sweep ids.

        Args:
            nsamps:``int``: The length of the pulse in samples.
            nbauds:``int``: The number of bauds for each code.

        Returns:
            pulse:``array``: A numpy array of shape 2nbaudsxnsamps holding the
            pulses.
            sweepid:``array``: Array of sweep ids.
            sweepnum:``array``: Array of sweep numbers.
    """
    blen = sp.array([4, 8, 16])
    nsampsarg = sp.argmin(sp.absolute(blen-nbauds))
    nbauds = blen[nsampsarg]
    samp_mat = sp.arange(nbauds).repeat(nsamps/nbauds)
    walshmat = sp.linalg.hadamard(nbauds*2)
    # Strong codes found in Hysell 2018 textbook
    if nbauds == 4:
        colsoct = ['00','01','02','04']
    elif nbauds == 8:
        colsoct = ['00','01','02','04','10','03','07','16']
    elif nbauds == 16:
        colsoct = ['00','01','03','04','10','20','17','37','21','14','31','35','24','06','15','32']
    cols = [int(i, 8) for i in colsoct]
    pulse = walshmat[:,cols]
    # List of alternating code phases are not the same as the order in the textbooks
    if nbauds == 16:
        phasefile = Path(__file__).resolve().parent / 'acphase.txt'
        all_phase = sp.genfromtxt(str(phasefile))
        pulse = sp.cos(sp.pi*all_phase/180.)
    pulse = pulse[:, samp_mat]
    sweepnum = sp.arange(nbauds*2)
    sweepid = sweepnum+1
    return pulse, sweepid, sweepnum
#%% dictionary file
def dict2h5(fn, dictin):
    """
        A function that will save a dictionary to a h5 file.
        Args:
            filename:``str``:The file name in a string.
            dictin:``dict``: A flat dictionary that will be saved out.
    """
    fn = Path(fn).expanduser()
    if fn.is_file():
        fn.unlink()
    with tables.open_file(str(fn), mode="w", title="RadarDataFile out.") as f:
        try:
            # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
            # Make group for each dictionary
            for cvar in dictin.keys():
                if type(dictin[cvar]) is list:
                    f.create_group('/', cvar)
                    lenzeros = len(str(len(dictin[cvar])))-1
                    for inum, datapnts in enumerate(dictin[cvar]):
                        f.create_array('/'+cvar, 'Inst{0:0{1:d}d}'.format(inum, lenzeros),datapnts, 'Static array')
                elif type(dictin[cvar]) is sp.ndarray:
                    f.create_array('/', cvar, dictin[cvar], 'Static array')
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
    h5file = tables.open_file(filename, mode="r")
    output = {}
    for group in h5file.walk_groups('/'):
        output[group._v_pathname] = {}
        for array in h5file.list_nodes(group, classname='Array'):
            output[group._v_pathname][array.name] = array.read()
    h5file.close()

    outdict= {}
    # first get the
    base_arrs = output['/']

    outdict = {ikey.strip('/'):base_arrs[ikey] for ikey in base_arrs.keys()}

    del output['/']
    for ikey in output.keys():
        sublist = [output[ikey][l] for l in output[ikey].keys()]
        outdict[ikey.strip('/')] = sublist

    return outdict
        #%% Test functions
def Chapmanfunc(z, H_0, Z_0, N_0):
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


def TempProfile(z, T0=1000., z0=100.):
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

def makeconfigfile(fname, beamlist, radarname, simparams_orig):
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
    keys2save = ['IPP', 'IPPsamps', 'TimeLim', 'Pulselength', 'Pulsetype', 'fsnum',
                 'fsden', 'Tint', 'Fitinter',
                 'dtype', 'species', 'numpoints', 'startfile',
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

def readconfigfile(fname, make_amb_bool=False):
    """
        This funciton will read in the ini or yaml files that are used for configuration.

        Args:
            fname - A string containing the file name and location.
            make_amb_bool - A bool to determine if the ambiguity functions should be
                       calculated because they take a lot of time.

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
    p_len = int(sp.round_(f_s*simparams['Pulselength']))
    (pulse, simparams['Pulselength'], sweepid, sweepnums) = makepulse(simparams['Pulsetype'],
                                                                      p_len, t_s)
    simparams['sweepids'] = sweepid
    simparams['sweepnums'] = sweepnums
    simparams['Pulse'] = pulse
    if make_amb_bool:
        simparams['amb_dict'] = make_amb(f_s/ds_fac, simparams['declist'], pulse,
                                         simparams['numpoints'], sweepid)
    simparams['angles'] = angles
    timing_dict = get_timing_dict()
    usweeps = sp.unique(sweepid)

    sig_list = sp.vstack([list(timing_dict[i][1]['signal']) for i in usweeps])
    blank_list = sp.vstack([list(timing_dict[i][1]['blank']) for i in usweeps])
    no_list = sp.vstack([list(timing_dict[i][1]['noise']) for i in usweeps])
    cal_list = sp.vstack([list(timing_dict[i][1]['calibration']) for i in usweeps])
    d_len = [blank_list[:, 1].min(), sig_list[:, 1].max()]
    n_len = [no_list[:, 0].min(), no_list[:, 1].max()]
    c_len = [cal_list[:, 0].min(), cal_list[:, 1].max()]
    simparams['datasamples'] = d_len
    simparams['noisesamples'] = n_len
    simparams['calsamples'] = c_len
    simparams['Timing_Dict'] = {i:timing_dict[i][1] for i in usweeps}
    rng_samprate = t_s*v_C_0*1e-3/2.
    rng_samprateds = rng_samprate/ds_fac
    rng_gates = sp.arange(d_len[0], d_len[1])*rng_samprate
    rng_gates_ds = rng_gates[::ds_fac]
    simparams['Timevec'] = sp.arange(0, time_lim, simparams['Fitinter'])
    simparams['Rangegates'] = rng_gates
    if 'lagtype' not in simparams.keys():
        simparams['lagtype'] = 'centered'

    plen_ds = int(p_len/ds_fac)
    sumrule = makesumrule(simparams['Pulsetype'], plen_ds, simparams['lagtype'])
    simparams['SUMRULE'] = sumrule
    minrg = plen_ds-1
    maxrg = len(rng_gates_ds)-plen_ds+1

    simparams['Rangegatesfinal'] = rng_gates_ds[minrg:maxrg]
    # HACK need to move this to the sensor constants part
    sensdict['CalDiodeTemp'] = 1689.21
    # Set the number of noise samples to IPP

    if ('startfile' in simparams.keys() and simparams['startfile']) and simparams['Pulsetype'].lower() != 'barker':
        relpath = Path(simparams['startfile'])
        if not relpath.is_absolute():
            # Some times the ini files may split the strings of the start
            # file because of white space in file names.
            if type(simparams['startfile']) is list:
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
# Millstone specific things

def get_timing_dict():
    return         {-1   : ('zero_mode', {'full' : (0,0), 'tx' : (0,0), 'blank' : (0,0), 'clutter' : (0,0), 'signal' : (0,0), 'noise' : (0,0), 'txnoise' : (0,0), 'calibration' : (0,0)}),

                    # Alternating Code, for the moment it is good we only use one...
                    1    : ('a480_30_8910_1', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    2    : ('a480_30_8910_2', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    3    : ('a480_30_8910_3', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    4    : ('a480_30_8910_4', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    5    : ('a480_30_8910_5', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    6    : ('a480_30_8910_6', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    7    : ('a480_30_8910_7', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    8    : ('a480_30_8910_8', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    9    : ('a480_30_8910_9', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    10   : ('a480_30_8910_10', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    11   : ('a480_30_8910_11', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    12   : ('a480_30_8910_12', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    13   : ('a480_30_8910_13', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    14   : ('a480_30_8910_14', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    15   : ('a480_30_8910_15', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    16   : ('a480_30_8910_16', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    17   : ('a480_30_8910_17', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    18   : ('a480_30_8910_18', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    19   : ('a480_30_8910_19', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    20   : ('a480_30_8910_20', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    21   : ('a480_30_8910_21', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    22   : ('a480_30_8910_22', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    23   : ('a480_30_8910_23', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    24   : ('a480_30_8910_24', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    25   : ('a480_30_8910_25', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    26   : ('a480_30_8910_26', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    27   : ('a480_30_8910_27', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    28   : ('a480_30_8910_28', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    29   : ('a480_30_8910_29', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    30   : ('a480_30_8910_30', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    31   : ('a480_30_8910_31', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),
                    32   : ('a480_30_8910_32', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,540), 'clutter' : (540,1100), 'signal' : (1100,7180), 'noise' : (7180,8180), 'txnoise' : (7180,8180), 'calibration' : (8180,8810)}),

                    # 480 usec single pulse, used interleaved with alternating code
                    300 : ('s480_8910', {'full' : (0,8910), 'tx' : (0,480), 'blank' : (0,700), 'clutter' : (700,800), 'signal' : (800,7100), 'noise' : (7100,7700), 'txnoise' : (7100,7700), 'calibration' : (7700,8300)}),

                    # 2000 usec single pulse
                    500 : ('std_s2000_34600', {'full' : (0,34599), 'tx' : (100,2101), 'blank' : (0,2500), 'clutter' : (2501,2700), 'signal' : (2701,28100), 'noise' : (28101,30100),'txnoise' : (28101,30100),  'calibration' : (30101,32100)}),

                    # 2000 usec single pulse extended blank
                    800 : ('scan_s2000_34600', {'full' : (0,34599), 'tx' : (100,2101), 'blank' : (0,3600), 'clutter' : (3601,3700), 'signal' : (3701,28100), 'noise' : (28101,30100),'txnoise' : (28101,30100),  'calibration' : (30101,32100)}),

                    # 1280 usec single pulse
                    502 : ('std_s1280_22400', {'full' : (0,22399), 'tx' : (100,1381), 'blank' : (0,1500), 'clutter' : (1501,2780), 'signal' : (2781,18100), 'noise' : (18101,20100), 'txnoise' : (18101,20100), 'calibration' : (20101,20700)}),

                    # 1280 usec single pulse extended blank
                    802 : ('scan_s1280_22400', {'full' : (0,22399), 'tx' : (100,1381), 'blank' : (0,2300), 'clutter' : (2301,2780), 'signal' : (2781,18100), 'noise' : (18101,20100), 'txnoise' : (18101,20100), 'calibration' : (20101,20700)}),

                    # 960 usec single pulse
                    516 : ('s960_17000', {'full' : (0,16999), 'tx' : (100,1061), 'blank' : (0,962), 'clutter' : (962,2660), 'signal' : (2661,13000), 'noise' : (13001,16000), 'txnoise' : (13001,16000), 'calibration' : (16001,16600)}),

                    # 960 usec single pulse extended blank
                    816 : ('scan_s960_17000', {'full' : (0,16999), 'tx' : (100,1061), 'blank' : (0,1700), 'clutter' : (1701,2660), 'signal' : (2661,13000), 'noise' : (13001,16000), 'txnoise' : (13001,16000), 'calibration' : (16001,16600)}),

                    # 640 usec single pulse
                    504 : ('std_s640_11600', {'full' : (0,11599), 'tx' : (100,741), 'blank' : (0,900), 'clutter' : (901,1540), 'signal' : (1541,9100), 'noise' : (9101,10100), 'txnoise' : (9101,10100), 'calibration' : (10101,10700)}),

                    # 410 usec single pulse
                    506 : ('std_s410_8000', {'full' : (0,7999), 'tx' : (100,511), 'blank' : (0,700), 'clutter' : (701,1100), 'signal' : (1101,6100), 'noise' : (7180,7300), 'txnoise' : (7180,7300), 'calibration' : (7301,7800)}),

                    # 410 usec single pulse alternate IPP length
                    508 : ('std_s410_7700', {'full' : (0,7699), 'tx' : (100,511), 'blank' : (0,700), 'clutter' : (701,1100), 'signal' : (1101,5900), 'noise' : (5901,7100), 'txnoise' : (5901,7100), 'calibration' : (7101,7600)}),

                    # 400 usec single pulse mode to replace the 410

                    524 : ('s400_7700', {'full' : (0,7699), 'tx' : (100,501), 'blank' : (0,700), 'clutter' : (701,1100), 'signal' : (1101,5900), 'noise' : (5901,7100), 'txnoise' : (5901,7100), 'calibration' : (7101,7600)}),

                    # 40 usec single pulse
                    514 : ('std_s40_3800', {'full' : (0,3799), 'tx' : (100,141), 'blank' : (0,300), 'clutter' : (301,350), 'signal' : (351,2800), 'noise' : (2801,3200), 'txnoise' : (2801,3200), 'calibration' : (3201,3700)}),

                    # 40 usec single pulse
                    1027 : ('s40_2000', {'full' : (0,1999), 'tx' : (100,141), 'blank' : (0,200), 'clutter' : (201,600), 'signal' : (601,1100), 'noise' : (1101,1400), 'txnoise' : (1101,1400), 'calibration' : (1401,1700)}),

                    # 40 usec single pulse extended IPP
                    1028 : ('s40_3000', {'full' : (0,2999), 'tx' : (100,141), 'blank' : (0,200), 'clutter' : (201,600), 'signal' : (601,2100), 'noise' : (2101,2400), 'txnoise' : (2101,2400), 'calibration' : (2401,2700)}),

                    # 40 usec single pulse extended IPP
                    1029 : ('s40_3600', {'full' : (0,3599), 'tx' : (100,141), 'blank' : (0,200), 'clutter' : (201,600), 'signal' : (601,3100), 'noise' : (3101,3400), 'txnoise' : (3101,3400), 'calibration' : (3401,3550)}),

                    # 60 usec single pulse duty cycle matched to s40_2000 for interleaving
                    1038 : ('s60_3000', {'full' : (0,2999), 'tx' : (100,161), 'blank' : (0,200), 'clutter' : (201,600), 'signal' : (601,2100), 'noise' : (2101,2400), 'txnoise' : (2101,2400), 'calibration' : (2401,2700)}),

                    # 300 usec single pulse
                    510 : ('std_s300_6200', {'full' : (0,6199), 'tx' : (100,300), 'blank' : (0,600), 'clutter' : (601,1000), 'signal' : (1001,5200), 'noise' : (5201,5600), 'txnoise' : (5201,5600), 'calibration' : (5601,6100)}),

                    # 40 usec single pulse
                    1027 : ('s40_2000', {'full' : (0,1999), 'tx' : (10,40), 'blank' : (0,100), 'clutter' : (101,450), 'signal' : (451,1100), 'noise' : (1101,1400), 'txnoise' : (1101,1400), 'calibration' : (1401,1700)}),

                    # 390 usec single pulse
                    521 : ('s390_7300', {'full' : (0,7299), 'tx' : (100,401), 'blank' : (0,700), 'clutter' : (701,1100),  'signal' : (1101, 5299), 'noise' : (5300,6669), 'txnoise': (5300,6669), 'calibration' : (6670,7200)}),

                    # 390 usec Barker code
                    621 : ('b390_30_7300', {'full' : (0,7299), 'tx' : (100,401), 'blank' : (0,700), 'clutter' : (701,1100),  'signal' : (1101, 5299), 'noise' : (5300,6669), 'txnoise': (5300,6669), 'calibration' : (6670,7200)}),

                    # calibration mode
                    2147483647 : ('calibration', {'full' : (0,34599), 'tx' : (0,100), 'blank' : (0,2500), 'clutter' : (0,100), 'signal' : (0,100), 'noise' : (2501,17100), 'txnoise' : (2501,17100), 'calibration' : (17101,32100)}),

                    # standby mode
                    0 : ('standby', {'full' : (0,34599), 'tx' : (100,2101), 'blank' : (0,2500), 'clutter' : (2501,2700), 'signal' : (2701,28100), 'noise' : (28101,30100),'txnoise' : (28101,30100),  'calibration' : (30101,32100)})

                    }
