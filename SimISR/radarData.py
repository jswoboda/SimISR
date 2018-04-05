#!/usr/bin/env python
"""
radarData.py
This file holds the RadarData class that hold the radar data and processes it.

@author: John Swoboda
"""
import ipdb
import scipy as sp
# My modules
from isrutilities.physConstants import v_C_0, v_Boltz
from SimISR import Path
from SimISR.IonoContainer import IonoContainer
from SimISR.utilFunctions import CenteredLagProduct, MakePulseDataRepLPC,dict2h5,h52dict,readconfigfile, BarkerLag, update_progress
#from SimISR import specfunctions
#from SimISR.analysisplots import plotspecsgen

class RadarDataFile(object):
    """
        This class will will take the ionosphere class and create radar data both
        at the IQ and fitted level.

        Variables
        simparams - A dictionary that holds simulation parameters the keys are the
            following
            'angles': Angles (in degrees) for the simulation.  The az and el angles
                are stored in a list of tuples [(az_0,el_0),(az_1,el_1)...]
            'IPP' : Interpulse period in seconds, this is the IPP from position to
                position.  In other words if a IPP is 10 ms and there are 10 beams it will
                take 100 ms to go through all of the beams.
            'Tint'  This is the integration time in seconds.
            'Timevec': This is the time vector for each frame.  It is referenced to
                the begining of the frame.  Also in seconds
            'Pulse': The shape of the pulse.  This is a numpy array.
            'TimeLim': The length of time that the experiment will run, which cooresponds
                to how many pulses per position are used.
        sensdict - A dictionary that holds the sensor parameters.
        rawdata: This is a NbxNpxNr numpy array that holds the raw IQ data.
        rawnoise: This is a NbxNpxNr numpy array that holds the raw noise IQ data.
    """


    def __init__(self, Ionodict, inifile, outdir, outfilelist=None):
        """
            This function will create an instance of the RadarData class.  It will
            take in the values and create the class and make raw IQ data.
            Inputs:
            sensdict - A dictionary of sensor parameters
            angles - A list of tuples which the first position is the az angle
                and the second position is the el angle.
            IPP - The interpulse period in seconds represented as a float.
            Tint - The integration time in seconds as a float.  This will be the
            integration time of all of the beams.
            time_lim - The length of time of the simulation the number of time points
                will be calculated.
            pulse - A numpy array that represents the pulse shape.
            rng_lims - A numpy array of length 2 that holds the min and max range
                that the radar will cover.
        """
        (sensdict, simparams) = readconfigfile(inifile)
        self.simparams = simparams
        N_angles = len(self.simparams['angles'])

        NNs = int(self.simparams['NNs'])
        self.sensdict = sensdict
        Npall = sp.floor(self.simparams['TimeLim']/self.simparams['IPP'])
        Npall = int(sp.floor(Npall/N_angles)*N_angles)
        Np = Npall/N_angles

        print("All spectrums created already")
        filetimes = Ionodict.keys()
        filetimes.sort()
        ftimes = sp.array(filetimes)
        simdtype = self.simparams['dtype']
        pulsetimes = sp.arange(Npall)*self.simparams['IPP'] +ftimes.min()
        pulsefile = sp.array([sp.where(itimes-ftimes >= 0)[0][-1] for itimes in pulsetimes])
        # differentiate between phased arrays and dish antennas
        if sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:

            beams = sp.tile(sp.arange(N_angles), Npall/N_angles)
        else:

            # for dish arrays
            brate = simparams['beamrate']
            beams2 = sp.repeat(sp.arange(N_angles), brate)
            beam3 = sp.concatenate((beams2, beams2[::-1]))
            ntile = int(sp.ceil(Npall/len(beam3)))
            leftover = int(Npall-ntile*len(beam3))
            if ntile > 0:
                beams = sp.tile(beam3, ntile)
                beams = sp.concatenate((beams, beam3[:leftover]))
            else:
                beams = beam3[:leftover]

        pulsen = sp.repeat(sp.arange(Np), N_angles)
        pt_list = []
        pb_list = []
        pn_list = []
        fname_list = []
        self.datadir = outdir
        self.maindir = outdir.parent
        self.procdir = self.maindir/'ACF'
        Nf = len(filetimes)
        progstr = 'Data from {:d} of {:d} being processed Name: {:s}.'
        if outfilelist is None:
            print('\nData Now being created.')

            Noisepwr = v_Boltz*sensdict['Tsys']*sensdict['BandWidth']
            self.outfilelist = []
            for ifn, ifilet in enumerate(filetimes):

                outdict = {}
                ifile = Ionodict[ifilet]
                ifilename = Path(ifile).name
                update_progress(float(ifn)/Nf, progstr.format(ifn, Nf, ifilename))
                curcontainer = IonoContainer.readh5(ifile)
                if ifn == 0:
                    self.timeoffset = curcontainer.Time_Vector[0, 0]
                pnts = pulsefile == ifn
                pt = pulsetimes[pnts]
                pb = beams[pnts]
                pn = pulsen[pnts].astype(int)
                rawdata = self.__makeTime__(pt, curcontainer.Time_Vector,
                                            curcontainer.Sphere_Coords,
                                            curcontainer.Param_List, pb)
                d_shape = rawdata.shape
                n_tempr = sp.random.randn(*d_shape).astype(simdtype)
                n_tempi = 1j*sp.random.randn(*d_shape).astype(simdtype)
                noise = sp.sqrt(Noisepwr/2)*(n_tempr+n_tempi)
                outdict['AddedNoise'] = noise
                outdict['RawData'] = rawdata+noise
                outdict['RawDatanonoise'] = rawdata
                outdict['NoiseData'] = sp.sqrt(Noisepwr/2)*(sp.random.randn(len(pn), NNs).astype(simdtype)+
                                                            1j*sp.random.randn(len(pn), NNs).astype(simdtype))
                outdict['Pulses'] = pn
                outdict['Beams'] = pb
                outdict['Time'] = pt
                fname = '{0:d} RawData.h5'.format(ifn)
                newfn = self.datadir/fname
                self.outfilelist.append(str(newfn))
                dict2h5(str(newfn), outdict)

                #Listing info
                pt_list.append(pt)
                pb_list.append(pb)
                pn_list.append(pn)
                fname_list.append(fname)
            infodict = {'Files':fname_list, 'Time':pt_list, 'Beams':pb_list, 'Pulses':pn_list}
            dict2h5(str(outdir.joinpath('INFO.h5')), infodict)

        else:
            infodict = h52dict(str(outdir.joinpath('INFO.h5')))
            alltime = sp.hstack(infodict['Time'])
            self.timeoffset = alltime.min()
            self.outfilelist = outfilelist


#%% Make functions
    def __makeTime__(self,pulsetimes,spectime,Sphere_Coords,allspecs,beamcodes):
        """This is will make raw radar data for a given time period and set of
        spectrums. This is an internal function called by __init__.
        Inputs-
        self - RadarDataFile object.
        pulsetimes - The time the pulses are sent out in reference to the spectrums.
        spectime - The times for the spectrums.
        Sphere_Coords - An Nlx3 array that holds the spherical coordinates of the spectrums.
        allspecs - An NlxNdtimexNspec array that holds the spectrums.
        beamcodes - A NBx4 array that holds the beam codes along with the beam location
        in az el along with a system constant in the array. """

        range_gates = self.simparams['Rangegates']
        pulse = self.simparams['Pulse']
        sensdict = self.sensdict
        pulse2spec = sp.array([sp.where(itimes-spectime >= 0)[0][-1] for itimes in pulsetimes])
        Np = len(pulse2spec)
        lp_pnts = len(pulse)
        samp_num = sp.arange(lp_pnts)
        #N_rg = len(range_gates)# take the size
        N_samps = len(range_gates)
        angles = self.simparams['angles']
        Nbeams = len(angles)
        rho = Sphere_Coords[:, 0]
        Az = Sphere_Coords[:, 1]
        El = Sphere_Coords[:, 2]
        rng_len = self.sensdict['t_s']*v_C_0*1e-3/2.
        speclen = allspecs.shape[-1]
        simdtype = self.simparams['dtype']
        out_data = sp.zeros((Np, N_samps), dtype=simdtype)
        weights = {ibn:self.sensdict['ArrayFunc'](Az, El, ib[0], ib[1], sensdict['Angleoffset'])
                   for ibn, ib in enumerate(angles)}
        for istn in range(len(spectime)):
            for ibn in range(Nbeams):
                #print('\t\t Making Beam {0:d} of {1:d}'.format(ibn, Nbeams))
                weight = weights[ibn]
                for isamp in range(N_samps):
                    range_g = range_gates[isamp]
                    # if isamp >= :
                    if range_g <= 0:
                        continue
                    range_m = range_g*1e3
                    rnglims = [range_g-rng_len/2.0, range_g+rng_len/2.0]
                    rangelog = (rho >= rnglims[0])&(rho < rnglims[1])
                    # Get the number of points covered
                    cur_pnts = samp_num+isamp
                    keep_pnt = sp.logical_and(cur_pnts >= 0, cur_pnts < N_samps)
                    cur_pnts = cur_pnts[keep_pnt]
                    # This is a nearest neighbors interpolation for the
                    # spectrums in the range domain
                    if sp.sum(rangelog) == 0:
                        minrng = sp.argmin(sp.absolute(range_g-rho))
                        rangelog[minrng] = True

                    #create the weights and weight location based on the beams pattern.
                    weight_cur = weight[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()
                    specsinrng = allspecs[rangelog]
                    if specsinrng.ndim == 3:
                        specsinrng = specsinrng[:, istn]
                    elif specsinrng.ndim == 2:
                        specsinrng = specsinrng[istn]
                    specsinrng = specsinrng*sp.tile(weight_cur[:, sp.newaxis], (1, speclen))
                    cur_spec = specsinrng.sum(0)
                    # based off new way of calculating
                    pow_num = sensdict['Pt']*sensdict['Ksys'][ibn]*sensdict['t_s']
                    pow_den = range_m**2
                    curdataloc = sp.where(sp.logical_and((pulse2spec == istn),
                                                         (beamcodes == ibn)))[0]
                    # create data
                    if len(curdataloc) == 0:
                        print('\t\t No data for {0:d} of {1:d} in this time period'.format(ibn, Nbeams))
                        continue
                    cur_pulse_data = MakePulseDataRepLPC(pulse, cur_spec, 20,
                                                         len(curdataloc), numtype=simdtype)
                    cur_pulse_data = cur_pulse_data*sp.sqrt(pow_num/pow_den)

                    for idatn, idat in enumerate(curdataloc):
                        cursum = cur_pulse_data[idatn, cur_pnts-isamp]+out_data[idat, cur_pnts]
                        out_data[idat, cur_pnts] = cursum
        return out_data
        #%% Processing
    def processdataiono(self):
        """ This will perform the the data processing and create the ACF estimates
        for both the data and noise but put it in an Ionocontainer.
        Inputs:

        Outputs:
        Ionocontainer- This is an instance of the ionocontainer class that will hold the acfs.
        """
        (datalags, noiselags) = self.processdata()
        return lagdict2ionocont(datalags, noiselags, self.sensdict, self.simparams,
                                datalags['Time'])

    def processdata(self):
        """ This will perform the the data processing and create the ACF estimates
        for both the data and noise.
        Inputs:
        timevec - A numpy array of times in seconds where the integration will begin.
        inttime - The integration time in seconds.
        lagfunc - A function that will make the desired lag products.
        Outputs:
        DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' that holds
        the numpy arrays of the data.
        NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' that holds
        the numpy arrays of the data.
        """
        timevec = self.simparams['Timevec'] +self.timeoffset
        inttime = self.simparams['Tint']
        # Get array sizes

        NNs = int(self.simparams['NNs'])
        range_gates = self.simparams['Rangegates']
        N_samps = len(range_gates)# take the size
        pulse = self.simparams['Pulse']
        Pulselen = len(pulse)
        N_rg = N_samps - Pulselen+1
        simdtype = self.simparams['dtype']
        Ntime = len(timevec)

        if 'outangles' in self.simparams.keys():
            Nbeams = len(self.simparams['outangles'])
            inttime = inttime
        else:
            Nbeams = len(self.simparams['angles'])

        # Choose type of processing
        if self.simparams['Pulsetype'].lower() == 'barker':
            lagfunc = BarkerLag
            Nlag = 1
        else:
            lagfunc = CenteredLagProduct
            Nlag = Pulselen
        # initialize output arrays
        outdata = sp.zeros((Ntime, Nbeams, N_rg, Nlag), dtype=simdtype)
        outaddednoise = sp.zeros((Ntime, Nbeams, N_rg, Nlag), dtype=simdtype)
        outnoise = sp.zeros((Ntime, Nbeams, NNs-Pulselen+1, Nlag), dtype=simdtype)
        pulses = sp.zeros((Ntime, Nbeams))
        pulsesN = sp.zeros((Ntime, Nbeams))
        timemat = sp.zeros((Ntime, 2))
        Ksysvec = self.sensdict['Ksys']
        # set up arrays that hold the location of pulses that are to be processed together
        infoname = self.datadir / 'INFO.h5'
        # Just going to assume that the info file is in the directory
        infodict = h52dict(str(infoname))
        flist = infodict['Files']
        file_list = [str(self.datadir/i) for i in flist]
        pulsen_list = infodict['Pulses']
        beamn_list = infodict['Beams']
        time_list = infodict['Time']
        file_loclist = [ifn*sp.ones(len(ifl)) for ifn, ifl in enumerate(beamn_list)]


        pulsen = sp.hstack(pulsen_list).astype(int)# pulse number
        beamn = sp.hstack(beamn_list).astype(int)# beam numbers
        ptimevec = sp.hstack(time_list).astype(float)# time of each pulse
        file_loc = sp.hstack(file_loclist).astype(int)# location in the file

        # run the time loop
        print("Forming ACF estimates")

        # For each time go through and read only the necisary files
        for itn, it in enumerate(timevec):
            update_progress(float(itn)/Ntime, "Time {0:d} of {1:d}".format(itn, Ntime))
            # do the book keeping to determine locations of data within the files
            cur_tlim = (it, it+inttime)
            curcases = sp.logical_and(ptimevec >= cur_tlim[0], ptimevec < cur_tlim[1])
            if  not sp.any(curcases):
                prog_str = "No pulses for time {0:d} of {1:d}, lagdata adjusted accordinly"
                update_progress(float(itn)/Ntime, prog_str.format(itn, Ntime))
                outdata = outdata[:itn]
                outnoise = outnoise[:itn]
                pulses = pulses[:itn]
                pulsesN = pulsesN[:itn]
                timemat = timemat[:itn]
                continue
            pulseset = set(pulsen[curcases])
            poslist = [sp.where(pulsen == item)[0] for item in pulseset]
            pos_all = sp.hstack(poslist)
            try:
                pos_all = sp.hstack(poslist)
                curfileloc = file_loc[pos_all]
            except:
                ipdb.set_trace()
            # Find the needed files and beam numbers
            curfiles = set(curfileloc)
            beamlocs = beamn[pos_all]
            timemat[itn, 0] = ptimevec[pos_all].min()
            timemat[itn, 1] = ptimevec[pos_all].max()
            # cur data pulls out all data from all of the beams and posisions
            curdata = sp.zeros((len(pos_all), N_samps), dtype=simdtype)
            curaddednoise = sp.zeros((len(pos_all), N_samps), dtype=simdtype)
            curnoise = sp.zeros((len(pos_all), NNs), dtype=simdtype)
            # Open files and get required data
            # XXX come up with way to get open up new files not have to reread in data that is already in memory
            for ifn in curfiles:
                curfileit = [sp.where(pulsen_list[ifn] == item)[0] for item in pulseset]
                curfileitvec = sp.hstack(curfileit)
                ifile = file_list[ifn]
                curh5data = h52dict(ifile)
                file_arlocs = sp.where(curfileloc == ifn)[0]
                curdata[file_arlocs] = curh5data['RawData'][curfileitvec]

                curaddednoise[file_arlocs] = curh5data['AddedNoise'].astype(simdtype)[curfileitvec]
                # Read in noise data when you have don't have ACFs

                curnoise[file_arlocs] = curh5data['NoiseData'].astype(simdtype)[curfileitvec]

            # differentiate between phased arrays and dish antennas
            if self.sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:
                # After data is read in form lags for each beam
                for ibeam in range(Nbeams):
                    prog_num = float(itn)/Ntime+float(ibeam)/Ntime/Nbeams
                    update_progress(prog_num, "Beam {0:d} of {1:d}".format(ibeam, Nbeams))
                    beamlocstmp = sp.where(beamlocs == ibeam)[0]
                    pulses[itn, ibeam] = len(beamlocstmp)

                    outdata[itn, ibeam] = lagfunc(curdata[beamlocstmp].copy(),
                                                  numtype=self.simparams['dtype'],
                                                  pulse=pulse, lagtype=self.simparams['lagtype'])

                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outnoise[itn, ibeam] = lagfunc(curnoise[beamlocstmp].copy(),
                                                   numtype=self.simparams['dtype'],
                                                   pulse=pulse, lagtype=self.simparams['lagtype'])
                    outaddednoise[itn, ibeam] = lagfunc(curaddednoise[beamlocstmp].copy(),
                                                        numtype=self.simparams['dtype'],
                                                        pulse=pulse, lagtype=self.simparams['lagtype'])
            else:
                for ibeam, ibeamlist in enumerate(self.simparams['outangles']):
                    prog_num = float(itn)/Ntime+float(ibeam)/Ntime/Nbeams
                    update_progress(prog_num, "Beam {0:d} of {1:d}".format(ibeam, Nbeams))
                    beamlocstmp = sp.where(sp.in1d(beamlocs, ibeamlist))[0]
                    inputdata = curdata[beamlocstmp].copy()
                    noisedata = curnoise[beamlocstmp].copy()
                    noisedataadd = curaddednoise[beamlocstmp].copy()
                    pulses[itn, ibeam] = len(beamlocstmp)
                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outdata[itn, ibeam] = lagfunc(inputdata, numtype=self.simparams['dtype'],
                                                  pulse=pulse, lagtype=self.simparams['lagtype'])
                    outnoise[itn, ibeam] = lagfunc(noisedata, numtype=self.simparams['dtype'],
                                                   pulse=pulse, lagtype=self.simparams['lagtype'])
                    outaddednoise[itn, ibeam] = lagfunc(noisedataadd, numtype=self.simparams['dtype'],
                                                        pulse=pulse, lagtype=self.simparams['lagtype'])
        # Create output dictionaries and output data
        DataLags = {'ACF':outdata, 'Pow':outdata[:, :, :, 0].real, 'Pulses':pulses,
                    'Time':timemat, 'AddedNoiseACF':outaddednoise}
        NoiseLags = {'ACF':outnoise, 'Pow':outnoise[:, :, :, 0].real, 'Pulses':pulsesN,
                     'Time':timemat}
        return(DataLags, NoiseLags)

#%% Make Lag dict to an iono container
def lagdict2ionocont(DataLags, NoiseLags, sensdict, simparams, time_vec):
    """This function will take the data and noise lags and create an instance of the
    Ionocontanier class. This function will also apply the summation rule to the lags.
    Inputs
    DataLags - A dictionary """
    # Pull in Location Data
    angles = simparams['angles']
    ang_data = sp.array([[iout[0], iout[1]] for iout in angles])
    rng_vec = simparams['Rangegates']
    n_samps = len(rng_vec)
    # pull in other data
    pulse = simparams['Pulse']
    p_samps = len(pulse)
    pulsewidth = p_samps*sensdict['t_s']
    txpower = sensdict['Pt']
    if sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:
        Ksysvec = sensdict['Ksys']
    else:

        beamlistlist = sp.array(simparams['outangles']).astype(int)
        inplist = sp.array([i[0] for i in beamlistlist])
        Ksysvec = sensdict['Ksys'][inplist]
        ang_data_temp = ang_data.copy()
        ang_data = sp.array([ang_data_temp[i].mean(axis=0) for i in beamlistlist])

    sumrule = simparams['SUMRULE']
    rng_vec2 = simparams['Rangegatesfinal']
    Nrng2 = len(rng_vec2)
    minrg = 2*p_samps-1+sumrule[0].min()
    maxrg = Nrng2+minrg


    # Copy the lags
    lagsData = DataLags['ACF'].copy()
    # Set up the constants for the lags so they are now
    # in terms of density fluxtuations.
    angtile = sp.tile(ang_data, (Nrng2, 1))
    rng_rep = sp.repeat(rng_vec2, ang_data.shape[0], axis=0)
    coordlist = sp.zeros((len(rng_rep), 3))
    [coordlist[:, 0], coordlist[:, 1:]] = [rng_rep, angtile]
    (Nt, Nbeams, Nrng, Nlags) = lagsData.shape

    # make a range average to equalize out the conntributions from each gate

    plen2 = int(sp.floor(float(p_samps-1)/2))
    samps = sp.arange(0, p_samps, dtype=int)
    rng_lags = CenteredLagProduct(rng_vec[sp.newaxis, :]*1e3, numtype=sp.float64,
                                  pulse=pulse)
    rng_ave = sp.ones_like(rng_lags)

    for isamp in range(Nrng):
        for ilag in range(p_samps):
            sampsred = samps[:p_samps-ilag]
            cursamps = isamp-sampsred

            keepsamps = sp.logical_and(cursamps >= 0, cursamps < Nrng)
            cursamps = cursamps[keepsamps]
            rng_samps = rng_lags[cursamps, ilag]
            keepsamps2 = rng_samps > 0
            if keepsamps2.sum() == 0:
                continue
            rng_samps = rng_samps[keepsamps2]
            rng_ave[isamp, ilag] = 1./(sp.mean(1./(rng_samps)))
    rng_ave_temp = rng_ave.copy()
    # rng_ave = rng_ave[int(sp.floor(plen2)):-int(sp.ceil(plen2))]
    # rng_ave = rng_ave[minrg:maxrg]
    rng3d = sp.tile(rng_ave[sp.newaxis, sp.newaxis, :, :], (Nt, Nbeams, 1, 1))
    ksys3d = sp.tile(Ksysvec[:, sp.newaxis, sp.newaxis, sp.newaxis], (Nt, 1, Nrng, Nlags))
    # rng3d = sp.tile(rng_ave[:, sp.newaxis, sp.newaxis, sp.newaxis], (1, Nlags, Nt, Nbeams))
    # ksys3d = sp.tile(Ksysvec[sp.newaxis, sp.newaxis, sp.newaxis, :], (Nrng2, Nlags, Nt, 1))
    radar2acfmult = rng3d/(pulsewidth*txpower*ksys3d)
    pulses = sp.tile(DataLags['Pulses'][:, :, sp.newaxis, sp.newaxis], (1, 1, Nrng, Nlags))
    time_vec = time_vec[:Nt]
    # Divid lags by number of pulses
    lagsData = lagsData/pulses
    # Set up the noise lags and divid out the noise.
    lagsNoise = NoiseLags['ACF'].copy()
    lagsNoise = sp.mean(lagsNoise, axis=2)
    pulsesnoise = sp.tile(NoiseLags['Pulses'][:, :, sp.newaxis], (1, 1, Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = sp.tile(lagsNoise[:, :, sp.newaxis, :], (1, 1, Nrng, 1))

    #ipdb.set_trace()
    # multiply the data and the sigma by inverse of the scaling from the radar
    lagsData = lagsData*radar2acfmult
    lagsNoise = lagsNoise*radar2acfmult
    # Apply summation rule
    # lags transposed from (time,beams,range,lag)to (range,lag,time,beams)
    lagsData = sp.transpose(lagsData, axes=(2, 3, 0, 1))
    lagsNoise = sp.transpose(lagsNoise, axes=(2, 3, 0, 1))
    lagsDatasum = sp.zeros((Nrng2, Nlags, Nt, Nbeams), dtype=lagsData.dtype)
    lagsNoisesum = sp.zeros((Nrng2, Nlags, Nt, Nbeams), dtype=lagsNoise.dtype)

    for irngnew, irng in enumerate(sp.arange(minrg, maxrg)):
        for ilag in range(Nlags):
            lsumtemp = lagsData[irng+sumrule[0, ilag]:irng+sumrule[1, ilag]+1, ilag].sum(axis=0)
            lagsDatasum[irngnew, ilag] = lsumtemp
            nsumtemp = lagsNoise[irng+sumrule[0, ilag]:irng+sumrule[1, ilag]+1, ilag].sum(axis=0)
            lagsNoisesum[irngnew, ilag] = nsumtemp
    # subtract out noise lags
    lagsDatasum = lagsDatasum-lagsNoisesum

    # Put everything in a parameter list
    Paramdata = sp.zeros((Nbeams*Nrng2, Nt, Nlags), dtype=lagsData.dtype)
    # Put everything in a parameter list
    # transpose from (range,lag,time,beams) to (beams,range,time,lag)
    # lagsDatasum = lagsDatasum*radar2acfmult
    # lagsNoisesum = lagsNoisesum*radar2acfmult
    lagsDatasum = sp.transpose(lagsDatasum, axes=(3, 0, 2, 1))
    lagsNoisesum = sp.transpose(lagsNoisesum, axes=(3, 0, 2, 1))

    # multiply the data and the sigma by inverse of the scaling from the radar
    # lagsDatasum = lagsDatasum*radar2acfmult
    # lagsNoisesum = lagsNoisesum*radar2acfmult

    # Calculate a variance using equation 2 from Hysell's 2008 paper. Done use full covariance matrix because assuming nearly diagonal.
    # Get the covariance matrix
    pulses_s = sp.transpose(pulses, axes=(1, 2 ,0, 3))[:, :Nrng2]
    Cttout = makeCovmat(lagsDatasum, lagsNoisesum, pulses_s, Nlags)

    Paramdatasig = sp.zeros((Nbeams*Nrng2 ,Nt, Nlags, Nlags), dtype=Cttout.dtype)

    curloc = 0
    for irng in range(Nrng2):
        for ibeam in range(Nbeams):
            Paramdata[curloc] = lagsDatasum[ibeam, irng].copy()
            Paramdatasig[curloc] = Cttout[ibeam, irng].copy()
            curloc += 1
    ionodata = IonoContainer(coordlist, Paramdata, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags)*sensdict['t_s'])
    ionosigs = IonoContainer(coordlist, Paramdatasig, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags**2).reshape(Nlags, Nlags)*sensdict['t_s'])
    return (ionodata, ionosigs)

def makeCovmat(lagsDatasum, lagsNoisesum, pulses_s, Nlags):
    """
        Makes the covariance matrix for the lags given the noise acf and number
        of pulses.
    """
    axvec = sp.roll(sp.arange(lagsDatasum.ndim), 1)
    # Get the covariance matrix
    R = sp.transpose(lagsDatasum/sp.sqrt(2.*pulses_s), axes=axvec)
    Rw = sp.transpose(lagsNoisesum/sp.sqrt(2.*pulses_s), axes=axvec)
    l = sp.arange(Nlags)
    T1, T2 = sp.meshgrid(l, l)
    R0 = R[sp.zeros_like(T1)]
    Rw0 = Rw[sp.zeros_like(T1)]
    Td = sp.absolute(T1-T2)
    Tl = T1>T2
    R12 = R[Td]
    R12[Tl] = sp.conjugate(R12[Tl])
    Rw12 = Rw[Td]
    Rw12[Tl] = sp.conjugate(Rw12[Tl])
    Ctt = R0*R12+R[T1]*sp.conjugate(R[T2])+Rw0*Rw12+Rw[T1]*sp.conjugate(Rw[T2])
    avecback = sp.roll(sp.arange(Ctt.ndim), -2)
    Cttout = sp.transpose(Ctt, avecback)
    return Cttout
