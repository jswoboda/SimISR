#!/usr/bin/env python
"""
radarData.py
This file holds the RadarData class that hold the radar data and processes it.

@author: John Swoboda
"""
import pdb
import scipy as sp
# My modules
from isrutilities.physConstants import v_C_0, v_Boltz
from SimISR import Path
from SimISR.IonoContainer import IonoContainer
from SimISR.utilFunctions import CenteredLagProduct, MakePulseDataRepLPC,dict2h5,h52dict,readconfigfile, BarkerLag, update_progress
import digital_rf as drf
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

        self.sensdict = sensdict
        Npall = sp.floor(self.simparams['TimeLim']/self.simparams['IPP'])
        Npall = int(sp.floor(Npall/N_angles)*N_angles)
        Np = Npall/N_angles

        ippsamps = self.simparams['IPPsamps']
        d_samps = self.simparams['datasamples']
        n_samps = self.simparams['noisesamples']
        c_samps = self.simparams['calsamples']
        n_cal = c_samps[1] - c_samps[0]

        dec_list = self.simparams['declist']
        ds_fac = float(sp.prod(dec_list))
        filetimes = Ionodict.keys()
        filetimes.sort()
        ftimes = sp.array(filetimes)
        simdtype = self.simparams['dtype']
        pulsetimes = sp.arange(Npall)*self.simparams['IPP'] +ftimes.min()
        pulsefile = sp.array([sp.where(itimes-ftimes >= 0)[0][-1] for itimes in pulsetimes])
        pulse_full = self.simparams['Pulse']

        #digital rf stuff

        sample_rate_numerator = self.simparams['fsnum']
        sample_rate_denominator = self.simparams['fsden']
        sample_rate = sp.longdouble(sample_rate_numerator) / sample_rate_denominator
        start_global_index = int(sample_rate*filetimes[0])
        dtype_str = 'complex64'  # complex64
        sub_cadence_secs = 1000  # Number of seconds of data in a subdirectory - typically MUCH larger
        file_cadence_millisecs = 10000  # Each file will have up to 400 ms of data
        compression_level = 1  # low level of compression
        checksum = False  # no checksum
        is_complex = True  # complex values
        is_continuous = True
        num_subchannels = 1  # only one subchannel
        marching_periods = False  # no marching periods when writing
        uuid = "SimISRDRFZenith"
        drfdir = outdir.parent.joinpath('drfdata', 'rf_data', 'zenith-l')
        data_object = drf.DigitalRFWriter(str(drfdir), dtype_str, sub_cadence_secs,
                                          file_cadence_millisecs, start_global_index,
                                          sample_rate_numerator, sample_rate_denominator,
                                          uuid, compression_level, checksum,
                                          is_complex, num_subchannels,
                                          is_continuous, marching_periods)
        drfdirtx = outdir.parent.joinpath('drfdata', 'rf_data', 'tx-h')
        data_object_tx = drf.DigitalRFWriter(str(drfdirtx), dtype_str, sub_cadence_secs,
                                          file_cadence_millisecs, start_global_index,
                                          sample_rate_numerator, sample_rate_denominator,
                                          uuid, compression_level, checksum,
                                          is_complex, num_subchannels,
                                          is_continuous, marching_periods)
        # Noise Scaling
        noisepwr = v_Boltz*sensdict['Tsys']*sample_rate
        calpwr = v_Boltz*sensdict['CalDiodeTemp']*sample_rate
        # digital metadata
        #TODO temp for original data
        # antenna control

        dmddir = outdir.parent.joinpath('drfdata', 'metadata')
        acmdir = dmddir.joinpath('antenna_control_metadata')
        acmdict = {'misa_elevation': 88.000488281200006,
                   'cycle_name': 'zenith_record_cycle_0',
                   'misa_azimuth': 178.000488281,
                   'rx_antenna': 'ZENITH',
                   'tx_antenna': 'ZENITH'}
        acmobj = drf.DigitalMetadataWriter(str(acmdir), 3600, 60,
                                           sample_rate_numerator,
                                           sample_rate_denominator,
                                           'ant')
        # id metadata
        iddir = dmddir.joinpath('id_metadata')
        idmdict = {'sweepid': 300, 'sample_rate': sample_rate,
                   'modeid': 100000001, 'sweepnum': 0}
        idmobj = drf.DigitalMetadataWriter(str(iddir), 3600, 1,
                                           sample_rate_numerator, sample_rate_denominator,
                                           'md')
        # power mete metadata
        pmdir = dmddir.joinpath('powermeter')
        pmmdict = {'zenith_power': self.sensdict['Pt'], 'misa_power': 3110.3}
        pmmobj = drf.DigitalMetadataWriter(str(pmdir), 3600, 5, 1, 1, 'power')
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
        if outfilelist is None:
            print('\nData Now being created.')

            self.outfilelist = []
            for ifn, ifilet in enumerate(filetimes):

                outdict = {}
                ifile = Ionodict[ifilet]
                ifilename = Path(ifile).name
                progstr1 = 'Data from {:d} of {:d} being created Name: {:s}.'
                update_progress(float(ifn)/Nf, progstr1.format(ifn, Nf, ifilename))
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
                rawdata_us = rawdata.copy()
                for idec in dec_list:
                    rawdata_us = sp.signal.resample(rawdata_us, rawdata_us.shape[1]*idec, axis=1)
                n_pulse_cur = rawdata.shape[0]
                alldata = sp.random.randn(n_pulse_cur, ippsamps) + 1j*sp.random.randn(n_pulse_cur, ippsamps)
                alldata = sp.sqrt(noisepwr/2.)*alldata
                noisedata = alldata[:, d_samps[0]:d_samps[1]]
                alldata[:, d_samps[0]:d_samps[1]] = rawdata_us + alldata[:, d_samps[0]:d_samps[1]]
                caldata = sp.sqrt(calpwr/2)*(sp.random.randn(n_pulse_cur, n_cal).astype(simdtype)+
                                             1j*sp.random.randn(n_pulse_cur, n_cal).astype(simdtype))
                alldata[:, c_samps[0]:c_samps[1]] = alldata[:, c_samps[0]:c_samps[1]] + caldata
                alldata = alldata/sp.sqrt(calpwr)
                rawdata_ds = rawdata_us.copy()/sp.sqrt(calpwr)
                noisedata_ds = noisedata.copy()/sp.sqrt(calpwr)

                alldata_ds = alldata.copy()
                for idec in dec_list:
                    rawdata_ds = sp.signal.resample(rawdata_ds, rawdata_ds.shape[1]/idec, axis=1)
                    noisedata_ds = sp.signal.resample(noisedata_ds, noisedata_ds.shape[1]/idec, axis=1)
                    alldata_ds = sp.signal.resample(alldata_ds, alldata_ds.shape[1]/idec, axis=1)
                outdict['AddedNoise'] = noisedata_ds
                outdict['RawData'] = alldata_ds[:, int(d_samps[0]/ds_fac):int(d_samps[1]/ds_fac)]
                outdict['RawDatanonoise'] = rawdata_ds
                outdict['NoiseData'] = alldata_ds[:, int(n_samps[0]/ds_fac):int(n_samps[1]/ds_fac)]
                outdict['CalData'] = alldata_ds[:, int(c_samps[0]/ds_fac):int(c_samps[1]/ds_fac)]
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
                #transmitdata
                tx_data = sp.zeros_like(alldata).astype('complex64')
                tx_data[:, :len(pulse_full)] = pulse_full.astype('complex64')
                data_object_tx.rf_write(tx_data.flatten())
                # extend array for digital rf to flattend array
                data_object.rf_write(alldata.flatten().astype('complex64'))
                id_strt = int(idmobj.get_samples_per_second()*ifilet)
                dmdplist = sp.arange(n_pulse_cur, dtype=int)*ippsamps + id_strt
                acmobj.write(int(acmobj.get_samples_per_second()*ifilet), acmdict)
                idmobj.write(dmdplist, idmdict)
                pmmobj.write(int(pmmobj.get_samples_per_second()*ifilet), pmmdict)

            infodict = {'Files':fname_list, 'Time':pt_list, 'Beams':pb_list, 'Pulses':pn_list}
            dict2h5(str(outdir.joinpath('INFO.h5')), infodict)
            data_object.close()
            data_object_tx.close()
        else:
            infodict = h52dict(str(outdir.joinpath('INFO.h5')))
            alltime = sp.hstack(infodict['Time'])
            self.timeoffset = alltime.min()
            self.outfilelist = outfilelist


#%% Make functions
    def __makeTime__(self, pulsetimes, spectime, Sphere_Coords, allspecs, beamcodes):
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
        sensdict = self.sensdict
        samp_range = self.simparams['datasamples']
        ds_fac = sp.prod(self.simparams['declist'])
        f_s = float(self.simparams['fsnum'])/self.simparams['fsden']/ds_fac
        t_s = float(ds_fac*self.simparams['fsden'])/self.simparams['fsnum']
        pulse = self.simparams['Pulse'][::ds_fac]
        #HACK number of lpc points connected to ratio of sampling frequency and
        # notial ion-line spectra with a factor of 10.
        nlpc = int(10*f_s/20e3) + 1
        pulse2spec = sp.array([sp.where(itimes-spectime >= 0)[0][-1] for itimes in pulsetimes])
        n_pulses = len(pulse2spec)
        lp_pnts = len(pulse)
        samp_num = sp.arange(lp_pnts)
        n_samps = (samp_range[1]-samp_range[0])/ds_fac
        angles = self.simparams['angles']
        n_beams = len(angles)
        rho = Sphere_Coords[:, 0]
        Az = Sphere_Coords[:, 1]
        El = Sphere_Coords[:, 2]

        rng_len = t_s*v_C_0*1e-3/2.
        speclen = allspecs.shape[-1]
        simdtype = self.simparams['dtype']
        out_data = sp.zeros((n_pulses, n_samps), dtype=simdtype)
        weights = {ibn:self.sensdict['ArrayFunc'](Az, El, ib[0], ib[1], sensdict['Angleoffset'])
                   for ibn, ib in enumerate(angles)}
        ntime = len(spectime)
        for istn in range(ntime):
            for ibn in range(n_beams):
                print('\t\t Making Beam {0:d} of {1:d}'.format(ibn, n_beams))
                weight = weights[ibn]
                for isamp in sp.arange(n_samps):

                    range_g = range_gates[isamp*ds_fac]
                    range_m = range_g*1e3
                    rnglims = [range_g-rng_len/2.0, range_g+rng_len/2.0]
                    rangelog = (rho >= rnglims[0])&(rho < rnglims[1])
                    cur_pnts = samp_num+isamp
                    cur_pnts = cur_pnts[cur_pnts < n_samps]
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
                    pow_num = sensdict['Pt']*sensdict['Ksys'][ibn]*t_s
                    pow_den = range_m**2
                    curdataloc = sp.where(sp.logical_and((pulse2spec == istn),
                                                         (beamcodes == ibn)))[0]
                    # create data
                    if not sp.any(curdataloc):
                        outstr = '\t\t No data for {0:d} of {1:d} in this time period'
                        print(outstr.format(ibn, n_beams))
                        continue
                    cur_pulse_data = MakePulseDataRepLPC(pulse, cur_spec, nlpc,
                                                         len(curdataloc), numtype=simdtype)
                    cur_pulse_data = cur_pulse_data*sp.sqrt(pow_num/pow_den)

                    for idatn, idat in enumerate(curdataloc):
                        out_data[idat, cur_pnts] = cur_pulse_data[idatn,:len(cur_pnts)]+out_data[idat, cur_pnts]

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
        timevec = self.simparams['Timevec']+self.timeoffset
        inttime = self.simparams['Tint']
        ds_fac = sp.prod(self.simparams['declist'])
        f_s = float(self.simparams['fsnum'])/self.simparams['fsden']/ds_fac
        # get cal info
        cal_temp = self.sensdict['CalDiodeTemp']
        calpwr = v_Boltz*cal_temp*f_s
        # Get array sizes
        samp_range = self.simparams['datasamples']
        d_samps = (samp_range[1]-samp_range[0])/ds_fac
        noise_range = self.simparams['noisesamples']
        n_samps = (noise_range[1]-noise_range[0])/ds_fac
        cal_range = self.simparams['calsamples']
        c_samps = (cal_range[1]-cal_range[0])/ds_fac

        range_gates = self.simparams['Rangegates']
        N_rg = len(range_gates)# take the size
        pulse = self.simparams['Pulse'][::ds_fac]
        pulselen = len(pulse)
        simdtype = self.simparams['dtype']
        Ntime = len(timevec)

        lagtype = self.simparams['lagtype']
        if 'outangles' in self.simparams.keys():
            n_beams = len(self.simparams['outangles'])
            inttime = inttime
        else:
            n_beams = len(self.simparams['angles'])

        # Choose type of processing
        if self.simparams['Pulsetype'].lower() == 'barker':
            lagfunc = BarkerLag
            Nlag = 1
        else:
            lagfunc = CenteredLagProduct
            Nlag = pulselen
        # initialize output arrays
        outdata = sp.zeros((Ntime, n_beams, d_samps-Nlag+1, Nlag), dtype=simdtype)
        outaddednoise = sp.zeros((Ntime, n_beams, d_samps-Nlag+1, Nlag), dtype=simdtype)
        outnoise = sp.zeros((Ntime, n_beams, n_samps-Nlag+1, Nlag), dtype=simdtype)
        outcal = sp.zeros((Ntime, n_beams, c_samps-Nlag+1, Nlag), dtype=simdtype)
        pulses = sp.zeros((Ntime, n_beams))
        pulsesN = sp.zeros((Ntime, n_beams))
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
        if 'NoiseTime'in infodict.keys():
            sridata = True
            tnoiselist = infodict['NoiseTime']
            nfile_loclist = [ifn*sp.ones(len(ifl)) for ifn, ifl in enumerate(tnoiselist)]
        else:
            sridata = False

        pulsen = sp.hstack(pulsen_list).astype(int)# pulse number
        beamn = sp.hstack(beamn_list).astype(int)# beam numbers
        ptimevec = sp.hstack(time_list).astype(float)# time of each pulse
        file_loc = sp.hstack(file_loclist).astype(int)# location in the file

        # run the time loop
        print("Forming ACF estimates")

        # For each time go through and read only the necisary files
        for itn, iti in enumerate(timevec):
            update_progress(float(itn)/Ntime, "Time {0:d} of {1:d}".format(itn, Ntime))
            # do the book keeping to determine locations of data within the files
            cur_tlim = (iti, iti+inttime)
            curcases = sp.logical_and(ptimevec >= cur_tlim[0], ptimevec < cur_tlim[1])

            if  not sp.any(curcases):
                progstr = "No pulses for time {0:d} of {1:d}, lagdata adjusted accordinly"
                update_progress(float(itn)/Ntime, progstr.format(itn, Ntime))
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
                pdb.set_trace()
            # Find the needed files and beam numbers
            curfiles = set(curfileloc)
            beamlocs = beamn[pos_all]
            timemat[itn, 0] = ptimevec[pos_all].min()
            timemat[itn, 1] = ptimevec[pos_all].max()
            # cur data pulls out all data from all of the beams and posisions
            curdata = sp.zeros((len(pos_all), d_samps), dtype=simdtype)
            curaddednoise = sp.zeros((len(pos_all), d_samps), dtype=simdtype)
            curnoise = sp.zeros((len(pos_all), n_samps), dtype=simdtype)
            curcal = sp.zeros((len(pos_all), c_samps), dtype=simdtype)
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
                curcal[file_arlocs] = curh5data['CalData'].astype(simdtype)[curfileitvec]
            # differentiate between phased arrays and dish antennas
            if self.sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:
                # After data is read in form lags for each beam
                for ibeam in range(n_beams):
                    progbeamstr = "Beam {0:d} of {1:d}".format(ibeam, n_beams)
                    update_progress(float(itn)/Ntime + float(ibeam)/Ntime/n_beams, progbeamstr)
                    beamlocstmp = sp.where(beamlocs == ibeam)[0]
                    pulses[itn, ibeam] = len(beamlocstmp)

                    outdata[itn, ibeam] = lagfunc(curdata[beamlocstmp].copy(),
                                                  numtype=simdtype, pulse=pulse,
                                                  lagtype=lagtype)

                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outnoise[itn, ibeam] = lagfunc(curnoise[beamlocstmp].copy(),
                                                   numtype=simdtype, pulse=pulse,
                                                   lagtype=lagtype)
                    outcal[itn, ibeam] = lagfunc(curcal[beamlocstmp].copy(),
                                                 numtype=simdtype, pulse=pulse,
                                                 lagtype=lagtype)
                    outaddednoise[itn, ibeam] = lagfunc(curaddednoise[beamlocstmp].copy(),
                                                        numtype=simdtype, pulse=pulse,
                                                        lagtype=lagtype)

            else:
                for ibeam, ibeamlist in enumerate(self.simparams['outangles']):
                    progbeamstr = "Beam {0:d} of {1:d}".format(ibeam, n_beams)
                    update_progress(float(itn)/Ntime + float(ibeam)/Ntime/n_beams, progbeamstr)
                    beamlocstmp = sp.where(sp.in1d(beamlocs, ibeamlist))[0]
                    curbeams = beamlocs[beamlocstmp]
                    ksysmat = Ksysvec[curbeams]
                    ksysmean = Ksysvec[ibeamlist[0]]
                    inputdata = curdata[beamlocstmp].copy()
                    noisedata = curnoise[beamlocstmp].copy()
                    noisedataadd = curaddednoise[beamlocstmp].copy()
                    caldata = curcal[beamlocstmp].copy()
                    ksysmult = ksysmean/sp.tile(ksysmat[:, sp.newaxis], (1, inputdata.shape[1]))
                    ksysmultn = ksysmean/sp.tile(ksysmat[:, sp.newaxis], (1, noisedata.shape[1]))
                    ksysmultna = ksysmean/sp.tile(ksysmat[:, sp.newaxis], (1, noisedataadd.shape[1]))
                    ksysmultc = ksysmean/sp.tile(ksysmat[:, sp.newaxis], (1, caldata.shape[1]))
                    pulses[itn, ibeam] = len(beamlocstmp)
                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outdata[itn, ibeam] = lagfunc(inputdata *ksysmult,
                                                  numtype=simdtype, pulse=pulse,
                                                  lagtype=lagtype)
                    outnoise[itn, ibeam] = lagfunc(noisedata*ksysmultn,
                                                   numtype=simdtype, pulse=pulse,
                                                   lagtype=lagtype)
                    outcal[itn, ibeam] = lagfunc(caldata*ksysmultc, numtype=simdtype,
                                                 pulse=pulse, lagtype=lagtype)

                    outaddednoise[itn, ibeam] = lagfunc(noisedataadd*ksysmultna,
                                                        numtype=simdtype, pulse=pulse,
                                                        lagtype=lagtype)

        # Create output dictionaries and output data
        data_lags = {'ACF':outdata, 'Pow':outdata[:, :, :, 0].real, 'Pulses':pulses,
                     'Time':timemat, 'AddedNoiseACF':outaddednoise}
        noise_lags = {'ACF':outnoise, 'Pow':outnoise[:, :, :, 0].real, 'Pulses':pulsesN,
                      'Time':timemat}
        return(data_lags, noise_lags)

#%% Make Lag dict to an iono container
def lagdict2ionocont(DataLags,NoiseLags,sensdict,simparams,time_vec):
    """This function will take the data and noise lags and create an instance of the
    Ionocontanier class. This function will also apply the summation rule to the lags.
    Inputs
    DataLags - A dictionary """
    # Pull in Location Data
    angles = simparams['angles']
    ang_data = sp.array([[iout[0], iout[1]] for iout in angles])
    rng_vec = simparams['Rangegates']
    # pull in other data
    t_s = float(simparams['fsden'])/simparams['fsnum']
    ds_fac = sp.prod(simparams['declist'])
    pulsewidth = len(simparams['Pulse'])*t_s
    txpower = sensdict['Pt']
    if sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:
        Ksysvec = sensdict['Ksys']
    else:

        beamlistlist = sp.array(simparams['outangles']).astype(int)
        inplist = sp.array([i[0] for i in beamlistlist])
        Ksysvec = sensdict['Ksys'][inplist]
        ang_data_temp = ang_data.copy()
        ang_data = sp.array([ang_data_temp[i].mean(axis=0)  for i in beamlistlist ])

    sumrule = simparams['SUMRULE']
    rng_vec2 = simparams['Rangegatesfinal']
    minrg = -sumrule[0].min()
    maxrg = len(rng_vec)-sumrule[1].max()
    Nrng2 = len(rng_vec2)

    # Copy the lags
    lagsData = DataLags['ACF'].copy()
    # Set up the constants for the lags so they are now
    # in terms of density fluxtuations.
    angtile = sp.tile(ang_data, (Nrng2, 1))
    rng_rep = sp.repeat(rng_vec2, ang_data.shape[0], axis=0)
    coordlist = sp.zeros((len(rng_rep), 3))
    [coordlist[:, 0], coordlist[:, 1:]] = [rng_rep, angtile]
    (Nt, n_beams, Nrng, Nlags) = lagsData.shape
    rng3d = sp.tile(rng_vec[sp.newaxis, sp.newaxis, :, sp.newaxis], (Nt, n_beams, 1, Nlags))*1e3
    ksys3d = sp.tile(Ksysvec[sp.newaxis, :, sp.newaxis, sp.newaxis], (Nt, 1, Nrng, Nlags))
    radar2acfmult = rng3d*rng3d/(pulsewidth*txpower*ksys3d)
    pulses = sp.tile(DataLags['Pulses'][:, :, sp.newaxis, sp.newaxis], (1, 1, Nrng, Nlags))
    time_vec = time_vec[:Nt]
    # Divid lags by number of pulses
    lagsData = lagsData/pulses
    # Set up the noise lags and divid out the noise.
    lagsNoise = NoiseLags['ACF'].copy()
    lagsNoise = sp.mean(lagsNoise,axis=2)
    pulsesnoise = sp.tile(NoiseLags['Pulses'][:, :, sp.newaxis],(1, 1, Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = sp.tile(lagsNoise[:, :, sp.newaxis, :],(1, 1, Nrng, 1))



    # subtract out noise lags
    lagsData = lagsData-lagsNoise
    # Calculate a variance using equation 2 from Hysell's 2008 paper. Done use full covariance matrix because assuming nearly diagonal.

    # multiply the data and the sigma by inverse of the scaling from the radar
    lagsData = lagsData*radar2acfmult
    lagsNoise = lagsNoise*radar2acfmult

    # Apply summation rule
    # lags transposed from (time,beams,range,lag)to (range,lag,time,beams)
    lagsData = sp.transpose(lagsData, axes=(2, 3, 0, 1))
    lagsNoise = sp.transpose(lagsNoise, axes=(2, 3, 0, 1))
    lagsDatasum = sp.zeros((Nrng2, Nlags, Nt, n_beams), dtype=lagsData.dtype)
    lagsNoisesum = sp.zeros((Nrng2 ,Nlags, Nt, n_beams), dtype=lagsNoise.dtype)
    for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
        for ilag in range(Nlags):
            lagsDatasum[irngnew,ilag] = lagsData[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].sum(axis=0)
            lagsNoisesum[irngnew,ilag] = lagsNoise[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].sum(axis=0)
    # Put everything in a parameter list
    Paramdata = sp.zeros((n_beams*Nrng2,Nt,Nlags),dtype=lagsData.dtype)
    # Put everything in a parameter list
    # transpose from (range,lag,time,beams) to (beams,range,time,lag)
    lagsDatasum = sp.transpose(lagsDatasum,axes=(3,0,2,1))
    lagsNoisesum = sp.transpose(lagsNoisesum,axes=(3,0,2,1))
    # Get the covariance matrix
    pulses_s=sp.transpose(pulses,axes=(1,2,0,3))[:,:Nrng2]
    Cttout=makeCovmat(lagsDatasum,lagsNoisesum,pulses_s,Nlags)

    Paramdatasig = sp.zeros((n_beams*Nrng2,Nt,Nlags,Nlags),dtype=Cttout.dtype)

    curloc = 0
    for irng in range(Nrng2):
        for ibeam in range(n_beams):
            Paramdata[curloc] = lagsDatasum[ibeam, irng].copy()
            Paramdatasig[curloc] = Cttout[ibeam, irng].copy()
            curloc += 1
    ionodata = IonoContainer(coordlist, Paramdata, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags)*t_s*ds_fac)
    ionosigs = IonoContainer(coordlist, Paramdatasig, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags*Nlags).reshape(Nlags,Nlags)*t_s*ds_fac)
    return (ionodata,ionosigs)

def makeCovmat(lagsDatasum,lagsNoisesum,pulses_s,Nlags):
    """
        Makes the covariance matrix for the lags given the noise acf and number
        of pulses.
    """
    axvec = sp.roll(sp.arange(lagsDatasum.ndim), 1)
    # Get the covariance matrix
    R = sp.transpose(lagsDatasum/sp.sqrt(2.*pulses_s), axes=axvec)
    Rw = sp.transpose(lagsNoisesum/sp.sqrt(2.*pulses_s), axes=axvec)
    l = sp.arange(Nlags)
    T1, T2 = sp.meshgrid(l,l)
    R0 = R[sp.zeros_like(T1)]
    Rw0 = Rw[sp.zeros_like(T1)]
    Td = sp.absolute(T1-T2)
    Tl = T1>T2
    R12 = R[Td]
    R12[Tl] = sp.conjugate(R12[Tl])
    Rw12 = Rw[Td]
    Rw12[Tl] = sp.conjugate(Rw12[Tl])
    Ctt = R0*R12+R[T1]*sp.conjugate(R[T2])+Rw0*Rw12+Rw[T1]*sp.conjugate(Rw[T2])
    avecback = sp.roll(sp.arange(Ctt.ndim),-2)
    Cttout = sp.transpose(Ctt,avecback)
    return Cttout
