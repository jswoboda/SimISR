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


    def __init__(self, config, outdir, outfilelist=None):
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
        (self.sensdict, self.simparams) = readconfigfile(config)


        self.datadir = Path(outdir)
        self.maindir = self.datadir.parent
        self.procdir = self.maindir/'ACF'
        self.drfdir = self.maindir/'drfdata'
        # HACK Need to clean up for mulitple channels
        self.drfdir_rf = self.drfdir.joinpath('rf_data')
        self.drfdir_md = self.drfdir.joinpath('metadata')
        #Make data
        self.outfilelist = []
        self.timeoffset = 0
        if not outfilelist is None:
            self.premadesetup(outfilelist)

    def premadesetup(self, outfilelist):
        """
            If data has been previously made set it up accordinly
        """
        infodict = h52dict(str(self.datadir.joinpath('INFO.h5')))
        alltime = sp.hstack(infodict['Time'])
        self.timeoffset = alltime.min()
        self.outfilelist = outfilelist

    def makerfdata(self, ionodict):
        """
            This method will take an ionocontainer object and create the associated
            RF data assoicated with the parameters from it.
        """

        filetimes = ionodict.keys()
        filetimes.sort()
        ftimes = sp.array(filetimes)
        N_angles = len(self.simparams['angles'])
        Npall = sp.floor(self.simparams['TimeLim']/self.simparams['IPP'])
        Npall = int(sp.floor(Npall/N_angles)*N_angles)
        Np = Npall/N_angles

        sweepids = self.simparams['sweepids']
        sweepnums = self.simparams['sweepnums']
        nseq = len(sweepids)
        usweep = sp.unique(sweepids)
        ippsamps = self.simparams['IPPsamps']
        d_samps = self.simparams['datasamples']
        n_samps = self.simparams['noisesamples']
        c_samps = self.simparams['calsamples']
        n_ns = n_samps[1] - n_samps[0]
        n_cal = c_samps[1] - c_samps[0]
        t_dict = self.simparams['Timing_Dict']
        ## HACK: Set max number of pulses per write based off of ippsamps
        max_write = 2*2**30 # 2 gigabye Write
        bps = 16 # byte per sample for complex128 in numpy array
        nippw = max_write/bps/ippsamps # number of ipps per write

        dec_list = self.simparams['declist']
        ds_fac = sp.prod(dec_list)

        simdtype = self.simparams['dtype']
        pulsetimes = sp.arange(Npall)*self.simparams['IPP'] +ftimes.min()
        pulsefile = sp.array([sp.where(itimes-ftimes >= 0)[0][-1] for itimes in pulsetimes])
        pulse_full = self.simparams['Pulse']
        print('\nData Now being created.')
        filetimes = ionodict.keys()
        filetimes.sort()
        ftimes = sp.array(filetimes)
        #digital rf stuff

        sample_rate_numerator = self.simparams['fsnum']
        sample_rate_denominator = self.simparams['fsden']
        sample_rate = sp.double(sample_rate_numerator) / sample_rate_denominator
        start_global_index = int(sample_rate*filetimes[0])
        dtype_str = 'complex64'  # complex64
        sub_cadence_secs = 3600  # Number of seconds of data in a subdirectory
        file_cadence_millisecs = 10000  # Each file will have up to 400 ms of data
        compression_level = 0  # no compression
        checksum = False  # no checksum
        is_complex = True  # complex values
        is_continuous = True
        num_subchannels = 1  # only one subchannel
        marching_periods = False  # no marching periods when writing
        uuid = "SimISRDRFZenith"
        zndatadir = self.drfdir_rf.joinpath('zenith-l')
        data_object = drf.DigitalRFWriter(str(zndatadir), dtype_str, sub_cadence_secs,
                                          file_cadence_millisecs, start_global_index,
                                          sample_rate_numerator, sample_rate_denominator,
                                          uuid, compression_level, checksum,
                                          is_complex, num_subchannels,
                                          is_continuous, marching_periods)
        drfdirtx = self.drfdir_rf.joinpath('tx-h')
        data_object_tx = drf.DigitalRFWriter(str(drfdirtx), dtype_str, sub_cadence_secs,
                                             file_cadence_millisecs, start_global_index,
                                             sample_rate_numerator, sample_rate_denominator,
                                             uuid, compression_level, checksum,
                                             is_complex, num_subchannels,
                                             is_continuous, marching_periods)
        # Noise Scaling
        noisepwr = v_Boltz*self.sensdict['Tsys']*sample_rate
        calpwr = v_Boltz*self.sensdict['CalDiodeTemp']*sample_rate
        # digital metadata
        #TODO temp for original data
        # antenna control

        dmddir = self.datadir.parent.joinpath('drfdata', 'metadata')
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

        idmobj = drf.DigitalMetadataWriter(str(iddir), 3600, 1,
                                           sample_rate_numerator, sample_rate_denominator,
                                           'md')
        # power mete metadata
        pmdir = dmddir.joinpath('powermeter')
        pmmdict = {'zenith_power': self.sensdict['Pt'], 'misa_power': 3110.3}
        pmmobj = drf.DigitalMetadataWriter(str(pmdir), 3600, 5, 1, 1, 'power')
        # differentiate between phased arrays and dish antennas
        if self.sensdict['Name'].lower() in ['risr', 'pfisr', 'risr-n']:

            beams = sp.tile(sp.arange(N_angles), Npall/N_angles)
        else:

            # for dish arrays
            brate = self.simparams['beamrate']
            beams2 = sp.repeat(sp.arange(N_angles), brate)
            beam3 = sp.concatenate((beams2, beams2[::-1]))
            ntile = int(sp.ceil(Npall/len(beam3)))
            leftover = int(Npall-ntile*len(beam3))
            if ntile > 0:
                beams = sp.tile(beam3, ntile)
                beams = sp.concatenate((beams, beam3[:leftover]))
            else:
                beams = beam3[:leftover]
        pulsen = sp.repeat(sp.arange(Np), N_angles).astype(int)
        pt_list = []
        pb_list = []
        pn_list = []
        fname_list = []
        si_list = []
        sn_list = []

        Nf = len(filetimes)



        for ifn, ifilet in enumerate(filetimes):

            outdict = {}
            ifile = ionodict[ifilet]
            curcontainer = IonoContainer.readh5(ifile)
            if ifn == 0:
                self.timeoffset = curcontainer.Time_Vector[0, 0]
            f_pulses = sp.where(pulsefile == ifn)[0]
            nwrites = sp.ceil(float(len(f_pulses))/nippw).astype(int)
            for ifwrite, i_pstart in enumerate(range(0, len(f_pulses), nippw)):
                progstr1 = 'Data for write {:d} of {:d} being created.'

                prog_level = float(ifn)/Nf+float(ifwrite)/Nf/nwrites
                update_progress(prog_level, progstr1.format(int(ifn*nwrites+ifwrite+1), int(Nf*nwrites)))
                i_pend = sp.minimum(i_pstart+nippw, len(f_pulses))
                cur_fpulses = f_pulses[i_pstart:i_pend]
                pt = pulsetimes[cur_fpulses]
                pb = beams[cur_fpulses]
                pn = pulsen[cur_fpulses]
                pulse_idx = pn%nseq
                s_i = sweepids[pulse_idx]
                s_n = sweepnums[pulse_idx]

                rawdata = self.__makeTime__(pt, curcontainer.Time_Vector,
                                            curcontainer.Sphere_Coords,
                                            curcontainer.Param_List, pb, pulse_idx)

                n_pulse_cur, n_raw = rawdata.shape
                rawdata_us = sp.signal.resample(rawdata, n_raw*ds_fac, axis=1)
                r_samps = sp.diff(d_samps)[0]
                rawdata_us = rawdata_us[:,:r_samps]
                alldata = sp.random.randn(n_pulse_cur, ippsamps) + 1j*sp.random.randn(n_pulse_cur, ippsamps)
                alldata = sp.sqrt(noisepwr/2.)*alldata
                caldata = sp.random.randn(n_pulse_cur, n_cal) + 1j*sp.random.randn(n_pulse_cur, n_cal)
                caldata = sp.sqrt(calpwr/2)*caldata.astype(simdtype)
                noisedata = alldata[:, d_samps[0]:d_samps[1]]
                for i_swid in usweep:
                    sw_ind = sp.where(s_i == i_swid)[0]
                    cur_tdict = t_dict[i_swid]
                    cur_c = cur_tdict['calibration']
                    cur_csamps = sp.diff(cur_c)[0]
                    alldata[sw_ind, d_samps[0]:d_samps[1]] += rawdata_us[sw_ind]
                    alldata[sw_ind, cur_c[0]:cur_c[1]] += caldata[sw_ind, :cur_csamps]

                alldata = alldata/sp.sqrt(calpwr/2.)
                rawdata = rawdata_us.copy()/sp.sqrt(calpwr/2.)
                noisedata = noisedata.copy()/sp.sqrt(calpwr/2.)
                cal_samps = sp.zeros_like(caldata)
                data_samps = sp.zeros_like(rawdata_us)
                noise_samps = sp.zeros((n_pulse_cur, n_ns), dtype=alldata.dtype)

                for i_swid in usweep:
                    sw_ind = sp.where(s_i == i_swid)[0]
                    cur_tdict = t_dict[i_swid]
                    cur_c = cur_tdict['calibration']
                    cur_csamps = sp.diff(cur_c)[0]
                    cur_n = cur_tdict['noise']
                    cur_nsamps = sp.diff(cur_n)[0]
                    data_samps[sw_ind] = alldata[sw_ind, d_samps[0]:d_samps[1]]
                    cal_samps[sw_ind, :cur_csamps] = alldata[sw_ind, cur_c[0]:cur_c[1]]
                    noise_samps[sw_ind, :cur_nsamps] = alldata[sw_ind, cur_n[0]:cur_n[1]]

                data_samps = sp.signal.resample(data_samps, n_raw, axis=1)
                noise_samps_ds = sp.signal.resample(noise_samps, n_ns/ds_fac, axis=1)
                # Down sample data using resample, keeps variance correct
                rawdata_ds = sp.signal.resample(rawdata_us, rawdata.shape[1]/ds_fac, axis=1)
                noisedata_ds = sp.signal.resample(noisedata, noisedata.shape[1]/ds_fac, axis=1)

                noise_est = sp.mean(sp.mean(noise_samps.real**2+noise_samps.imag**2))
                cal_est = sp.mean(sp.mean(cal_samps.real**2+cal_samps.imag**2))
                calfac = calpwr/(cal_est-noise_est)
                outdict['AddedNoise'] = noisedata_ds
                outdict['RawData'] = data_samps
                outdict['RawDatanonoise'] = rawdata_ds
                outdict['NoiseData'] = noise_samps_ds
                outdict['CalFactor'] = sp.array([calfac])
                outdict['Pulses'] = pn
                outdict['Beams'] = pb
                outdict['Time'] = pt
                fname = '{0:d} RawData.h5'.format(ifwrite)
                newfn = self.datadir/fname
                self.outfilelist.append(str(newfn))
                dict2h5(str(newfn), outdict)
                #Listing info
                pt_list.append(pt)
                pb_list.append(pb)
                pn_list.append(pn)
                si_list.append(s_i)
                sn_list.append(s_n)
                fname_list.append(fname)

                #HACK Add a constant to set the noise level to be atleast 1 bit
                num_const = 200.
                #transmitdata
                tx_data = sp.zeros_like(alldata).astype('complex64')
                tx_data[:, :pulse_full.shape[-1]] = pulse_full[pulse_idx].astype('complex64')
                data_object_tx.rf_write(tx_data.flatten()*num_const)
                # extend array for digital rf to flattend array

                data_object.rf_write(alldata.flatten().astype('complex64')*num_const)
                id_strt = int(idmobj.get_samples_per_second()*pt[0])
                dmdplist = sp.arange(n_pulse_cur, dtype=int)*ippsamps + id_strt
                acmobj.write(int(acmobj.get_samples_per_second()*pt[0]), acmdict)
                idmdict = {'sweepid': s_i, 'sample_rate': sp.array([sample_rate]*len(s_n)),
                           'modeid': sp.array([100000001]*len(s_n)), 'sweepnum': s_n}
                idmobj.write(dmdplist, idmdict)
                pmmobj.write(int(pmmobj.get_samples_per_second()*pt[0]), pmmdict)

        infodict = {'Files':fname_list, 'Time':pt_list, 'Beams':pb_list, 'Pulses':pn_list}
        dict2h5(str(self.datadir.joinpath('INFO.h5')), infodict)
        data_object.close()
        data_object_tx.close()



#%% Make functions
    def __makeTime__(self, pulsetimes, spectime, Sphere_Coords, allspecs, beamcodes, pulse_idx):
        """
            This will make raw radar data for a given time period and set of
            spectrums. This is an internal function called by __init__.
            Args:
                self (:obj:'RadarDataFile'): RadarDataFile object.
                pulsetimes (:obj:'ndarray'): The time the pulses are sent out in reference to the spectrums.
                spectime (:obj:'ndarray'): The times for the spectrums.
                Sphere_Coords (:obj:'ndarray'): An Nlx3 array that holds the spherical coordinates of the spectrums.
                allspecs (:obj:'ndarray'): An NlxNdtimexNspec array that holds the spectrums.
                beamcodes (:obj:'ndarray'): The index of the beams that are used.
                pulse_idx (:obj:'ndarray'): The index of the pulses that will be used.
            Returns:
                outdata (:obj:'ndarray'): A numpy array with the shape of the rep1xlen(pulse)
        """
        range_gates = self.simparams['Rangegates']
        sensdict = self.sensdict
        samp_range = self.simparams['datasamples']
        ds_fac = sp.prod(self.simparams['declist'])
        f_s = float(self.simparams['fsnum'])/self.simparams['fsden']/ds_fac
        t_s = float(ds_fac*self.simparams['fsden'])/self.simparams['fsnum']
        pulse = self.simparams['Pulse'][:,::ds_fac]
        #HACK number of lpc points connected to ratio of sampling frequency and
        # notial ion-line spectra with a factor of 10.
        nlpc = int(10*f_s/20e3) + 1
        pulse2spec = sp.array([sp.where(itimes-spectime >= 0)[0][-1] for itimes in pulsetimes])
        n_pulses = len(pulse2spec)
        lp_pnts = pulse.shape[-1]
        samp_num = sp.arange(lp_pnts)
        n_samps = int(sp.ceil(float(samp_range[1]-samp_range[0])/ds_fac))
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
                #print('\t\t Making Beam {0:d} of {1:d}'.format(ibn, n_beams))
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
                    cur_pidx = pulse_idx[curdataloc]

                    # create data
                    if not sp.any(curdataloc):
                        # outstr = '\t\t No data for {0:d} of {1:d} in this time period'
                        # print(outstr.format(ibn, n_beams))
                        continue
                    cur_pulse_data = MakePulseDataRepLPC(pulse, cur_spec, nlpc,
                                                         cur_pidx, numtype=simdtype)
                    cur_pulse_data = cur_pulse_data*sp.sqrt(pow_num/pow_den)
                    out_data[curdataloc][:, cur_pnts] += cur_pulse_data[:, cur_pnts-isamp]
                    # for idatn, idat in enumerate(curdataloc):
                    #     out_data[idat, cur_pnts] += cur_pulse_data[idatn, cur_pnts-isamp]


        return out_data
        #%% Processing
    def processdataiono(self):
        """
            This will call the data processing function and place the resulting
            ACFs in an Ionocontainer so it can be saved out.

            Returns:
                Ionocontainer (:obj:'IonoContainer') This is an instance of the
                    ionocontainer class that will hold the acfs.
        """
        outdict = self.processdatadrf()
        alldata = {}

        dec_list =self.simparams['declist']
        ds_fac = sp.prod(dec_list)
        t_dict = self.simparams['Timing_Dict']
        t_s = float(self.simparams['fsden'])/self.simparams['fsnum']
        rng_samprate = t_s*v_C_0*1e-3/2.
        plen_ds = self.simparams['Pulse'].shape[-1]/ds_fac
        if 'LP' in outdict:
            datalagslp = outdict['LP']['datalags']
            noiselagslp = outdict['LP']['noiselags']
            d_len = t_dict[300]['signal']
            rng_gates = sp.arange(d_len[0], d_len[1])*rng_samprate

            rng_gates_ds = rng_gates[::ds_fac]
            self.simparams['Rangegates'] = rng_gates
            minrg = plen_ds-1
            maxrg = len(rng_gates_ds)-plen_ds+1
            self.simparams['Rangegatesfinal'] = rng_gates_ds[minrg:maxrg]
            (ionoout, ionosig) = lagdict2ionocont(datalagslp, noiselagslp, self.sensdict,
                                                  self.simparams, datalagslp['Time'])
            alldata['LP'] = [ionoout, ionosig]
        if 'AC' in outdict:
            datalagsac = outdict['AC']['datalags']
            noiselagsac = outdict['AC']['noiselags']

            d_len = t_dict[1]['signal']
            rng_gates = sp.arange(d_len[0], d_len[1])*rng_samprate

            rng_gates_ds = rng_gates[::ds_fac]
            self.simparams['Rangegates'] = rng_gates
            minrg = plen_ds-1
            maxrg = len(rng_gates_ds)-plen_ds+1
            self.simparams['Rangegatesfinal'] = rng_gates_ds[minrg:maxrg]
            ionoout, ionosig = lagdict2ionocont(datalagsac, noiselagsac, self.sensdict,
                                                self.simparams, datalagsac['Time'])
            alldata['AC'] = [ionoout, ionosig]
        return alldata

    def processdatadrf(self):
        """
            From the digital_rf channel data create ACFs and place them in a dictionary.

            Returns:
                mode_dict (:obj:'dict') This holds the processed ACFs before they're put into
                    the ionocontainer. The dictionary keys are the different modes that are
                    avalible. The sub dictionary is then data and noise lags.
        """

        outdir = self.datadir
        t_dict = self.simparams['Timing_Dict']
        rfdir = self.drfdir_rf

        simdtype = self.simparams['dtype']
        lagtype = self.simparams['lagtype']
        dmddir = outdir.parent.joinpath('drfdata', 'metadata')
        acmdir = dmddir.joinpath('antenna_control_metadata')
        iddir = dmddir.joinpath('id_metadata')

        dec_list = self.simparams['declist']
        ds_fac = sp.prod(dec_list)
        pulse_arr = self.simparams['Pulse'][:,::ds_fac]
        s_id_p1 = self.simparams['sweepids']
        pulse_dict = {i:j for i, j in zip(s_id_p1, pulse_arr)}
        idmobj = drf.DigitalMetadataReader(str(iddir))
        drfObj = drf.DigitalRFReader(str(rfdir))
        objprop = drfObj.get_properties('zenith-l')
        sps = objprop['samples_per_second']
        d_bnds = drfObj.get_bounds('zenith-l')

        time_list = self.simparams['Timevec']*sps + d_bnds[0]
        time_list = time_list.astype(int)
        calpwr = v_Boltz*self.sensdict['CalDiodeTemp']*sps/ds_fac
        pulse_dict = {i:j for i, j in zip(s_id_p1, pulse_arr)}


        acode_swid = sp.arange(1, 33)
        lp_swid = 300
        # Choose type of processing
        if self.simparams['Pulsetype'].lower() == 'barker':
            lagfunc = BarkerLag
            Nlag = 1
        else:
            lagfunc = CenteredLagProduct

        mode_dict = {'AC':{'swid':acode_swid.tolist()}, 'LP':{'swid':[lp_swid]}}
        # make data lags
        Ntime = len(time_list)
        n_beams = 1
        timemat = sp.column_stack((time_list/sps, time_list/sps+int(self.simparams['Tint'])))
        for i_type in mode_dict.keys():
            swid_1 = mode_dict[i_type]['swid'][0]
            t_info = t_dict[swid_1]
            d_samps = int(sp.diff(t_info['signal'])[0]/ds_fac)
            n_samps = int(sp.diff(t_info['noise'])[0]/ds_fac)
            Nlag = pulse_dict[swid_1].shape[0]
            outdata = sp.zeros((Ntime, n_beams, d_samps-Nlag+1, Nlag), dtype=simdtype)
            outnoise = sp.zeros((Ntime, n_beams, n_samps-Nlag+1, Nlag), dtype=simdtype)
            pulses = sp.zeros((Ntime, n_beams))
            outcal = sp.zeros((Ntime, n_beams))
            data_lags = {'ACF':outdata, 'Pulses':pulses,
                         'Time':timemat, "CalFactor":outcal}
            noise_lags = {'ACF':outnoise, 'Pulses':pulses,
                         'Time':timemat}
            mode_dict[i_type]['datalags'] = data_lags
            mode_dict[i_type]['noiselags'] = noise_lags

        for itn, itb in enumerate(time_list):
            ite = itb + int(self.simparams['Tint']*sps)
            id_dict = idmobj.read_flatdict(itb, ite)

            un_id, id_ind_list = sp.unique(id_dict['sweepid'], return_inverse=True)
            p_indx = id_dict['index']
            data_dict = {'AC':{'sig':[], 'cal':[], 'noise':[], 'NIPP':[]},
                         'LP':{'sig':None, 'cal':None, 'noise':None, 'NIPP':None}}
            for i_idn, i_id in enumerate(un_id):
                t_info = t_dict[i_id]
                ipp_samp = t_info['full'][1]
                sig_bnd = t_info['signal']
                noise_bnd = t_info['noise']
                cal_bnd = t_info['calibration']
                cur_pulse = pulse_dict[i_id]
                curlocs = sp.where(id_ind_list == i_idn)[0]
                sig_data = sp.zeros((len(curlocs), sp.diff(sig_bnd)[0]), dtype=sp.complex64)
                cal_data = sp.zeros((len(curlocs), sp.diff(cal_bnd)[0]), dtype=sp.complex64)
                n_data = sp.zeros((len(curlocs), sp.diff(noise_bnd)[0]), dtype=sp.complex64)
                # HACK Need to come up with clutter cancellation
                for ar_id, id_ind in enumerate(curlocs):
                    dr_ind = p_indx[id_ind]
                    raw_data = drfObj.read_vector(dr_ind, ipp_samp, 'zenith-l', 0)
                    sig_data[ar_id] = raw_data[sig_bnd[0]:sig_bnd[1]]
                    cal_data[ar_id] = raw_data[cal_bnd[0]:cal_bnd[1]]
                    n_data[ar_id] = raw_data[noise_bnd[0]:noise_bnd[1]]
                # Down sample data
                sig_data = sp.signal.resample(sig_data, sig_data.shape[1]/ds_fac, axis=1)
                cal_data = sp.signal.resample(cal_data, cal_data.shape[1]/ds_fac, axis=1)
                n_data = sp.signal.resample(n_data, n_data.shape[1]/ds_fac, axis=1)
                # make lag products
                sig_acf = lagfunc(sig_data, numtype=simdtype, pulse=cur_pulse,
                                  lagtype=lagtype)
                n_acf = lagfunc(n_data, numtype=simdtype, pulse=cur_pulse,
                                lagtype=lagtype)
                cal_acf = lagfunc(cal_data, numtype=simdtype, pulse=cur_pulse,
                                  lagtype=lagtype)
                cal_acf = sp.median(cal_acf, axis=0)[0].real

                if i_id in acode_swid:
                    data_dict['AC']['sig'].append(sig_acf)
                    data_dict['AC']['cal'].append(cal_acf)
                    data_dict['AC']['noise'].append(n_acf)
                    data_dict['AC']['NIPP'].append(len(curlocs))
                elif i_id == lp_swid:
                    data_dict['LP'] = {"sig":sig_acf, "cal":cal_acf,
                                       "noise":n_acf, 'NIPP':len(curlocs)}

            # AC stuff
            if data_dict['AC']['sig']:
                data_dict['AC'] = {ikey:sum(data_dict['AC'][ikey]) for ikey in data_dict['AC']}
                cal_est = data_dict['AC']['cal']/data_dict['AC']['NIPP']
                noise_est = sp.median(data_dict['AC']['noise'].real,axis=0)[0]/data_dict['AC']['NIPP']
                calfac = calpwr/(cal_est-noise_est)
                mode_dict['AC']['datalags']['ACF'][itn, 0] = data_dict['AC']['sig']
                mode_dict['AC']['datalags']['Pulses'][itn, 0] = data_dict['AC']['NIPP']
                mode_dict['AC']['datalags']['CalFactor'][itn, 0] = calfac

                mode_dict['AC']['noiselags']['ACF'][itn, 0] = data_dict['AC']['noise']
                mode_dict['AC']['noiselags']['Pulses'][itn, 0] = data_dict['AC']['NIPP']
            else:
                del mode_dict['AC']
            # LP stuff
            if data_dict['LP']['sig'] is None:
                del mode_dict['LP']
            else:
                cal_est = data_dict['LP']['cal']/data_dict['LP']['NIPP']
                noise_est = sp.median(data_dict['LP']['noise'].real,axis=0)[0]/data_dict['LP']['NIPP']
                calfac = calpwr/(cal_est-noise_est)
                mode_dict['LP']['datalags']['ACF'][itn, 0] = data_dict['LP']['sig']
                mode_dict['LP']['datalags']['Pulses'][itn, 0] = data_dict['LP']['NIPP']
                mode_dict['LP']['datalags']['CalFactor'][itn, 0] = calfac

                mode_dict['LP']['noiselags']['ACF'][itn, 0] = data_dict['LP']['noise']
                mode_dict['LP']['noiselags']['Pulses'][itn, 0] = data_dict['LP']['NIPP']

        return mode_dict

    def processdata(self):
        """
        This will perform the the data processing and create the ACF estimates
        for both the data and noise.
            Inputs:

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

        # Get array sizes
        samp_range = self.simparams['datasamples']
        d_samps = (samp_range[1]-samp_range[0])/ds_fac
        noise_range = self.simparams['noisesamples']
        n_samps = (noise_range[1]-noise_range[0])/ds_fac
        cal_range = self.simparams['calsamples']

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
        outcal = sp.zeros((Ntime, n_beams), dtype=simdtype)
        pulses = sp.zeros((Ntime, n_beams))
        pulsesN = sp.zeros((Ntime, n_beams))
        timemat = sp.zeros((Ntime, 2))
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
                ipdb.set_trace()
            # Find the needed files and beam numbers
            curfiles = set(curfileloc)
            beamlocs = beamn[pos_all]
            timemat[itn, 0] = ptimevec[pos_all].min()
            timemat[itn, 1] = ptimevec[pos_all].max()
            # cur data pulls out all data from all of the beams and posisions
            curdata = sp.zeros((len(pos_all), d_samps), dtype=simdtype)
            curaddednoise = sp.zeros((len(pos_all), d_samps), dtype=simdtype)
            curnoise = sp.zeros((len(pos_all), n_samps), dtype=simdtype)
            curcal = sp.zeros((len(pos_all)), dtype=simdtype)
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
                curcal[file_arlocs] = curh5data['CalFactor'].astype(simdtype)[0]
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

                    outaddednoise[itn, ibeam] = lagfunc(curaddednoise[beamlocstmp].copy(),
                                                        numtype=simdtype, pulse=pulse,
                                                        lagtype=lagtype)

                    outcal[itn, ibeam] = curcal[beamlocstmp]

            else:
                for ibeam, ibeamlist in enumerate(self.simparams['outangles']):
                    progbeamstr = "Beam {0:d} of {1:d}".format(ibeam, n_beams)
                    update_progress(float(itn)/Ntime + float(ibeam)/Ntime/n_beams, progbeamstr)
                    beamlocstmp = sp.where(sp.in1d(beamlocs, ibeamlist))[0]
                    inputdata = curdata[beamlocstmp].copy()
                    noisedata = curnoise[beamlocstmp].copy()
                    noisedataadd = curaddednoise[beamlocstmp].copy()

                    pulses[itn, ibeam] = len(beamlocstmp)
                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outdata[itn, ibeam] = lagfunc(inputdata,
                                                  numtype=simdtype, pulse=pulse,
                                                  lagtype=lagtype)
                    outnoise[itn, ibeam] = lagfunc(noisedata,
                                                   numtype=simdtype, pulse=pulse,
                                                   lagtype=lagtype)


                    outaddednoise[itn, ibeam] = lagfunc(noisedataadd,
                                                        numtype=simdtype, pulse=pulse,
                                                        lagtype=lagtype)
                    outcal[itn, ibeam] = curcal[beamlocstmp].mean()
        # Create output dictionaries and output data
        data_lags = {'ACF':outdata, 'Pow':outdata[:, :, :, 0].real, 'Pulses':pulses,
                     'Time':timemat, 'AddedNoiseACF':outaddednoise, "CalFactor":outcal}
        noise_lags = {'ACF':outnoise, 'Pow':outnoise[:, :, :, 0].real, 'Pulses':pulsesN,
                      'Time':timemat}
        return(data_lags, noise_lags)

#%% Make Lag dict to an iono container
def lagdict2ionocont(DataLags, NoiseLags, sensdict, simparams, time_vec):
    """This function will take the data and noise lags and create an instance of the
    Ionocontanier class. This function will also apply the summation rule to the lags.
    Inputs
    DataLags - A dictionary """
    # Pull in Location Data
    angles = simparams['angles']
    ang_data = sp.array([[iout[0], iout[1]] for iout in angles])

    # pull in other data
    t_s = float(simparams['fsden'])/simparams['fsnum']
    ds_fac = sp.prod(simparams['declist'])

    rng_vec = simparams['Rangegates'][::ds_fac]
    n_samps = len(rng_vec)
    p_samps = len(simparams['Pulse'][::ds_fac])
    pulsewidth = p_samps*t_s*ds_fac
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

    minrg = -sumrule[0].min()
    maxrg = Nrng2 + minrg

    # Copy the lags
    lagsData = DataLags['ACF'].copy()
    calfactor = DataLags['CalFactor'].copy().real
    # Set up the constants for the lags so they are now
    # in terms of density fluxtuations.
    angtile = sp.tile(ang_data, (Nrng2, 1))
    rng_rep = sp.repeat(rng_vec2, ang_data.shape[0], axis=0)
    coordlist = sp.zeros((len(rng_rep), 3))
    [coordlist[:, 0], coordlist[:, 1:]] = [rng_rep, angtile]
    (Nt, n_beams, Nrng, Nlags) = lagsData.shape
    plen2 = float(p_samps-1)/2
    samps = sp.arange(0, p_samps, dtype=int)
    rng_ave = sp.ones_like(rng_vec)
    rng_vectemp = rng_vec.copy()
    rng_vectemp[rng_vec <= 0] = 0
    for isamp in range(n_samps):
        cursamps = isamp-samps
        keepsamps = cursamps >= 0
        cursamps = cursamps[keepsamps]
        rng_samps = rng_vectemp[cursamps]
        keepsamps2 = rng_samps > 0
        if keepsamps2.sum() == 0:
            continue
        rng_samps = rng_samps[keepsamps2]
        rng_ave[isamp] = 1./sp.sqrt(sp.sum(1./sp.power(rng_samps*1e3, 2))/p_samps)
    rng_ave_temp = rng_ave.copy()
    rng_ave = rng_ave[int(sp.floor(plen2)):-int(sp.ceil(plen2))]
    rng_ave = rng_ave[minrg:maxrg]


    pulses = sp.tile(DataLags['Pulses'][:, :, sp.newaxis, sp.newaxis], (1, 1, Nrng, Nlags))
    time_vec = time_vec[:Nt]
    # Divid lags by number of pulses
    lagsData = lagsData/pulses
    # Set up the noise lags and divid out the noise.
    lagsNoise = NoiseLags['ACF'].copy()
    lagsNoise = sp.mean(lagsNoise, axis=2)
    pulsesnoise = sp.tile(NoiseLags['Pulses'][:, :, sp.newaxis], (1, 1, Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = sp.tile(lagsNoise[:, :, sp.newaxis, :],(1, 1, Nrng, 1))

    # Set Up arrays for range adjustment
    rng3d = sp.tile(rng_ave[:, sp.newaxis, sp.newaxis, sp.newaxis], (1, Nlags, Nt, n_beams))
    ksys3d = sp.tile(Ksysvec[sp.newaxis, sp.newaxis, sp.newaxis, :], (Nrng2, Nlags, Nt, 1))
    cal_fac = sp.tile(calfactor[sp.newaxis, sp.newaxis, :, :], (Nrng2, Nlags, 1, 1))
    radar2acfmult = cal_fac*rng3d*rng3d/(pulsewidth*txpower*ksys3d)

    # subtract out noise lags
    lagsData = lagsData-lagsNoise
    # Calculate a variance using equation 2 from Hysell's 2008 paper. Done use full covariance matrix because assuming nearly diagonal.

    # multiply the data and the sigma by inverse of the scaling from the radar
    # lagsData = lagsData*radar2acfmult
    # lagsNoise = lagsNoise*radar2acfmult

    # Apply summation rule
    # lags transposed from (time,beams,range,lag)to (range,lag,time,beams)
    lagsData = sp.transpose(lagsData, axes=(2, 3, 0, 1))
    lagsNoise = sp.transpose(lagsNoise, axes=(2, 3, 0, 1))
    lagsDatasum = sp.zeros((Nrng2, Nlags, Nt, n_beams), dtype=lagsData.dtype)
    lagsNoisesum = sp.zeros((Nrng2 ,Nlags, Nt, n_beams), dtype=lagsNoise.dtype)


    for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
        for ilag in range(Nlags):
            lagsDatasum[irngnew, ilag] = lagsData[irng+sumrule[0, ilag]:irng+sumrule[1, ilag]+1, ilag].sum(axis=0)
            lagsNoisesum[irngnew, ilag] = lagsNoise[irng+sumrule[0, ilag]:irng+sumrule[1, ilag]+1, ilag].sum(axis=0)
    # Put everything in a parameter list
    Paramdata = sp.zeros((n_beams*Nrng2, Nt, Nlags), dtype=lagsData.dtype)
    lagsDatasum = lagsDatasum*radar2acfmult
    lagsNoisesum = lagsNoisesum*radar2acfmult
    # Put everything in a parameter list
    # transpose from (range,lag,time,beams) to (beams,range,time,lag)
    lagsDatasum = sp.transpose(lagsDatasum, axes=(3, 0, 2, 1))
    lagsNoisesum = sp.transpose(lagsNoisesum, axes=(3, 0, 2, 1))
    # Get the covariance matrix
    pulses_s = sp.transpose(pulses, axes=(1, 2, 0, 3))[:, :Nrng2]
    Cttout = makeCovmat(lagsDatasum, lagsNoisesum, pulses_s, Nlags)

    Paramdatasig = sp.zeros((n_beams*Nrng2, Nt, Nlags, Nlags), dtype=Cttout.dtype)

    curloc = 0
    for irng in range(Nrng2):
        for ibeam in range(n_beams):
            Paramdata[curloc] = lagsDatasum[ibeam, irng].copy()
            Paramdatasig[curloc] = Cttout[ibeam, irng].copy()
            curloc += 1
    ionodata = IonoContainer(coordlist, Paramdata, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags)*t_s*ds_fac)
    ionosigs = IonoContainer(coordlist, Paramdatasig, times=time_vec, ver=1,
                             paramnames=sp.arange(Nlags*Nlags).reshape(Nlags, Nlags)*t_s*ds_fac)
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
