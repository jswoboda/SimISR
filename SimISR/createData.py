#!/usr/bin/env python
"""

radarData.py
This file holds the RadarData class that hold the radar data and processes it.

@author: John Swoboda

"""

from pathlib import Path
from fractions import Fraction
import numpy as np
import scipy.signal as sig
import scipy.constants as sc
import pandas as pd
import pytz

elec_radius = sc.e**2.0 * sc.mu_0 / (4.0 * sc.pi * sc.m_e)
# My modules
from .utilFunctions import MakePulseDataRepLPC, update_progress
import digital_rf as drf
import xarray as xr
from .CoordTransforms import cartisian2Sphereical, wgs2ecef, ecef2enul

from mitarspysigproc import (
    pfb_decompose,
    pfb_reconstruct,
    kaiser_coeffs,
    kaiser_syn_coeffs,
    npr_analysis,
    npr_synthesis,
    rref_coef,
)


class RadarDataCreate(object):
    """Runs the overall process for SimISR for the ion line.

    Attributes
    ----------
    experiment : Experiment
        Experiment settings.
    save_directory : str
        Directory where the digital RF data will be saved.
    save_path : Path
        save_directory but as a Path object.
    first_time : bool
        Is this the first write.

    """
    def __init__(self, experiment, save_directory=None):
        self.experiment = experiment

        if save_directory is None:
            self.save_directory = experiment.save_directory
        else:
            self.save_directory = save_directory
        self.save_path = Path(self.save_directory)
        self.first_time = True
        self.tx_beams = {}
        self.rx_beams = {}

    def spatial_set_up(self, coordinates, origlla):
        """Performs all of the coordinate transforms and weighting from the sensor and each beam pattern.

        Parameters
        ----------
        coordinates : DatasetCoordinates
            Dataset coordinates from xarray. Coordinates must be named x,y,z and be in the ENU configuration
        origlla : ndarray
            wgs coordinates of the origin point for the coordinates. Should be part of the attr object in the Dataset.

        Returns
        -------
        phys_xr : Dataset
            Xarray Dataset, uses input coordinates for spatial axis and adds a pairs coordinate. Holds range, and spatial losses as variables.

        """
        origecef = wgs2ecef(origlla[:, np.newaxis])

        # Index each spatial set up by name of tx and rx of radars and beam code. This leads to a spatial code number
        spatial_code = 0
        sp_setup = []
        ksys_list = []
        bi_rng_list = []
        sp_loss_list = []
        pair_num = 0
        pair_dict = {}
        beam_ls_list = []
        dims = ["locs", "pairs"]
        # each sequence will have it's own spatial setup.
        rdr_objs = self.experiment.radarobjs
        # Go through each sequence
        for ixr, iseq in self.experiment.codes.items():

            # split up each radar
            rdr_list = iseq.radarnames
            txrx = iseq.txorrx
            tx_int = [inum for inum, itxrx in enumerate(txrx) if itxrx == "tx"]
            rx_int = [inum for inum, itxrx in enumerate(txrx) if itxrx == "rx"]
            bco = iseq.beamcodes
            # setup the tx maps
            for ibco in bco:
                for itx_el in tx_int:
                    txname = rdr_list[itx_el]
                    tx_rdr = rdr_objs[txname]
                    tx_bco = [ibco[itx_el]]
                    # Lets assume that the frequencies are from the transmitter?
                    lam = sc.c / tx_rdr["radar"].freq
                    az_tx, el_tx, tx_ksys = tx_rdr["radar"].get_angle_info(tx_bco)
                    radarllat = tx_rdr["site"].get_lla()[:, np.newaxis]
                    enushift = ecef2enul(origecef, radarllat)

                    txecef = wgs2ecef(radarllat)
                    newx = coordinates["x"] + enushift[0, 0]
                    newy = coordinates["y"] + enushift[1, 0]
                    newz = coordinates["z"] + enushift[2, 0]

                    sp_coords = cartisian2Sphereical(np.vstack((newx, newy, newz)))
                    r_coords_t, az_coords, el_coords = sp_coords[:]
                    tx_beam = tx_rdr["radar"].antenna_func.calc_pattern(
                        az_coords, el_coords, az_tx[0], el_tx[0]
                    )
                    for irx_el in rx_int:
                        rxname = rdr_list[irx_el]
                        rx_rdr = rdr_objs[rxname]
                        rx_bco = [ibco[irx_el]]
                        cur_sp_setup = (txname, rxname, tx_bco[0], rx_bco[0])

                        if cur_sp_setup in sp_setup:
                            continue
                        else:
                            spatial_code += 1
                            sp_setup.append(cur_sp_setup)

                        az_rx, el_rx, rx_ksys = rx_rdr["radar"].get_angle_info(rx_bco)

                        radarlla = rx_rdr["site"].get_lla()[:, np.newaxis]
                        enushift = ecef2enul(origecef, radarlla)
                        rxecef = wgs2ecef(radarlla)
                        newx = coordinates["x"] + enushift[0, 0]
                        newy = coordinates["y"] + enushift[1, 0]
                        newz = coordinates["z"] + enushift[2, 0]
                        sp_coords = cartisian2Sphereical(np.vstack((newx, newy, newz)))
                        r_coords_r, az_coords, el_coords = sp_coords[:]
                        rx_beam = rx_rdr["radar"].antenna_func.calc_pattern(
                            az_coords, el_coords, az_rx[0], el_rx[0]
                        )

                        r_tr2 = np.sum((txecef - rxecef) ** 2)
                        cosalph = (r_coords_t**2 + r_coords_r**2 - r_tr2) / (
                            2 * r_coords_t * r_coords_r
                        )
                        if rx_rdr["radar"].rx_gain >= tx_rdr["radar"].tx_gain:
                            ant_gain = tx_rdr["radar"].tx_gain
                            rngloss = np.power(r_coords_t * 1e3, -2.0)
                        else:
                            ant_gain = rx_rdr["radar"].rx_gain
                            rngloss = np.power(r_coords_r * 1e3, -2.0)

                        ant_los = np.power(10, (ant_gain - rx_rdr["radar"].loss) / 10)
                        # do ksys that is not spatial
                        ksys_all = (
                            sc.c
                            * elec_radius**2
                            * ant_los
                            * lam**2
                            / (32.0 * np.log(2))
                        )
                        ksys_list.append(ksys_all)
                        # All of the spatial losses.
                        sp_los = rngloss / (1 + cosalph)
                        sp_loss_list.append(sp_los)
                        # Bistatic range
                        bis_rng = r_coords_t + r_coords_r
                        bi_rng_list.append(bis_rng)
                        # Beam pattern loss
                        beam_loss = tx_beam * rx_beam
                        beam_ls_list.append(beam_loss)

        coords_dict = dict(coordinates.variables)
        del coords_dict["time"]
        if "freqs" in coords_dict.keys():
            del coords_dict["freqs"]
        coords_dict["pairs"] = np.arange(spatial_code)
        attrs = {"pairs": sp_setup}
        sp_all = np.column_stack(sp_loss_list)
        beam_all = np.column_stack(beam_ls_list)
        bis_rng_alls = np.column_stack(bi_rng_list)
        idict = {
            "sploss": (dims, sp_all),
            "beam_loss": (dims, beam_all),
            "bi_rng": (dims, bis_rng_alls),
            "ksys": (("pairs"), np.array(ksys_list)),
        }
        phys_xr = xr.Dataset(idict, coords=coords_dict, attrs=attrs)
        return phys_xr

    def write_chan(self, sp_obj, phys_ds, rx_name, ichan_name, log_func=print):
        """Writes out channel of data over the whole expeirment for a single channel.

        Parameters
        ----------
        phys_ds : Dataset
            Xarray Dataset, uses input coordinates for spatial axis and adds a pairs coordinate. Holds range, and spatial losses as variables.
        origlla : ndarray
            wgs coordinates of the origin point for the coordinates. Should be part of the attr object in the Dataset.

        Returns
        -------
        phys_xr : Dataset
            Xarray Dataset, uses input coordinates for spatial axis and adds a pairs coordinate. Holds range, and spatial losses as variables.

        """
        rdr_cmb, cmball, tall = self.experiment.make_sequence()

        if isinstance(sp_obj, (Path, str)):
            spec_ds = xr.open_dataset(str(sp_obj), engine="netcdf4")
        elif isinstance(sp_obj, xr.Dataset):
            spec_ds = sp_obj
        else:
            raise ValueError("The specfile_obj needs to be a dataset or a filename.")

        coords = spec_ds.coords
        rdr_objs = self.experiment.radarobjs
        seq = self.experiment.codes
        seq_ord = self.experiment.code_order
        simdtype = "complex64"
        rx_obj = rdr_objs[rx_name]["radar"]

        # These are a pandas datetime object
        st_dt = self.experiment.exp_start
        end_dt = self.experiment.exp_end

        # Normalize everything to the beginning of the experiment.
        tvec_norm = spec_ds.coords["time"].data - st_dt
        # This is an overly clever way of finding things that are inbetween different data points.
        tall_q = np.digitize(tall, tvec_norm.astype(tall.dtype)) - 1

        pair_dict = {ip: inum for inum, ip in enumerate(phys_ds.attrs["pairs"])}
        spec_attrs = spec_ds.attrs
        # sample rate that the data will be created at.
        sr_create = Fraction(spec_attrs["sr_num"], spec_attrs["sr_den"])
        chan_obj = self.experiment.iline_chans[ichan_name]
        # Sample rate data will be saved at.
        sr_save = chan_obj.sr
        ds_fac = int(sr_save / sr_create)
        nlevel = sc.k * float(sr_save) * rx_obj.tsys
        clevel = sc.k * float(sr_save) * rx_obj.cal_temp

        # HACK number of lpc points connected to ratio of sampling frequency and
        # notial ion-line spectra with a factor of 10.
        nlpc = int(10 * sr_create / 20e3) + 1
        comboarr = rdr_cmb[rx_name]["rx"]
        pulse_perwrite = 200000
        tstart = np.arange(0, len(cmball), pulse_perwrite)
        tstop = np.roll(tstart, -1)
        tstop[-1] = len(cmball)
        outlist = []
        ord_list = []
        #
        for ist, iend in zip(tstart, tstop):
            curcmball = cmball[ist:iend]
            cur_tq = tall_q[ist:iend]

            for cur_comb in comboarr:
                cur_pidx = np.where(curcmball == cur_comb)[0]
                pl = len(cur_pidx)
                seq_info = self.experiment.seq_info[cur_comb]
                rdr_list = seq_info["radars"]
                txrx = seq_info["txrx"]
                bco = seq_info["bco"]
                pcodes = seq_info["pcode"]
                seq_num = seq_info["seq"]
                seq_obj = self.experiment.codes[seq_num]

                rx_log = np.logical_and(
                    np.array(txrx) == "rx", np.array(rdr_list) == rx_name
                )

                rx_rdr = np.where(rx_log)[0][0]
                cur_rx_pcode = pcodes[rx_rdr]
                rx_pobj = seq_obj.pulseseq[cur_rx_pcode]

                # get the rx raster
                rst = self.experiment.seq_info[cur_comb]["rasters"][rx_rdr]
                ns2save = int(1e9 / sr_save)
                # create the timing vectors for the pulse in ns and then down sample appropriately.

                fasttime_sig = np.arange(*rst["signal"])[::ns2save]
                nsamp = rst["full"][1] // ns2save
                sig_pos = fasttime_sig // ns2save
                full_sig = np.arange(*rst["full"])[::ns2save]  # pulse pulse
                cal_sig = np.arange(*rst["calibration"])[::ns2save]  # cal time
                cal_sam = cal_sig // (ns2save)
                ncal = (rst["calibration"][1] - rst["calibration"][0]) // ns2save
                fast_time_cr = fasttime_sig[::ds_fac]

                rx_rng = fast_time_cr * sc.c * 1e-9 * 1e-3
                rng_res = sc.c * 1e-9 * 1e-3 * ns2save * ds_fac
                ft_res = 1e-9 * ns2save * ds_fac
                # Now go through all of the transmiters for this receiver
                tx_locs = np.where(np.array(txrx) == "tx")[0]
                rawdata = np.zeros((len(cur_pidx), len(rx_rng)), dtype=simdtype)
                uniqtq, inv_ind = np.unique(cur_tq, return_inverse=True)

                for iutn, iut in enumerate(uniqtq):
                    cur_ind = np.where(inv_ind == iutn)[0]
                    for itx_loc in tx_locs:
                        txname = rdr_list[itx_loc]
                        tx_obj = rdr_objs[txname]["radar"]
                        txp = tx_obj.tx_power

                        pair_tup = (
                            rdr_list[itx_loc],
                            rx_name,
                            bco[itx_loc],
                            bco[rx_rdr],
                        )
                        curpair = pair_dict[pair_tup]

                        cur_tx_pcode = pcodes[itx_loc]
                        tx_pobj = seq_obj.pulseseq[cur_tx_pcode]
                        pulse_env = tx_pobj.get_iq(sr_create)
                        # change grom bistatic range to km
                        phys_ds_cur_pair = phys_ds.isel(pairs=curpair)
                        sysw = phys_ds_cur_pair.ksys.data * ft_res * txp
                        bi_rng_data = phys_ds_cur_pair.bi_rng.data

                        rx_rng_log = np.logical_and(
                            rx_rng >= bi_rng_data.min(),
                            rx_rng + rng_res < bi_rng_data.max(),
                        )
                        rx_rng_it = np.where(rx_rng_log)[0]

                        # get info for each range gate

                        for irng in rx_rng_it:
                            cur_rng = rx_rng[irng]
                            cur_rng_log = np.logical_and(
                                bi_rng_data >= cur_rng, bi_rng_data < cur_rng + rng_res
                            )
                            rng_locs = np.where(cur_rng_log)[0]
                            # Hack if this is the case should we do interpolation?
                            if sum(cur_rng_log) == 0:
                                continue
                            pds = phys_ds.isel(pairs=curpair, locs=rng_locs)
                            scene = pds.beam_loss * pds.sploss
                            sds = spec_ds.isel(locs=rng_locs, time=iut)

                            spec_scen = scene * sds.iline
                            cur_spec = spec_scen.mean(dim="locs").data
                            rcs = scene * sds.rcs
                            mainmult = sysw * rcs.mean(dim="locs").data
                            nrepp = len(cur_ind)
                            hkidn = np.arange(nrepp, dtype=int)
                            p_mat = np.tile(pulse_env[np.newaxis, :], (nrepp, 1))
                            cur_pulse_data = MakePulseDataRepLPC(
                                p_mat, cur_spec, nlpc, hkidn, numtype=simdtype
                            )
                            rngspd = min(irng + len(pulse_env), len(rx_rng))
                            rng_sl = slice(irng, rngspd)
                            pden = cur_pulse_data.flatten().var()
                            cpd = np.sqrt(mainmult / pden) * cur_pulse_data

                            for idatn, idat in enumerate(cur_ind):
                                rawdata[idat, rng_sl] += cpd[idatn, : rngspd - irng]
                # Data for receiver noise across entire IPP
                r_rand = np.random.randn(pl, nsamp)
                i_rand = np.random.randn(pl, nsamp)
                # Calibration noise injected
                r_cal = np.random.randn(pl, len(cal_sam))
                i_cal = np.random.randn(pl, len(cal_sam))
                # Create vector of noise after it properly measured
                xout = np.sqrt(nlevel / 2) * (r_rand + 1j * i_rand)
                # Add the calibration information to the pulses
                xout[:, cal_sam] += np.sqrt(clevel / 2) * (r_cal + 1j * i_cal)
                # up sample the data
                # This specific type of resampling keeps the power level the same without any sort of extra factors thrown in.
                rawout = sig.resample(rawdata, len(rx_rng) * ds_fac, axis=1)
                xout[:, sig_pos] += rawout
                # Now scale everything to be approximately at the noise level.
                xout = xout / np.sqrt(nlevel / 2)
                xlist = [i for i in xout]
                outlist = outlist + xlist
                ord_list = ord_list + list(cur_ind)

            # for the id_meta need index,modeid(sequence code), sweepid(pulsecode), sweepnum (pulsenum)
            sort_idx = np.argsort(ord_list)
            out_2 = [outlist[ind] for ind in sort_idx]
            outdata = np.concatenate(out_2, axis=0)
            chan_obj.drf_out.rf_write(outdata.astype(chan_obj.numtype))

        # write out the Tx channels
        # for irdr, chan_dict in self.radar2chans.items():
        #     tx_chans  = chan_dict['txpulse']
        #     for itx_chan in tx_chans:

        # for ichname, ichan in self.experiment.tx_chans.items():

        # # write out the ion-line channesl
        # for ichname,
        # # #write out pline channels


def create_pline(
    rng_vec,
    specs,
    nchans,
    numtaps,
):
    fs = 25000000

    nchans = 250
    c_bw = fs / nchans
    n_fpfb = 1024
    freq_vec = np.fft.fftshift(np.fft.fftfreq(nchans, 1 / fs))
    freq_ind = np.arange(nchans)
    bw2 = fs / nchans / 2
    freq_l = freq_vec - bw2
    freq_h = freq_vec + bw2
    cfreqvec = np.fft.fftshift(np.fft.fftfreq(n_fpfb, 1 / c_bw))
    fwhm = 2 * gam
    skirt = fwhm * 2
    npulses = 100

    g_del = nchans * (ntaps - 1) // 2 // 2
    g_delp = nchans * (ntaps - 1) // 2

    n_rg_bins = len(rng_vec) + plen + g_delp // nchans
    rng_s = 1e-3 * sconst.c / c_bw / 2

    dout = np.zeros((nchans, n_rg_bins, npulses), dtype=np.complex64)
    dpulse = np.ones((npulses, plen))
    pkeep = np.arange(npulses)
    for irng, irngv in enumerate(rng_vec):
        # lower line
        f_0m = frm[irng]
        lb = freq_ind[f_0m - skirt > freq_l][-1]
        ub = freq_ind[f_0m + skirt < freq_h][0]
        cur_bin = freq_ind[lb : ub + 1]
        # cur_bin = np.where(np.logical_and(ub,lb))[0]

        # if len(cur_bin)==0:
        #     ipdb.set_trace()
        for bin_i in cur_bin:
            cf_i = freq_vec[bin_i]
            spec_i = spec_func(cfreqvec + cf_i, f_0m, gam)
            data_i = MakePulseDataRepLPC(
                dpulse, spec_i, 25, pkeep, numtype=np.complex64
            )
            dout[bin_i, irng : irng + plen, pkeep] += data_i

        # #upper line
        f_0p = frp[irng]
        lb = freq_ind[f_0p - skirt > freq_l][-1]
        ub = freq_ind[f_0p + skirt < freq_h][0]
        cur_bin = freq_ind[lb : ub + 1]

        for bin_i in cur_bin:
            cf_i = freq_vec[bin_i]
            spec_i = spec_func(cfreqvec + cf_i, f_0p, gam)
            data_i = MakePulseDataRepLPC(
                dpulse, spec_i, 25, pkeep, numtype=np.complex64
            )
            dout[bin_i, irng : irng + plen, pkeep] += data_i

    coeffs = rref_coef(nchans, ntaps)
    # have to flip the bins because the synthesis is expecting things to be in the wrong direction.
    dout = np.fft.fftshift(dout[::-1], axes=0)
    full_data = npr_synthesis(dout, nchans, coeffs)
    full_data = np.roll(full_data, -g_del, axis=0)
    syn_coeffs = kaiser_syn_coeffs(nchans, 8)
    mask = np.ones(nchans, dtype=bool)
    fillmethod = ""
    fillparams = [0, 0]

    full_data_pfb = pfb_reconstruct(
        dout, nchans, syn_coeffs, mask, fillmethod, fillparams=[], realout=False
    )
    full_data_pfb = np.roll(full_data_pfb, -g_delp, 0)
    return full_data_pfb


class RadarDataFile(object):
    """
    This class will will take the ionosphere class and create radar data both
    at the IQ and fitted level.

    Variables
    simparams - A dictionary that holds simulation parameters the keys are the following
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
        """This function will create an instance of the RadarData class.  It will take in the values and create the class and make raw IQ data.

        Parameters
        ----------
        config : str
            Configuration file
        outdir : str
            Directory data will be saved.
        """
        (self.sensdict, self.simparams) = readconfigfile(config)

        self.datadir = Path(outdir)
        self.maindir = self.datadir.parent
        self.procdir = self.maindir / "ACF"
        self.drfdir = self.maindir / "drfdata"
        # HACK Need to clean up for mulitple channels
        self.drfdir_rf = self.drfdir.joinpath("rf_data")
        self.drfdir_md = self.drfdir.joinpath("metadata")
        # Make data
        self.outfilelist = []
        self.timeoffset = 0
        if not outfilelist is None:
            self.premadesetup(outfilelist)

    def premadesetup(self, outfilelist):
        """
        If data has been previously made set it up accordinly
        """
        infodict = load_dict_from_hdf5(str(self.datadir.joinpath("INFO.h5")))
        alltime = np.hstack(infodict["Time"])
        self.timeoffset = alltime.min()
        self.outfilelist = outfilelist

    def makerfdata(self, ionodict):
        """
        This method will take an ionocontainer object and create the associated
        RF data assoicated with the parameters from it.
        """

        filetimes = list(ionodict.keys())
        filetimes.sort()
        ftimes = np.array(filetimes)
        N_angles = len(self.simparams["angles"])
        Npall = np.floor(self.simparams["TimeLim"] / self.simparams["IPP"])
        Npall = int(np.floor(Npall / N_angles) * N_angles)
        Np = Npall / N_angles

        sweepids = self.simparams["sweepids"]
        sweepnums = self.simparams["sweepnums"]
        nseq = len(sweepids)
        usweep = np.unique(sweepids)
        ippsamps = self.simparams["IPPsamps"]
        d_samps = self.simparams["datasamples"]
        n_samps = self.simparams["noisesamples"]
        c_samps = self.simparams["calsamples"]
        n_ns = n_samps[1] - n_samps[0]
        n_cal = c_samps[1] - c_samps[0]
        t_dict = self.simparams["Timing_Dict"]
        ## HACK: Set max number of pulses per write based off of ippsamps
        max_write = 2 * 2**30  # 2 gigabye Write
        bps = 16  # byte per sample for complex128 in numpy array
        nippw = int(max_write / bps / ippsamps)  # number of ipps per write

        dec_list = self.simparams["declist"]
        ds_fac = np.prod(dec_list)

        simdtype = self.simparams["dtype"]
        pulsetimes = np.arange(Npall) * self.simparams["IPP"] + ftimes.min()
        pulsefile = np.array(
            [np.where(itimes - ftimes >= 0)[0][-1] for itimes in pulsetimes]
        )
        pulse_full = self.simparams["Pulse"]
        print("\nData Now being created.")

        # digital rf stuff

        sample_rate_numerator = self.simparams["fsnum"]
        sample_rate_denominator = self.simparams["fsden"]
        sample_rate = np.double(sample_rate_numerator) / sample_rate_denominator
        start_global_index = int(sample_rate * filetimes[0])
        dtype_str = "complex64"  # complex64
        sub_cadence_secs = 3600  # Number of seconds of data in a subdirectory
        file_cadence_millisecs = 10000  # Each file will have up to 400 ms of data
        compression_level = 0  # no compression
        checksum = False  # no checksum
        is_complex = True  # complex values
        is_continuous = True
        num_subchannels = 1  # only one subchannel
        marching_periods = False  # no marching periods when writing
        uuid = "SimISRDRFZenith"
        zndatadir = self.drfdir_rf.joinpath("zenith-l")
        data_object = drf.DigitalRFWriter(
            str(zndatadir),
            dtype_str,
            sub_cadence_secs,
            file_cadence_millisecs,
            start_global_index,
            sample_rate_numerator,
            sample_rate_denominator,
            uuid,
            compression_level,
            checksum,
            is_complex,
            num_subchannels,
            is_continuous,
            marching_periods,
        )
        drfdirtx = self.drfdir_rf.joinpath("tx-h")
        data_object_tx = drf.DigitalRFWriter(
            str(drfdirtx),
            dtype_str,
            sub_cadence_secs,
            file_cadence_millisecs,
            start_global_index,
            sample_rate_numerator,
            sample_rate_denominator,
            uuid,
            compression_level,
            checksum,
            is_complex,
            num_subchannels,
            is_continuous,
            marching_periods,
        )
        # Noise Scaling
        noisepwr = sc.k * self.sensdict["Tsys"] * sample_rate
        calpwr = sc.k * self.sensdict["CalDiodeTemp"] * sample_rate
        # digital metadata
        # TODO temp for original data
        # antenna control

        dmddir = self.datadir.parent.joinpath("drfdata", "metadata")
        acmdir = dmddir.joinpath("antenna_control_metadata")
        acmdict = {
            "misa_elevation": 88.000488281200006,
            "cycle_name": "zenith_record_cycle_0",
            "misa_azimuth": 178.000488281,
            "rx_antenna": "ZENITH",
            "tx_antenna": "ZENITH",
        }
        acmobj = drf.DigitalMetadataWriter(
            str(acmdir), 3600, 60, sample_rate_numerator, sample_rate_denominator, "ant"
        )
        # id metadata
        iddir = dmddir.joinpath("id_metadata")

        idmobj = drf.DigitalMetadataWriter(
            str(iddir), 3600, 1, sample_rate_numerator, sample_rate_denominator, "md"
        )
        # power mete metadata
        pmdir = dmddir.joinpath("powermeter")
        # power metadata is transmit power and with keys of zenith_power, and misa_power.
        pmmdict = {"zenith_power": self.sensdict["Pt"], "misa_power": 3110.3}
        pmmobj = drf.DigitalMetadataWriter(str(pmdir), 3600, 5, 1, 1, "power")
        # differentiate between phased arrays and dish antennas
        if self.sensdict["Name"].lower() in ["risr", "pfisr", "risr-n"]:

            beams = np.tile(np.arange(N_angles), Npall // N_angles)
        else:

            # for dish arrays
            brate = self.simparams["beamrate"]
            beams2 = np.repeat(np.arange(N_angles), brate)
            beam3 = np.concatenate((beams2, beams2[::-1]))
            ntile = int(np.ceil(Npall / len(beam3)))
            leftover = int(Npall - ntile * len(beam3))
            if ntile > 0:
                beams = np.tile(beam3, ntile)
                beams = np.concatenate((beams, beam3[:leftover]))
            else:
                beams = beam3[:leftover]
        pulsen = np.repeat(np.arange(Np), N_angles).astype(int)
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
            f_pulses = np.where(pulsefile == ifn)[0]
            nwrites = np.ceil(float(len(f_pulses)) / nippw).astype(int)
            for ifwrite, i_pstart in enumerate(range(0, len(f_pulses), nippw)):
                progstr1 = "Data written, {:d} of {:d} now being created."

                prog_level = float(ifn) / Nf + float(ifwrite) / Nf / nwrites
                update_progress(
                    prog_level,
                    progstr1.format(
                        int(ifn * nwrites + ifwrite + 1), int(Nf * nwrites)
                    ),
                )
                i_pend = np.minimum(i_pstart + nippw, len(f_pulses))
                cur_fpulses = f_pulses[i_pstart:i_pend]
                pt = pulsetimes[cur_fpulses]
                pb = beams[cur_fpulses]
                pn = pulsen[cur_fpulses]
                pulse_idx = pn % nseq
                s_i = sweepids[pulse_idx]
                s_n = sweepnums[pulse_idx]

                rawdata = self.__makeTime__(
                    pt,
                    curcontainer.Time_Vector,
                    curcontainer.Sphere_Coords,
                    curcontainer.Param_List,
                    pb,
                    pulse_idx,
                )
                n_pulse_cur, n_raw = rawdata.shape
                # nbreak = 400
                r_samps = np.diff(d_samps)[0]
                # orig_ind = np.arange(n_pulse_cur)
                # indsplit = np.array_split(orig_ind,nbreak)
                # arg_list = [(iindx,rawdata[iindx],n_raw*ds_fac,1) for iindx in indsplit]
                # pool = mp.Pool(processes=self.simparams['numprocesses'])
                #
                # results = [pool.apply_async(resample_worker, args=x) for x in arg_list]
                # results = [p.get() for p in results]
                # rawdata_us = np.zeros((n_pulse_cur,r_samps), dtype=simdtype)
                # for p in results:
                #     iindx, idata = p.get()
                #     rawdata_us[iindx] = idata[:,:r_samps]
                rawdata_us = sig.resample(rawdata, n_raw * ds_fac, axis=1)
                rawdata_us = rawdata_us[:, :r_samps]
                alldata = 1j * np.random.randn(n_pulse_cur, ippsamps)
                alldata = alldata + np.random.randn(n_pulse_cur, ippsamps)
                alldata = np.sqrt(noisepwr / 2.0) * alldata.astype(simdtype)
                caldata = np.random.randn(n_pulse_cur, n_cal) + 1j * np.random.randn(
                    n_pulse_cur, n_cal
                )
                caldata = np.sqrt(calpwr / 2) * caldata.astype(simdtype)
                # noisedata = alldata[:, d_samps[0]:d_samps[1]]

                for i_swid in usweep:
                    sw_ind = np.where(s_i == i_swid)[0]
                    cur_tdict = t_dict[i_swid]
                    cur_c = cur_tdict["calibration"]
                    cur_csamps = np.diff(cur_c)[0]
                    alldata[sw_ind, d_samps[0] : d_samps[1]] += rawdata_us[sw_ind]
                    alldata[sw_ind, cur_c[0] : cur_c[1]] += caldata[sw_ind, :cur_csamps]

                alldata = alldata / np.sqrt(calpwr / 2.0)
                # rawdata = rawdata_us.copy()/np.sqrt(calpwr/2.)
                # noisedata = noisedata.copy()/np.sqrt(calpwr/2.)
                # cal_samps = np.zeros_like(caldata)
                # data_samps = np.zeros_like(rawdata_us)
                # noise_samps = np.zeros((n_pulse_cur, n_ns), dtype=alldata.dtype)
                #
                # for i_swid in usweep:
                #     sw_ind = np.where(s_i == i_swid)[0]
                #     cur_tdict = t_dict[i_swid]
                #     cur_c = cur_tdict['calibration']
                #     cur_csamps = np.diff(cur_c)[0]
                #     cur_n = cur_tdict['noise']
                #     cur_nsamps = np.diff(cur_n)[0]
                #     data_samps[sw_ind] = alldata[sw_ind, d_samps[0]:d_samps[1]]
                #     cal_samps[sw_ind, :cur_csamps] = alldata[sw_ind, cur_c[0]:cur_c[1]]
                #     noise_samps[sw_ind, :cur_nsamps] = alldata[sw_ind, cur_n[0]:cur_n[1]]

                # data_samps = sig.resample(data_samps, n_raw, axis=1)
                # noise_samps_ds = sig.resample(noise_samps, n_ns/ds_fac, axis=1)
                # # Down sample data using resample, keeps variance correct
                # rawdata_ds = sig.resample(rawdata_us, rawdata.shape[1]/ds_fac, axis=1)
                # noisedata_ds = sig.resample(noisedata, noisedata.shape[1]/ds_fac, axis=1)
                # noise_est = np.mean(np.mean(noise_samps.real**2+noise_samps.imag**2))
                # cal_est = np.mean(np.mean(cal_samps.real**2+cal_samps.imag**2))
                # calfac = calpwr/(cal_est-noise_est)
                # outdict['AddedNoise'] = noisedata_ds
                # outdict['RawData'] = data_samps
                # outdict['RawDatanonoise'] = rawdata_ds
                # outdict['NoiseData'] = noise_samps_ds
                # outdict['CalFactor'] = np.array([calfac])
                # outdict['Pulses'] = pn
                # outdict['Beams'] = pb
                # outdict['Time'] = pt
                # fname = '{0:d} RawData.h5'.format(ifwrite)
                # newfn = self.datadir/fname
                # self.outfilelist.append(str(newfn))
                # dict2h5(str(newfn), outdict)

                # Listing info
                # pt_list.append(pt)
                # pb_list.append(pb)
                # pn_list.append(pn)
                # si_list.append(s_i)
                # sn_list.append(s_n)
                # fname_list.append(fname)

                # HACK Add a constant to set the noise level to be atleast 1 bit
                num_const = 200.0
                # transmitdata
                tx_data = np.zeros_like(alldata).astype("complex64")
                tx_data[:, : pulse_full.shape[-1]] = pulse_full[pulse_idx].astype(
                    "complex64"
                )
                data_object_tx.rf_write(tx_data.flatten() * num_const)
                # extend array for digital rf to flattend array

                data_object.rf_write(alldata.flatten().astype("complex64") * num_const)
                del alldata
                id_strt = int(idmobj.get_samples_per_second() * pt[0])
                dmdplist = np.arange(n_pulse_cur, dtype=int) * ippsamps + id_strt
                acmobj.write(int(acmobj.get_samples_per_second() * pt[0]), acmdict)
                idmdict = {
                    "sweepid": s_i,
                    "sample_rate": np.array([sample_rate] * len(s_n)),
                    "modeid": np.array([100000001] * len(s_n)),
                    "sweepnum": s_n,
                }
                idmobj.write(dmdplist, idmdict)
                pmmobj.write(int(pmmobj.get_samples_per_second() * pt[0]), pmmdict)

        # infodict = {'Files':fname_list, 'Time':pt_list, 'Beams':pb_list, 'Pulses':pn_list}
        # dict2h5(str(self.datadir.joinpath('INFO.h5')), infodict)
        data_object.close()
        data_object_tx.close()

    # %% Make functions
    def __makeTime__(
        self, pulsetimes, spectime, Sphere_Coords, allspecs, beamcodes, pulse_idx
    ):
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
        range_gates = self.simparams["Rangegates"]
        sensdict = self.sensdict
        samp_range = self.simparams["datasamples"]
        ds_fac = np.prod(self.simparams["declist"])
        f_s = float(self.simparams["fsnum"]) / self.simparams["fsden"] / ds_fac
        t_s = float(ds_fac * self.simparams["fsden"]) / self.simparams["fsnum"]
        pulse = self.simparams["Pulse"][:, ::ds_fac]
        # HACK number of lpc points connected to ratio of sampling frequency and
        # notial ion-line spectra with a factor of 10.
        nlpc = int(10 * f_s / 20e3) + 1
        pulse2spec = np.array(
            [np.where(itimes - spectime >= 0)[0][-1] for itimes in pulsetimes]
        )
        n_pulses = len(pulse2spec)
        lp_pnts = pulse.shape[-1]
        samp_num = np.arange(lp_pnts)
        n_samps = int(np.ceil(float(samp_range[1] - samp_range[0]) / ds_fac))

        angles = self.simparams["angles"]
        n_beams = len(angles)
        rho = Sphere_Coords[:, 0]
        Az = Sphere_Coords[:, 1]
        El = Sphere_Coords[:, 2]

        rng_len = t_s * sc.c * 1e-3 / 2.0
        speclen = allspecs.shape[-1]
        simdtype = self.simparams["dtype"]
        out_data = np.zeros((n_pulses, n_samps), dtype=simdtype)
        weights = {
            ibn: self.sensdict["ArrayFunc"](
                Az, El, ib[0], ib[1], sensdict["Angleoffset"]
            )
            for ibn, ib in enumerate(angles)
        }
        ntime = len(spectime)
        # pool = mp.Pool(processes=self.simparams['numprocesses'])
        results = []
        for istn in range(ntime):
            for ibn in range(n_beams):
                # print('\t\t Making Beam {0:d} of {1:d}'.format(ibn, n_beams))
                weight = weights[ibn]
                for isamp in np.arange(n_samps):

                    range_g = range_gates[isamp * ds_fac]

                    range_m = range_g * 1e3
                    rnglims = [range_g - rng_len / 2.0, range_g + rng_len / 2.0]
                    rangelog = (rho >= rnglims[0]) & (rho < rnglims[1])
                    # Get the number of points covered
                    cur_pnts = samp_num + isamp
                    cur_pnts = cur_pnts[cur_pnts < n_samps]
                    # This is a nearest neighbors interpolation for the
                    # spectrums in the range domain
                    if np.sum(rangelog) == 0:
                        minrng = np.argmin(np.absolute(range_g - rho))

                        rangelog[minrng] = True

                    # create the weights and weight location based on the beams pattern.
                    weight_cur = weight[rangelog]
                    weight_cur = weight_cur / weight_cur.sum()
                    specsinrng = allspecs[rangelog]
                    if specsinrng.ndim == 3:
                        specsinrng = specsinrng[:, istn]
                    elif specsinrng.ndim == 2:
                        specsinrng = specsinrng[istn]
                    specsinrng = specsinrng * np.tile(
                        weight_cur[:, np.newaxis], (1, speclen)
                    )
                    cur_spec = specsinrng.sum(0)
                    # based off new way of calculating
                    pow_num = sensdict["Pt"] * sensdict["Ksys"][ibn] * t_s
                    pow_den = range_m**2
                    curdataloc = np.where(
                        np.logical_and((pulse2spec == istn), (beamcodes == ibn))
                    )[0]
                    cur_pidx = pulse_idx[curdataloc]

                    # create data
                    if not np.any(curdataloc):
                        # outstr = '\t\t No data for {0:d} of {1:d} in this time period'
                        # print(outstr.format(ibn, n_beams))
                        continue
                    # mpargs = (pulse, cur_spec, nlpc, cur_pidx, cur_pnts, curdataloc, isamp, simdtype, np.sqrt(pow_num/pow_den))
                    # results.append(pool.apply_async(pulseworkerfunction, args=mpargs))

                    cur_pulse_data = MakePulseDataRepLPC(
                        pulse, cur_spec, nlpc, cur_pidx, numtype=simdtype
                    )
                    cur_pulse_data = cur_pulse_data[:, cur_pnts - isamp] * np.sqrt(
                        pow_num / pow_den
                    )
                    # Need to do the adding in a loop, can't find a way to get a round this.
                    for idatn, idat in enumerate(curdataloc):
                        out_data[idat, cur_pnts] += cur_pulse_data[idatn]
        # for p in results:
        #     (cur_pnts, curdataloc, cur_pulse_data,cur_pidx) = p.get()
        #     for idatn, idat in enumerate(curdataloc):
        #         out_data[idat, cur_pnts] += cur_pulse_data[idatn]
        return out_data
        # %% Processing

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

        dec_list = self.simparams["declist"]
        ds_fac = np.prod(dec_list)
        t_dict = self.simparams["Timing_Dict"]
        t_s = float(self.simparams["fsden"]) / self.simparams["fsnum"]
        rng_samprate = t_s * sc.c * 1e-3 / 2.0
        plen_ds = self.simparams["Pulse"].shape[-1] / ds_fac
        if "LP" in outdict:
            datalagslp = outdict["LP"]["datalags"]
            noiselagslp = outdict["LP"]["noiselags"]
            d_len = t_dict[300]["signal"]
            rng_gates = np.arange(d_len[0], d_len[1]) * rng_samprate

            rng_gates_ds = rng_gates[::ds_fac]
            self.simparams["Rangegates"] = rng_gates
            minrg = plen_ds - 1
            maxrg = len(rng_gates_ds) - plen_ds + 1
            self.simparams["Rangegatesfinal"] = rng_gates_ds[minrg:maxrg]
            (ionoout, ionosig) = lagdict2ionocont(
                datalagslp,
                noiselagslp,
                self.sensdict,
                self.simparams,
                datalagslp["Time"],
            )
            alldata["LP"] = [ionoout, ionosig]
        if "AC" in outdict:
            datalagsac = outdict["AC"]["datalags"]
            noiselagsac = outdict["AC"]["noiselags"]

            d_len = t_dict[1]["signal"]
            rng_gates = np.arange(d_len[0], d_len[1]) * rng_samprate

            rng_gates_ds = rng_gates[::ds_fac]
            self.simparams["Rangegates"] = rng_gates
            minrg = plen_ds - 1
            maxrg = len(rng_gates_ds) - plen_ds + 1
            self.simparams["Rangegatesfinal"] = rng_gates_ds[minrg:maxrg]
            ionoout, ionosig = lagdict2ionocont(
                datalagsac,
                noiselagsac,
                self.sensdict,
                self.simparams,
                datalagsac["Time"],
            )
            alldata["AC"] = [ionoout, ionosig]
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
        t_dict = self.simparams["Timing_Dict"]
        rfdir = self.drfdir_rf

        simdtype = self.simparams["dtype"]
        lagtype = self.simparams["lagtype"]
        dmddir = outdir.parent.joinpath("drfdata", "metadata")
        acmdir = dmddir.joinpath("antenna_control_metadata")
        iddir = dmddir.joinpath("id_metadata")
        acode_swid = np.arange(1, 33)
        lp_swid = 300

        dec_list = self.simparams["declist"]
        ds_fac = np.prod(dec_list)
        pulse_arr = self.simparams["Pulse"][:, ::ds_fac]
        nlags = pulse_arr.shape[-1]
        s_id_p1 = self.simparams["sweepids"]
        pulse_dict = {i: j for i, j in zip(s_id_p1, pulse_arr)}
        idmobj = drf.DigitalMetadataReader(str(iddir))
        drfObj = drf.DigitalRFReader(str(rfdir))
        objprop = drfObj.get_properties("zenith-l")
        sps = objprop["samples_per_second"]
        d_bnds = drfObj.get_bounds("zenith-l")

        time_list = self.simparams["Timevec"] * sps + d_bnds[0]
        time_list = time_list.astype(int)
        calpwr = sc.k * self.sensdict["CalDiodeTemp"] * sps / ds_fac
        pulse_dict = {i: j for i, j in zip(s_id_p1, pulse_arr)}

        ac_in_swid = [i in pulse_dict.keys() for i in acode_swid]
        if np.any(ac_in_swid):
            ac_ind = np.where(ac_in_swid)[0]
            ac_swid = np.array(pulse_dict.keys())[ac_ind]
            ac_codes = np.array([pulse_dict[iac] for iac in ac_swid])
            num_codes = len(ac_codes)
            decode_arr = np.zeros((num_codes, nlags))
            for ilag in range(nlags):
                if ilag == 0:
                    decode_arr[:, ilag] = np.sum(ac_codes * ac_codes, axis=-1) / nlags
                else:
                    decode_arr[:, ilag] = np.sum(
                        ac_codes[:, :-ilag] * ac_codes[:, ilag:], axis=-1
                    )
            decode_dict = {ac_swid[i]: decode_arr[i] for i in range(num_codes)}
        # Choose type of processing
        if self.simparams["Pulsetype"].lower() == "barker":
            lagfunc = BarkerLag
            Nlag = 1
        else:
            lagfunc = CenteredLagProduct

        mode_dict = {"AC": {"swid": acode_swid.tolist()}, "LP": {"swid": [lp_swid]}}
        # make data lags
        Ntime = len(time_list)
        n_beams = 1
        timemat = np.column_stack(
            (time_list / sps, time_list / sps + int(self.simparams["Tint"]))
        )
        for i_type in mode_dict.keys():
            swid_1 = mode_dict[i_type]["swid"][0]
            t_info = t_dict[swid_1]
            d_samps = int(np.diff(t_info["signal"])[0] / ds_fac)
            n_samps = int(np.diff(t_info["noise"])[0] / ds_fac)
            Nlag = pulse_dict[swid_1].shape[0]
            outdata = np.zeros(
                (Ntime, n_beams, d_samps - Nlag + 1, Nlag), dtype=simdtype
            )
            outnoise = np.zeros(
                (Ntime, n_beams, n_samps - Nlag + 1, Nlag), dtype=simdtype
            )
            pulses = np.zeros((Ntime, n_beams))
            outcal = np.zeros((Ntime, n_beams))
            data_lags = {
                "ACF": outdata,
                "Pulses": pulses,
                "Time": timemat,
                "CalFactor": outcal,
            }
            noise_lags = {"ACF": outnoise, "Pulses": pulses, "Time": timemat}
            mode_dict[i_type]["datalags"] = data_lags
            mode_dict[i_type]["noiselags"] = noise_lags

        for itn, itb in enumerate(time_list):
            ite = itb + int(self.simparams["Tint"] * sps)
            id_dict = idmobj.read_flatdict(itb, ite)

            un_id, id_ind_list = np.unique(id_dict["sweepid"], return_inverse=True)
            p_indx = id_dict["index"]
            data_dict = {
                "AC": {"sig": [], "cal": [], "noise": [], "NIPP": []},
                "LP": {"sig": None, "cal": None, "noise": None, "NIPP": None},
            }
            for i_idn, i_id in enumerate(un_id):
                t_info = t_dict[i_id]
                ipp_samp = t_info["full"][1]
                sig_bnd = t_info["signal"]
                noise_bnd = t_info["noise"]
                cal_bnd = t_info["calibration"]
                cur_pulse = pulse_dict[i_id]
                curlocs = np.where(id_ind_list == i_idn)[0]
                sig_data = np.zeros(
                    (len(curlocs), np.diff(sig_bnd)[0]), dtype=np.complex64
                )
                cal_data = np.zeros(
                    (len(curlocs), np.diff(cal_bnd)[0]), dtype=np.complex64
                )
                n_data = np.zeros(
                    (len(curlocs), np.diff(noise_bnd)[0]), dtype=np.complex64
                )
                # HACK Need to come up with clutter cancellation
                for ar_id, id_ind in enumerate(curlocs):
                    dr_ind = p_indx[id_ind]
                    raw_data = drfObj.read_vector(dr_ind, ipp_samp, "zenith-l", 0)
                    sig_data[ar_id] = raw_data[sig_bnd[0] : sig_bnd[1]]
                    cal_data[ar_id] = raw_data[cal_bnd[0] : cal_bnd[1]]
                    n_data[ar_id] = raw_data[noise_bnd[0] : noise_bnd[1]]
                # Down sample data
                sig_data = sig.resample(sig_data, sig_data.shape[1] / ds_fac, axis=1)
                cal_data = sig.resample(cal_data, cal_data.shape[1] / ds_fac, axis=1)
                n_data = sig.resample(n_data, n_data.shape[1] / ds_fac, axis=1)
                # make lag products
                sig_acf = lagfunc(
                    sig_data, numtype=simdtype, pulse=cur_pulse, lagtype=lagtype
                )
                n_acf = lagfunc(
                    n_data, numtype=simdtype, pulse=cur_pulse, lagtype=lagtype
                )
                cal_acf = lagfunc(
                    cal_data, numtype=simdtype, pulse=cur_pulse, lagtype=lagtype
                )
                cal_acf = np.median(cal_acf, axis=0)[0].real

                if i_id in acode_swid:
                    decode_ar = decode_dict[i_id]
                    dcsig_acf = sig_acf * np.repeat(
                        decode_ar[np.newaxis], sig_acf.shape[0], axis=0
                    )
                    data_dict["AC"]["sig"].append(dcsig_acf)
                    data_dict["AC"]["cal"].append(cal_acf)
                    dc_n_acf = n_acf * np.repeat(
                        decode_ar[np.newaxis], n_acf.shape[0], axis=0
                    )
                    data_dict["AC"]["noise"].append(dc_n_acf)
                    data_dict["AC"]["NIPP"].append(len(curlocs))
                elif i_id == lp_swid:
                    data_dict["LP"] = {
                        "sig": sig_acf,
                        "cal": cal_acf,
                        "noise": n_acf,
                        "NIPP": len(curlocs),
                    }

            # AC stuff
            if data_dict["AC"]["sig"]:
                data_dict["AC"] = {
                    ikey: sum(data_dict["AC"][ikey]) for ikey in data_dict["AC"]
                }
                cal_est = data_dict["AC"]["cal"] / data_dict["AC"]["NIPP"]
                noise_est = (
                    np.median(data_dict["AC"]["noise"].real, axis=0)[0]
                    / data_dict["AC"]["NIPP"]
                )
                calfac = calpwr / (cal_est - noise_est)
                mode_dict["AC"]["datalags"]["ACF"][itn, 0] = data_dict["AC"]["sig"]
                mode_dict["AC"]["datalags"]["Pulses"][itn, 0] = data_dict["AC"]["NIPP"]
                mode_dict["AC"]["datalags"]["CalFactor"][itn, 0] = calfac

                mode_dict["AC"]["noiselags"]["ACF"][itn, 0] = data_dict["AC"]["noise"]
                mode_dict["AC"]["noiselags"]["Pulses"][itn, 0] = data_dict["AC"]["NIPP"]
            else:
                del mode_dict["AC"]
            # LP stuff
            if data_dict["LP"]["sig"] is None:
                del mode_dict["LP"]
            else:
                cal_est = data_dict["LP"]["cal"] / data_dict["LP"]["NIPP"]
                noise_est = (
                    np.median(data_dict["LP"]["noise"].real, axis=0)[0]
                    / data_dict["LP"]["NIPP"]
                )
                calfac = calpwr / (cal_est - noise_est)
                mode_dict["LP"]["datalags"]["ACF"][itn, 0] = data_dict["LP"]["sig"]
                mode_dict["LP"]["datalags"]["Pulses"][itn, 0] = data_dict["LP"]["NIPP"]
                mode_dict["LP"]["datalags"]["CalFactor"][itn, 0] = calfac

                mode_dict["LP"]["noiselags"]["ACF"][itn, 0] = data_dict["LP"]["noise"]
                mode_dict["LP"]["noiselags"]["Pulses"][itn, 0] = data_dict["LP"]["NIPP"]

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
        timevec = self.simparams["Timevec"] + self.timeoffset
        inttime = self.simparams["Tint"]
        ds_fac = np.prod(self.simparams["declist"])
        f_s = float(self.simparams["fsnum"]) / self.simparams["fsden"] / ds_fac

        # Get array sizes
        samp_range = self.simparams["datasamples"]
        d_samps = (samp_range[1] - samp_range[0]) / ds_fac
        noise_range = self.simparams["noisesamples"]
        n_samps = (noise_range[1] - noise_range[0]) / ds_fac
        cal_range = self.simparams["calsamples"]

        range_gates = self.simparams["Rangegates"]
        N_rg = len(range_gates)  # take the size
        pulse = self.simparams["Pulse"][::ds_fac]
        pulselen = len(pulse)

        simdtype = self.simparams["dtype"]
        Ntime = len(timevec)

        lagtype = self.simparams["lagtype"]
        if "outangles" in self.simparams.keys():
            n_beams = len(self.simparams["outangles"])
            inttime = inttime
        else:
            n_beams = len(self.simparams["angles"])

        # Choose type of processing
        if self.simparams["Pulsetype"].lower() == "barker":
            lagfunc = BarkerLag
            Nlag = 1
        else:
            lagfunc = CenteredLagProduct
            Nlag = pulselen
        # initialize output arrays
        outdata = np.zeros((Ntime, n_beams, d_samps - Nlag + 1, Nlag), dtype=simdtype)
        outaddednoise = np.zeros(
            (Ntime, n_beams, d_samps - Nlag + 1, Nlag), dtype=simdtype
        )
        outnoise = np.zeros((Ntime, n_beams, n_samps - Nlag + 1, Nlag), dtype=simdtype)
        outcal = np.zeros((Ntime, n_beams), dtype=simdtype)
        pulses = np.zeros((Ntime, n_beams))
        pulsesN = np.zeros((Ntime, n_beams))
        timemat = np.zeros((Ntime, 2))
        # set up arrays that hold the location of pulses that are to be processed together
        infoname = self.datadir / "INFO.h5"
        # Just going to assume that the info file is in the directory
        infodict = load_dict_from_hdf5(str(infoname))
        flist = infodict["Files"]
        file_list = [str(self.datadir / i) for i in flist]
        pulsen_list = infodict["Pulses"]
        beamn_list = infodict["Beams"]
        time_list = infodict["Time"]
        file_loclist = [ifn * np.ones(len(ifl)) for ifn, ifl in enumerate(beamn_list)]

        pulsen = np.hstack(pulsen_list).astype(int)  # pulse number
        beamn = np.hstack(beamn_list).astype(int)  # beam numbers
        ptimevec = np.hstack(time_list).astype(float)  # time of each pulse
        file_loc = np.hstack(file_loclist).astype(int)  # location in the file

        # run the time loop
        print("Forming ACF estimates")

        # For each time go through and read only the necisary files
        for itn, iti in enumerate(timevec):
            update_progress(
                float(itn) / Ntime, "Time {0:d} of {1:d}".format(itn, Ntime)
            )
            # do the book keeping to determine locations of data within the files
            cur_tlim = (iti, iti + inttime)
            curcases = np.logical_and(ptimevec >= cur_tlim[0], ptimevec < cur_tlim[1])

            if not np.any(curcases):
                progstr = (
                    "No pulses for time {0:d} of {1:d}, lagdata adjusted accordinly"
                )
                update_progress(float(itn) / Ntime, progstr.format(itn, Ntime))

                outdata = outdata[:itn]
                outnoise = outnoise[:itn]
                pulses = pulses[:itn]
                pulsesN = pulsesN[:itn]
                timemat = timemat[:itn]
                continue
            pulseset = set(pulsen[curcases])
            poslist = [np.where(pulsen == item)[0] for item in pulseset]
            pos_all = np.hstack(poslist)

            try:
                pos_all = np.hstack(poslist)
                curfileloc = file_loc[pos_all]
            except:
                ipdb.set_trace()
            # Find the needed files and beam numbers
            curfiles = set(curfileloc)
            beamlocs = beamn[pos_all]
            timemat[itn, 0] = ptimevec[pos_all].min()
            timemat[itn, 1] = ptimevec[pos_all].max()
            # cur data pulls out all data from all of the beams and posisions
            curdata = np.zeros((len(pos_all), d_samps), dtype=simdtype)
            curaddednoise = np.zeros((len(pos_all), d_samps), dtype=simdtype)
            curnoise = np.zeros((len(pos_all), n_samps), dtype=simdtype)
            curcal = np.zeros((len(pos_all)), dtype=simdtype)
            # Open files and get required data
            # XXX come up with way to get open up new files not have to reread in data that is already in memory
            for ifn in curfiles:
                curfileit = [np.where(pulsen_list[ifn] == item)[0] for item in pulseset]
                curfileitvec = np.hstack(curfileit)
                ifile = file_list[ifn]
                curh5data = load_dict_from_hdf5(ifile)
                file_arlocs = np.where(curfileloc == ifn)[0]
                curdata[file_arlocs] = curh5data["RawData"][curfileitvec]

                curaddednoise[file_arlocs] = curh5data["AddedNoise"].astype(simdtype)[
                    curfileitvec
                ]
                # Read in noise data when you have don't have ACFs
                curnoise[file_arlocs] = curh5data["NoiseData"].astype(simdtype)[
                    curfileitvec
                ]
                curcal[file_arlocs] = curh5data["CalFactor"].astype(simdtype)[0]
            # differentiate between phased arrays and dish antennas
            if self.sensdict["Name"].lower() in ["risr", "pfisr", "risr-n"]:
                # After data is read in form lags for each beam
                for ibeam in range(n_beams):
                    progbeamstr = "Beam {0:d} of {1:d}".format(ibeam, n_beams)
                    update_progress(
                        float(itn) / Ntime + float(ibeam) / Ntime / n_beams, progbeamstr
                    )
                    beamlocstmp = np.where(beamlocs == ibeam)[0]
                    pulses[itn, ibeam] = len(beamlocstmp)

                    outdata[itn, ibeam] = lagfunc(
                        curdata[beamlocstmp].copy(),
                        numtype=simdtype,
                        pulse=pulse,
                        lagtype=lagtype,
                    )

                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outnoise[itn, ibeam] = lagfunc(
                        curnoise[beamlocstmp].copy(),
                        numtype=simdtype,
                        pulse=pulse,
                        lagtype=lagtype,
                    )

                    outaddednoise[itn, ibeam] = lagfunc(
                        curaddednoise[beamlocstmp].copy(),
                        numtype=simdtype,
                        pulse=pulse,
                        lagtype=lagtype,
                    )

                    outcal[itn, ibeam] = curcal[beamlocstmp]

            else:
                for ibeam, ibeamlist in enumerate(self.simparams["outangles"]):
                    progbeamstr = "Beam {0:d} of {1:d}".format(ibeam, n_beams)
                    update_progress(
                        float(itn) / Ntime + float(ibeam) / Ntime / n_beams, progbeamstr
                    )
                    beamlocstmp = np.where(np.in1d(beamlocs, ibeamlist))[0]
                    inputdata = curdata[beamlocstmp].copy()
                    noisedata = curnoise[beamlocstmp].copy()
                    noisedataadd = curaddednoise[beamlocstmp].copy()

                    pulses[itn, ibeam] = len(beamlocstmp)
                    pulsesN[itn, ibeam] = len(beamlocstmp)
                    outdata[itn, ibeam] = lagfunc(
                        inputdata, numtype=simdtype, pulse=pulse, lagtype=lagtype
                    )
                    outnoise[itn, ibeam] = lagfunc(
                        noisedata, numtype=simdtype, pulse=pulse, lagtype=lagtype
                    )

                    outaddednoise[itn, ibeam] = lagfunc(
                        noisedataadd, numtype=simdtype, pulse=pulse, lagtype=lagtype
                    )
                    outcal[itn, ibeam] = curcal[beamlocstmp].mean()
        # Create output dictionaries and output data
        data_lags = {
            "ACF": outdata,
            "Pow": outdata[:, :, :, 0].real,
            "Pulses": pulses,
            "Time": timemat,
            "AddedNoiseACF": outaddednoise,
            "CalFactor": outcal,
        }
        noise_lags = {
            "ACF": outnoise,
            "Pow": outnoise[:, :, :, 0].real,
            "Pulses": pulsesN,
            "Time": timemat,
        }
        return (data_lags, noise_lags)


def makedrf(
    name,
    start_global_index,
    dtypestr,
    is_complex,
    sample_rate_numerator,
    sample_rate_denominator,
    uuid,
    num_subchannels,
):

    dtype_strs = {
        "complexint": np.dtype([("r", "<i2"), ("i", "<i2")]),
        "complexlong": np.dtype([("r", "<i4"), ("i", "<i4")]),
        "complexlonglong": np.dtype([("r", "<i8"), ("i", "<i8")]),
        "complexfloat64": np.dtype("complex64"),
        "complexfloat128": np.dtype("complex128"),
    }

    dtype = dtype_strs.get(dtypestr, dtypestr)

    sub_cadence_secs = 3600  # Number of seconds of data in a subdirectory
    file_cadence_millisecs = 1000  # Each file will have up to 400 ms of data
    compression_level = 0  # no compression
    checksum = False  # no checksum

    is_continuous = True
    marching_periods = False  # no marching periods when writing
    drf_out = drf.DigitalRFWriter(
        name,
        dtype,
        sub_cadence_secs,
        file_cadence_millisecs,
        start_global_index,
        sample_rate_numerator,
        sample_rate_denominator,
        uuid,
        compression_level,
        checksum,
        is_complex,
        num_subchannels,
        is_continuous,
        marching_periods,
    )

    return drf_out
