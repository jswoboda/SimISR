#!/usr/bin/env python
"""
radarobjs.py
This module contains classes and functions for setting up and running the experiment.
@author: John Swoboda
"""
import warnings
import yaml
from copy import copy
import shutil
from pathlib import Path
from fractions import Fraction
import numpy as np
from .antennapatterncalc import antpatternplugs
import yamale
import digital_rf as drf
import scipy.constants as sc
from .h5fileIO import load_dict_from_hdf5, save_dict_to_hdf5

TSYS_CONV = dict(
    fixed_zero=0.0,
    fixed_cooled=20.0,
    fixed_vlow=31.0,
    fixed_low=40.0,
    fixed_medium=50.0,
    fixed_high=70.0,
    amisr=120.0,
)


class Experiment(object):
    """Container class for the experiment information


    Attributes
    ----------
    name : str
        Name of experiment
    codes : dict
        A dictionary with keys being the radar name string and values being lists of the sequence objects.
    code_order : list
        The order of the codes for each of the sequence.
    tx_chans : d dict
        Keys are the name of the channel and the values are the radar system object.
    iline_chans : dict
        Keys are the name of the channel and the values are the radar system object.
    plinechans  dict
        Keys are the name of the channel and the values are the radar system object.
    save_directory : str
        Overall directory all of the data will be saved.
    """

    def __init__(
        self,
        experiment_name,
        sequence,
        sequence_order,
        exp_time,
        channels,
        save_directory="tmp",
        exp_start=None,
        exp_end=None,
        radar_files=[],
        pulse_files=[],
    ):
        """

        Parameters
        ----------
        experiment_name : str
            Name of experiment
        sequence : list
            List of dictionaries holding information to make a sequence objects.
        sequence_order : list
            List giving the order of the sequences that the experiment will repeat until finished.
        exp_time : float
            Length of the experiment.
        channels : list
            List of dictionaries holding channel info.
        save_directory : str
            Overall directory all of the data will be saved.
        exp_start :
            Start time of the folder as datetime (if in ISO8601 format: 2016-01-01T15:24:00Z) or Unix time (if float/int).
        exp_end :
            End time of the folder as datetime (if in ISO8601 format: 2016-01-01T15:24:00Z) or Unix time (if float/int).
        radar_files : str or list
            A string or list of strings of yaml files. Can also be a directory with yaml files in them.
        pulse_files :  str or list
            A string or list of strings of yaml files. Can also be a directory with yaml files in them.

        """
        self.name = experiment_name
        rdr, sites = get_radars(radar_files)

        code_set = {}
        seqcp = copy(sequence)
        radarlist = []
        for iseq in seqcp:
            iseq["pulsefolders"] = pulse_files
            pseq = PulseSequence(**iseq)
            code_set[iseq["id_code"]] = pseq
            radarlist += pseq.radarnames
        self.codes = code_set

        self.radarobjs = {}
        self.radar2chans = dict()
        basedict = dict(txpulse=[],ionline=[],plasmaline=[])
        for irdr in radarlist:
            self.radar2chans[irdr] = copy(basedict)
            rdrobj = RadarSystem(**rdr[irdr])
            stobj = RadarSite(**sites[rdrobj.site])
            self.radarobjs[irdr] = {"radar": rdrobj, "site": stobj}

        self.code_order = sequence_order
        self.exp_time = exp_time
        self.tx_chans = {}
        self.iline_chans = {}
        self.pline_chans = {}
        self.save_directory = save_directory

        for isys, dlist in channels.items():
            for idict in dlist:
                curchan = Channel(**idict)
                ckey = isys + "-" + curchan.name
                if curchan.radardatatype == "txpulse":
                    self.tx_chans[ckey] = curchan
                    self.radar2chans[isys]['txpulse'].append(ckey)
                elif curchan.radardatatype == "ionline":
                    self.iline_chans[ckey] = curchan
                    self.radar2chans[isys]['ionline'].append(ckey)
                elif curchan.radardatatype == "plasmaline":
                    self.pline_chans[ckey] = curchan
                    self.radar2chans[isys]['plasmaline'].append(ckey)

    def make_sequence(self,nseconds):
        """"""
        code_ord = self.code_order
        tot_list = []
        time_list = []
        rdr_combo_lists = {i:{'tx':[],'rx':[]} for i in self.radarobjs.keys()}

        npulse = 0
        # Run through the sequencies and get all of the timing
        for icode in code_ord:
            cur_code = self.codes[icode]
            tnano, time_vec = cur_code.get_pulse_timing()
            tot_list.append(tnano)
            time_list.append(time_vec)
            combos = np.arange(npulse,npulse+len(time_vec))
            npulse+=len(time_vec)
            for irdr,itx in zip(cur_code.radarnames,cur_code.txorrx):
                rdr_combo_lists[irdr][itx].append(combos)

        # Fill out a dictionary for the codes, tying the radar to the codes.
        rdr_combos = {}
        for irdr,idict in rdr_combo_lists.items():
            tmp = {'tx':np.arange(0),'rx':np.arange(0)}
            for ix, ilist in idict.items():
                if ilist:
                    tmp[ix] = np.concatenate(ilist)
            rdr_combos[irdr] = tmp

        time_vec = np.concatenate(time_list,axis=0)
        tot = sum(tot_list)
        num_repeats = nseconds//tot
        combo_vec = np.arange(len(time_vec))
        combo_all = np.tile(combo_vec,num_repeats)
        t_rep,rep_num = np.meshgrid(time_vec,np.arange(num_repeats))
        time_mat = t_rep+tot*rep_num
        # import ipdb
        # ipdb.set_trace()
        time_all = time_mat.flatten()
        return rdr_combos,combo_all,time_all

    def setup_channels(self, save_directory, start_time):
        """Perform the set up of the drf channels

        Parameters
        ----------
        save_directory : str
            Base directory for the data sets.
        start_time :
            Start time of the folder as datetime (if in ISO8601 format: 2016-01-01T15:24:00Z) or Unix time (if float/int). (default: start ASAP)
        """
        # Write Tx channels
        for ikey, itx in self.tx_chans.items():
            itx.makedrf(save_directory, start_time)

        for ikey, iil in self.iline_chans.items():
            iil.makedrf(save_directory, start_time)
        for ikey, ipl in self.pline_chans.items():
            ipl.makedrf(save_directory, start_time)

    def close_channels(self):
        for ikey, itx in self.tx_chans.items():
            itx.drf_out.close()
        for ikey, iil in self.iline_chans.items():
            iil.drf_out.close()
        for ikey, ipl in self.pline_chans.items():
            ipl.drf_out.close()


class Channel(object):
    """Holds the information for each channel includingthe writer instance for digital RF.

    Attributes
    ----------
    name : str
        Name of the channel
    sr : Fraction
        Output sample rate of the channel.
    numtype : str
        Description of output datatype.
    is_complex : bool
        Is the data complex
    radardatatype : str
        This labels the type of data, either transmit, ion-line receive or plasma line receive.
    uuid : str
        UUID string that will act as a unique identifier for the data and can be used to tie the data files to metadata. If None, a random UUID will be generated.
    num : sub_channels
        Number of subchannels in the data.
    drf_out : DigitalRFWriter
        Instance to write the digital RF data.
    """

    def __init__(
        self,
        name,
        sample_rate_numerator,
        sample_rate_denominator,
        is_complex,
        numtype,
        radardatatype,
        uuid,
        num_subchannels=1,
    ):
        """ """
        self.name = name
        self.sr = Fraction(sample_rate_numerator, sample_rate_denominator)
        self.numtype = numtype
        self.is_complex = is_complex
        self.radardatatype = radardatatype
        self.uuid = uuid
        self.num_subchannels = num_subchannels
        self.drf_out = None

    def makedrf(self, outdir, start_time):
        """Creates the digital rf dataset folders for the channel.

        Parameters
        ----------
        outdir : str
            Out directory where the overall data set will be stored.
        start_time :
             Start time of the folder as datetime (if in ISO8601 format: 2016-01-01T15:24:00Z) or Unix time (if float/int). (default: start ASAP)
        """
        dtype_strs = {
            "complexint": np.dtype([("r", "<i2"), ("i", "<i2")]),
            "complexlong": np.dtype([("r", "<i4"), ("i", "<i4")]),
            "complexlonglong": np.dtype([("r", "<i8"), ("i", "<i8")]),
            "complexfloat64": np.dtype("complex64"),
            "complexfloat128": np.dtype("complex128"),
        }
        uuid = self.uuid
        outpath = Path(outdir).expanduser()
        drfname = outpath.joinpath(self.name)
        sr = self.sr
        dtype = dtype_strs.get(self.numtype, self.numtype)

        sub_cadence_secs = 3600  # Number of seconds of data in a subdirectory
        file_cadence_millisecs = 1000  # Each file will have up to 400 ms of
        compression_level = 0  # no compression
        checksum = False  # no checksum
        st_sample = drf.util.parse_identifier_to_sample(start_time, int(sr))
        is_continuous = True
        marching_periods = False  # no marching periods when writing

        if drfname.exists():
            shutil.rmtree(str(drfname))
        drfname.mkdir(parents=True)
        drf_out = drf.DigitalRFWriter(
            str(drfname),
            dtype,
            sub_cadence_secs,
            file_cadence_millisecs,
            st_sample,
            sr.numerator,
            sr.denominator,
            uuid,
            compression_level,
            checksum,
            self.is_complex,
            self.num_subchannels,
            is_continuous,
            marching_periods,
        )

        self.drf_out = drf_out


class RadarSystem(object):
    """This container class holds information on the radar system including gain, power, noise levels and system constants

    Attributes
    ----------
    ant_type : str
        Type of antenna, will determine how beam pattern is estimated.
    az_rotation : float
        Physical rotation from due north in azimuth in degrees.
    el_tilt : float
        Physical rotation from ground in elevation in degrees.
    steering_mask : list
        The limits in azimuth and elevation in degrees.
    freq : float
        Center Frequency of the radar in Hz.
    tx_gain : float
        Tx gain of the system in dBi.
    rx_gain : float
        Tx gain of the system in dBi.
    tx_power : float
        Peak poower in Watts.
    duty : float
        Duty factor of the radar.
    tsys : float
        Noise power in K.
    antennaparmaeters : dict
        Parameters for the antenna
    kmat_file : str
        Location of the h5 file that holds the system constant information.
    notes : str
        Extra notes about the system.
    k_dict : dict
        Extra information on the radar system.
    kmat : ndarray
        Matrix holding the system constants, beamcodes and angles for the radar.
    cal_temp : float
        Calibration temperature in K. This will be used to normalize all of the data by dividing everything by kbTB.
    """

    def __init__(
        self,
        ant_type,
        az_rotation,
        el_tilt,
        steering_mask,
        freq,
        tx_gain,
        rx_gain,
        tx_power,
        duty,
        tsys_type,
        site,
        antennaparmaeters,
        loss=0.0,
        cal_temp=1689.21,
        xtra_tsys=0.0,
        kmat_file="",
        notes="",
    ):
        """Initializes radar system object.

        Parameters
        ----------
        ant_type : str
            Type of antenna, will determine how beam pattern is estimated.
        az_rotation : float
            Physical rotation from due north in azimuth in degrees.
        el_tilt : float
            Physical rotation from ground in elevation in degrees.
        steering_mask : list
            The limits in azimuth and elevation in degrees.
        freq : float
            Center Frequency of the radar in Hz.
        tx_gain : float
            Tx gain of the system in dBi.
        rx_gain : float
            Tx gain of the system in dBi.
        tx_power : float
            Peak poower in Watts.
        duty : float
            Duty factor of the radar.
        loss : float
            Loss in dB.
        tsys_type : str
            The type of tsys that will determine the base noise level.
        cal_temp : float
            Calibration temperature in K. This will be used to normalize all of the data by dividing everything by kbTB.
        xtra_tsys : float
            Additional noise power in K.
        kmat_file : str
            Location of the h5 file that holds the system constant information.
        notes : str
            Extra notes about the system.
        """
        self.ant_type = ant_type
        self.az_rotation = az_rotation
        self.el_tilt = el_tilt
        self.steering_mask = steering_mask
        self.freq = freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.tx_power = tx_power
        self.duty = duty
        self.tsys = self.get_tsys(tsys_type, xtra_tsys)
        self.tsys_type = tsys_type
        self.xtra_tsys = xtra_tsys
        self.kmat_file = kmat_file
        self.notes = notes
        self.loss = loss
        self.k_dict, self.kmat = self.read_kmat()
        self.cal_temp = cal_temp
        self.site = site
        ant_params = antennaparmaeters
        ant_params["freq"] = freq
        ant_params["az_rotation"] = az_rotation
        ant_params["el_tilt"] = el_tilt
        self.antenna_func = antpatternplugs[ant_type](**ant_params)

    def calc_ksys(self, losses_db=0):
        """Calculates the system constant.

        Parameters
        ----------
        losses_db : float
            Additional losses in the system in dB.

        Returns
        -------
        calc_ksys : float
            Calculated ksys term.

        """
        totlos = self.loss + losses_db
        losses = 10 ** (totlos / 10.0)
        lamb = sc.c / self.freq
        G = 10 ** (max(self.tx_gain, self.rx_gain) / 10.0)
        e_rad = sc.e**2.0 * sc.mu_0 / (4.0 * sc.pi * sc.m_e)
        calc_ksys = G * sc.c * e_rad**2 * lamb**2 / (8 * np.pi) / losses

        return calc_ksys

    def get_tsys(self, tsys_type, xtra_tsys):
        """

        Parameters
        ----------
        tsys_type : str
            Type of tsys key.
        xtra_tsys : float
            Extra system temperature in degrees.

        Returns
        -------
        tout : float
            Output tsys given the type and xtra tsys.
        """
        tout = TSYS_CONV[tsys_type] + xtra_tsys
        return tout

    def write_kmat(self, fname, losses_db=0):
        """Write out a kmat file that can be saved.

        Parameters
        ----------
        fname : str
            Name of the file to be saved.
        losses_db : float
            Extra system losses in dB to be added.
        """
        outdict = {"Params": {}}
        if self.ant_type == "circ":
            stmask = self.steering_mask
            az_v = np.arange(stmask[0], stmask[1], dtype=float)
            el_v = np.arange(stmask[2], stmask[3], dtype=float)
            az_m, el_m = np.meshgrid(az_v, el_v)
            az_all = az_m.flatten()
            el_all = el_m.flatten()
            if stmask[3] == 90.0:
                az_all = np.append(az_all, 0.0)
                el_all = np.append(el_all, 0.0)
            bco = np.arange(len(az_all))
            ksys = self.calc_ksys() * np.ones_like(az_all)
            kmat = np.column_stack((bco, az_all, el_all, ksys))

        outdict["Params"]["Kmat"] = kmat
        save_dict_to_hdf5(outdict, fname)

    def read_kmat(self):
        """Reads the kmat file for the beam pattern.


        Returns
        -------
        k_dict : dict
            Dictionary with keys of the beamcode and values angles and system constants.
        kmat : ndarray
            First column is beamcodes, second column is az angle, third is el angle, and last is system constant.
        """
        ppath = Path(".")
        filepath = Path(self.kmat_file)
        if filepath == Path(""):
            warnings.warn("the input is 0!")
            return None, None
        elif filepath.exists():
            param_dict = load_dict_from_hdf5(self.kmat_file)
        elif filepath.parent == ppath:
            modpath = Path(__file__).parent.parent
            sensor_folder = modpath.joinpath("config", "sensor_info")
            newkfile = sensor_folder.joinpath(self.kmat_file)
            param_dict = load_dict_from_hdf5(str(newkfile))

        else:
            raise ValueError(f"Could not find kmat file{str(filepath)}")
        kmat = param_dict["Params"]["Kmat"]
        bcodes = kmat[:, 0]
        az = kmat[:, 1]
        el = kmat[:, 2]
        ksys = kmat[:, 3]
        k_dict = {}
        for ib, ia, iel, ik in zip(bcodes, az, el, ksys):
            tempd = dict(az=ia, el=iel, ksys=ik)
            k_dict[int(ib)] = tempd
        return k_dict, kmat

    def get_closest(self, az, el):
        """ """
        assert el > 0, "Elevation must be in degrees > 0"
        assert ~(self.kmat is None), "kmat is set to None, make a kmat h5 file. "
        az_a = self.kmat[:, 1]
        el_a = self.kmat[:, 2]

        az = az % 360

        min_ind = np.argmin((az_a - az) ** 2 + (el_a - el) ** 2)
        return int(self.kmat[min_ind, 0])

    def get_angle_info(self, beamcodes):
        """Using the beamcode table lookup for the angle and ksys info.

        Parameters
        ----------
        beamcodes : list
            Beamcodes for the lookup.

        Returns
        -------
        az : ndarray
            Azimuth angle for beamcode in degrees.
        el : ndarray
            Elevation angle for the beamcode in degrees
        ksys:
            System constant for the beamcode.
        """


        assert ~(self.kmat is None), "kmat is set to None, make a kmat h5 file. "

        if not  hasattr(beamcodes,'__len__'):
            beamcodes = beamcodes[0]
        az_out = []
        el_out = []
        k_out = []

        for ibeam in beamcodes:
            c_d = self.k_dict[ibeam]
            az_out.append(c_d["az"])
            el_out.append(c_d["el"])
            k_out.append(c_d["ksys"])

        az = np.array(az_out)
        el = np.array(el_out)
        ksys = np.array(k_out)

        return az, el, ksys


class RadarSite(object):
    """Holds information on the radar site and gives out the location info.

    Attributes
    ----------
    latitude : float
        Site latitude in degrees
    longitude : float
        Site longitude in degrees
    altitude : float
        Site altitude in meters
    elevation_mask : float
        Lowest elevation angle in degrees
    description : str
        Site description
    notes : str
        Notes on the site.
    """

    def __init__(
        self, latitude, longitude, altitude, elevation_mask, description, notes
    ):
        """Initializes the radar site object.

        Parameters
        ----------
        latitude : float
            Site latitude in degrees
        longitude : float
            Site longitude in degrees
        altitude : float
            Site altitude in meters
        elevation_mask : float
            Lowest elevation angle in degrees
        description : str
            Site description
        notes : str
            Notes on the site.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.elevation_mask = elevation_mask
        self.description = description
        self.notes = notes

    def get_lla(self):
        """

        Returns
        -------
        lla : ndarray
            Numpy array with latitude longitude and altitude.
        """
        lla = [self.latitude, self.longitude, self.altitude]
        return np.array(lla)


def read_config_yaml(yamlfile, schematype):
    """Parse config files.

    The function parses the given file and returns a dictionary with the values.

    Note
    ----
    Sections should be named: siminfo and channels

    Parameters
    ----------
    yamlfile : str
        The name of the file to be read including path.

    Returns
    -------
    objs : dictionay
        Dictionary with name given by [Section] each of which contains.
    """

    # dirname = Path(__file__).expanduser().parent
    # schemafile = dirname / "configschema.yaml"
    schemafile = getschemafile(schematype)
    schema = yamale.make_schema(schemafile)
    data = yamale.make_data(yamlfile)
    _ = yamale.validate(schema, data)

    return data[0][0]


def getschemafile(schematype):
    """Return the schema file name.

    Parameters
    ----------
    schematype : str
        Type of schema file, either mapping or radar.

    Returns
    : str
        Name of the schema file.
    """
    schema_dict = {
        "experiment": "experiment_schema.yaml",
        "pulse": "pulse_schema.yaml",
        "radar": "radar_schema.yaml",
    }
    modpath = Path(__file__).expanduser().parent.parent
    schemadir = modpath.joinpath("config", "schema")
    return str(schemadir.joinpath(schema_dict[schematype]))


def get_radars(files=None):
    """Gets dictionaries of radars and sites with the key as the name and the value a dictionary of parameters.

    Parameters
    ----------
    files : str or list
        A string or list of strings of yaml files. Can also be a directory with yaml files in them.

    Returns
    -------
    radar_dict : dict
        Dictionary with keys as names of ISRs, values are sub dictionaries of parameters.
    site_dict : dict
        Dictionary with keys as names of ISR sites, values are sub dictionaries of parameters.

    """

    modpath = Path(__file__).expanduser().parent.parent
    files_base = str(modpath.joinpath("config", "sensor_info"))

    if files is None:
        files = files_base

    if isinstance(files, str) or isinstance(files, Path):
        files = [files_base, files]
    elif isinstance(files, list):
        files.insert(0, files_base)

    paths = []
    for ifile in files:
        fpath = Path(ifile)
        if fpath.is_dir():
            paths += list(fpath.glob("*.yaml"))
        elif fpath.is_file():
            paths.append(fpath)

    radar_dict = {}
    site_dict = {}
    for ipath in paths:
        file_dict = read_config_yaml(ipath, "radar")
        r_list = file_dict["radars"]
        for iradar in r_list:
            r_name = iradar["name"]
            del iradar["name"]
            radar_dict[r_name] = iradar

        s_list = file_dict.get("sites", [])
        for isite in s_list:
            s_name = isite["name"]
            del isite["name"]
            site_dict[s_name] = isite

    return radar_dict, site_dict


class PulseSequence(object):
    """Holds information on the pulse sequence.

    Attributes
    ----------
    name : str
        Name of the pulse.
    id_code : int
        Code for sequency type.
    txorrx : list
        Determines if it's tx or rx or both.
    radarnames : list
        Names of radars that will exist in the pulse sequence.
    pulseseq : dict
        Dictionary where keys are the pulse number and the values are the associated PulseTiming object.
    pulsecodes : list
        This list of list holds the pulse code order because the dictionary holds the unique code.
    beamcodes : list
        List of lists coresponding to beamcodes for each pulse.
    """

    def __init__(
        self,
        name,
        id_code,
        txrxname,
        pulsecodes,
        beamcodes,
        pulsefolders=[],
        txorrx=None,
    ):
        """Creates the sequence object.

        Parameters
        ----------
        name : str
            Name of the pulse sequenc.
        id_code : int
            Code for sequence.
        pulsecodes : list
            Sequence of pulse codes.
        beamcodes : list
            Sequency of beamcodes.
        pulsefolders : list
            List of folders that have pulse yaml files. Can overload with different pulse files.
        """
        self.name = name
        self.id_code = id_code

        if isinstance(txrxname, str):
            self.radarnames = [txrxname, txrxname]
        elif isinstance(txrxname, list):
            if len(txrxname) == 1:
                self.radarnames = [txrxname[0], txrxname[0]]
        self.radarnames = txrxname

        if (txorrx is None) and (len(self.radarnames) == 2):
            self.txorrx = ["tx", "rx"]
        elif len(txorrx) != len(self.radarnames):
            raise ValueError("txorrx length does not align with txrxname")
        else:
            self.txorrx = txorrx

        if isinstance(pulsefolders, str):
            pulsefolders = [pulsefolders]
        pulsefolders = [Path(i) for i in pulsefolders]

        mod_path = Path(__file__).expanduser().parent.parent
        ppath = mod_path.joinpath("config", "pulse_files")
        pulsefolders.insert(0, ppath)

        pdict_all = {}
        allcodes = []

        if isinstance(pulsecodes[0], list):
            allcodes = sum(pulsecodes, [])
            ucodes = list(set(allcodes))
            if len(pulsecodes[0]) == 1:
                mult = len(self.radarnames)
            else:
                mult = 1
            fullcodes = [iclist * mult for iclist in pulsecodes]
        elif isinstance(pulsecodes[0], int):
            ucodes = list(set(pulsecodes))
            fullcodes = [[icode] * len(self.radarnames) for icode in pulsecodes]

        self.pulsecodes = fullcodes

        for ifold in pulsefolders:
            # Use both the yml or yaml extention
            plist = list(ifold.glob("*.y*ml"))

            for ifile in plist:
                #                p_dict = read_from_yamle(str(ifile),str(schemafile))
                p_dict = read_config_yaml(str(ifile), "pulse")
                cur_pulse = PulseTime(**p_dict)
                pdict_all[cur_pulse.code] = cur_pulse
        self.pulseseq = {icode: pdict_all[icode] for icode in ucodes}

        if isinstance(beamcodes[0], list):
            if len(beamcodes[0]) == 1:
                mult = len(self.radarnames)
            else:
                mult = 1
            fullbeams = [iclist * mult for iclist in beamcodes]
        else:
            fullbeams = [[icode] * len(self.radarnames) for icode in beamcodes]
        self.beamcodes = fullbeams

    def get_pulse_codes(self):
        """Outputs the pulse codes in a list.

        Returns
        -------
        pcodes : list
            Pulse code numbers in the sequence.
        """
        pcodes = list(self.pulsecodes)
        return pcodes

    def outforyaml(self):
        """Outputs a dictionary that should be easy to write as part of a yaml file.

        Returns
        -------
        outyaml : dict
            Dictionary that can be saved to yaml easily.
        """
        outyaml = self.__dict__
        outyaml["pulsecodes"] = self.get_pulse_codes()
        del outyaml["pulseseq"]
        return outyaml

    def get_pulse_iq(self, sr, interpmethod="repeat"):
        """Get all of the pulses in a dictionary.

        Parameters
        ----------
        sr : Fraction
            Sampling rate as a rational number.
        interpmethod : str
            Method to interpolate the pulses.

        Returns
        -------
        outiq : dict
            Output numpy arrays in a dictionary with keys of pulsecode.
        beamcodes : list
            Beamcodes for spatial information.
        """

        outiq = {}
        for ipcode, iptobj in self.pulseseq.items():
            outiq[ipcode] = iptobj.get_iq(sr, interpmethod)

        return outiq, self.beamcodes


    def get_pulse_timing(self):
        """Get the full time of the sequence in ns.

        Returns
        -------
        n_nano : int
            The length of the sequence in nano seconds.
        time_vec : ndarray
            Holds the start time in ns for pulse in the sequence
        """
        totnano = 0
        codes = self.get_pulse_codes()
        time_vec = np.zeros(len(codes),dtype=np.int64)
        for icn, icode in enumerate(codes):
            # HACK
            # Assume for now that pulse times in tx and rx all the same!
            curcode = icode[0]
            rast = self.pulseseq[curcode].get_raster()
            flist = rast['full']
            n_nano = flist[-1]
            time_vec[icn] = totnano
            totnano+=n_nano

        return totnano,time_vec

class PulseTime(object):
    """Holds information for each pulse mainly timing, in nanoseconds.

    Attributes
    ----------
    code : int
        Pulse code for pulse type.
    name : str
        Name of the pulse.
    baudlen : int
        Length of each baud represented in the bauds dict in ns.
    nbauds : int
        Number of bauds for the pulse.
    bauds : dict
        Dictionary containing the info for the bauds.
    raster : dict
        Dictionary containing pulse timing in ns.
    pulsetype : str
        Name of the type of pulse.
    pulse_iq : ndarray
        The IQ of each baud.

    """

    def __init__(
        self, code, name, baudlen, nbauds, bauds, raster, pulsetype="alternatingcode"
    ):
        """Creates a PulseTime object

        Parameters
        ----------
        code : int
            Pulse code for pulse type.
        name : str
            Name of the pulse.
        baudlen : int
            Length of each baud represented in the bauds dict in ns.
        nbauds : int
            Number of bauds for the pulse.
        bauds : dict
            Dictionary containing the info for the bauds.
        raster : dict
            Dictionary containing pulse timing in ns.
        pulsetype : str
            Name of the type of pulse.
        """
        self.code = code
        self.name = name
        self.baudlen = baudlen
        self.nbauds = nbauds
        self.raster = raster
        self.pulsetype = pulsetype

        self.pulse_iq = np.empty(nbauds, dtype=np.complex64)

        baud_keys = list(bauds.keys())
        if "real" in baud_keys:
            idata = np.array(bauds["real"]).astype(np.float32)
            qdata = np.array(bauds["imag"]).astype(np.float32)
            self.pulse_iq = idata + 1j * qdata
        elif "phase" in baud_keys:
            ph = np.exp(2j * np.pi * bauds["phases"])
            if not "amp" in baud_keys:
                bauds["amp"] = np.ones(nbauds)
            self.pulse_iq = bauds["amp"] * ph
        elif "pols" in baud_keys:
            tmp_dict = {"+": 1, "-": -1}
            ph = np.array([np.exp(2j * np.pi * tmp_dict[i]) for i in bauds["pols"]])
            if not "amp" in baud_keys:
                bauds["amp"] = np.ones(nbauds)
            self.pulse_iq = bauds["amp"] * ph

    def get_raster(self):
        """Outputs the raster dictionary

        Returns
        -------
        raster : dict
            Dictionary with format of different parts of the pulse.
        """
        return self.raster

    def get_iq(self, sr, interpmethod="repeat"):
        """Outputs the IQ representation of each pulse

        Parameters
        ----------
        sr : Fraction
            Sampling rate as a rational number.
        interpmethod : str
            Method to interpolate the pulses.

        Returns
        -------
        outiq : np.ndarray
            Output pulse data in a numpy array.
        """
        fns = Fraction(1000000000, 1)
        tb = np.arange(self.nbauds) * self.baudlen

        orig_iq = self.pulse_iq

        samp_periodns = fns / sr
        plenns = self.nbauds * self.baudlen
        nbaudsus = plenns / samp_periodns
        ts = np.arange(int(nbaudsus)) * int(samp_periodns)

        if interpmethod == "repeat":
            Xold, Ynew = np.meshgrid(tb, ts)
            Z = Xold - Ynew
            zterm = np.logical_and(Z <= 0, Z > -self.baudlen)
            tin, tout = np.where(zterm)
            outiq = orig_iq[tout]
        elif "linear":
            outiq = np.interp(ts, tb, orig_iq)
        else:
            raise ValueError("interpmethod can only be linear or repeat.")

        return outiq

    def write_yaml(self, folder):
        """Write out object to a yaml file. The name of the file is based off of the name variable.

        Parameters
        ----------
        folder : str
            Folder that will save the file.

        """
        yamldict = self.__dict__

        real_bauds = [int(i) for i in self.pulse_iq.real.tolist()]
        imag_bauds = [int(i) for i in self.pulse_iq.imag.tolist()]
        bauds_dict = dict(real=real_bauds, imag=imag_bauds)
        yamldict["bauds"] = bauds_dict
        del yamldict["pulse_iq"]

        outpath = Path(folder)
        fname = outpath.joinpath(self.name + ".yaml")

        if fname.exists():
            fname.unlink()
        with open(str(fname), "w") as outfile:
            yaml.dump(yamldict, outfile, default_flow_style=False, sort_keys=False)
