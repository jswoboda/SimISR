#!/usr/bin/env python
"""
timingobj.py
This module classes and functions for timing.
@author: John Swoboda
"""
import yaml
from pathlib import Path
from fractions import Fraction
import numpy as np
import yamale
import digital_rf as drf

from SimISR import load_dict_from_hdf5

TSYS_CONV  = dict(fixed_zero=0.0, fixed_cooled=20.0, fixed_vlow=31.0, fixed_low=40.0,fixed_medium=50.0,fixed_high=70.0,amisr=120.0)


class Experiment(object):
    """Container class for the experiment information


    Attributes
    ----------
    name : str
        Name of experiment
    radartx : list
        Names of radar systems used for transmit.
    radartx : list
        Names of radar systems used for receive
    codes :
    code_order :
    code_repeats:
    tx_chans
    iline_chans
    plinechans
    save_directory
    """
    def __init__(self, experiment_name, radartx, radarrx, sequence, sequence_order, sequence_repeats, exp_time, channels, save_directory='tmp', exp_start=None, exp_end=None, radar_files=None,pulse_files=None):
        """

        Parameters
        ----------
        experiment_name,
        radartx,
        radarrx,
        sequence,
        sequence_order,
        sequence_repeats,
        exp_time,
        channels,
        save_directory
        exp_start=None
        exp_end,
        radar_files : str or list
            A string or list of strings of yaml files. Can also be a directory with yaml files in them.
        pulse_files=None

        """
        self.name = experiment_name
        rdr,sites=get_radars(radar_files)
        if isinstance(radartx,str):
            self.radartx = [radartx]
        elif isinstance(radartx,list):
            self.radartx = radartx

        if isinstance(radarrx,str):
            self.radarrx = [radarrx]
        elif isinstance(radarrx,list):
            self.radarrx = radarrx

        self.codes = sequence
        self.code_order = sequence_order
        self.code_repeats = sequence_repeats
        self.exp_time = exp_time
        self.tx_chans = {}
        self.iline_chans = {}
        self.pline_chans = {}
        self.save_directory= save_directory
        for isys,dlist in channels.items():
            for idict in dlist:
                curchan = Channel(**idict)
                if curchan.radardatatype== 'txpulse':
                    self.tx_chans[isys+'-'+curchan.name] = curchan
                elif curchan.radardatatype== 'ionline':
                     self.iline_chans[isys+'-'+curchan.name] = curchan
                elif curchan.radardatatype== 'plasmaline':
                     self.pline_chans[isys+'-'+curchan.name] = curchan



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
    radardatatype :
    uuid : str
        UUID string that will act as a unique identifier for the data and can be used to tie the data files to metadata. If None, a random UUID will be generated.
    num : sub_channels
        Number of subchannels in the data.
    drf_out : DigitalRFWriter
        Instance to write the digital RF data.
    """
    def __init__(self,name,sample_rate_numerator,sample_rate_denominator, is_complex, numtype, radardatatype, uuid, num_subchannels = 1):
        """ """
        self.name = name
        self.sr = Fraction(sample_rate_numerator,sample_rate_denominator)
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
        dtype_strs = {'complexint':np.dtype( [('r', '<i2'), ('i', '<i2')] ),
            "complexlong":np.dtype([('r', '<i4'), ('i', '<i4')]),
            "complexlonglong":np.dtype([('r', '<i8'), ('i', '<i8')]),
            "complexfloat64":np.dtype('complex64'),
            "complexfloat128":np.dtype('complex128'),
        }
        uuid= self.uuid
        outpath = Path(outdir).expanduser()
        drfname = outpath.joinpath(self.name)
        sr = self.sr
        dtype = dtype_strs.get(self.numtype,self.numtype)

        sub_cadence_secs = 3600  # Number of seconds of data in a subdirectory
        file_cadence_millisecs = 1000  # Each file will have up to 400 ms of
        compression_level = 0  # no compression
        checksum = False  # no checksum
        st_sample = drf.util.parse_identifier_to_sample(start_time,int(sr))
        is_continuous = True
        marching_periods = False  # no marching periods when writing

        drf_out = drf.DigitalRFWriter(drfname, dtype, sub_cadence_secs,
            file_cadence_millisecs, st_sample,
            sr.numerator, sr.denominator,
            uuid, compression_level, checksum,
            self.is_complex, self.num_subchannels,
            is_continuous, marching_periods)

        self.drf_out = drf_out
class RadarSystem(object):
    """

    """
    def __init__(self, ant_type, az_rotation, el_tilt, steering_mask, freq, tx_gain, rx_gain, tx_power,duty, tsys_type, cal_temp = 1689.21, xtra_tsys=0., kmat_file='', notes=''):

        self.ant_type=ant_type
        self.az_rotation = az_rotation
        self.el_tilt = el_tilt
        self.steering_mask = steering_mask
        self.freq = freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.tx_power = tx_power
        self.duty = duty
        self.tsys_type = tsys_type
        self.xtra_tsys = xtra_tsys
        self.kmat_file = kmat_file
        self.notes = notes
        self.k_dict,self.kmat = self.read_kmat()


    def get_tsys(self, more_tsys = 0.):
        tout = TSYS_CONV[self.tsys_type] + self.xtra_tsys
        return tout+more_tsys
    def read_kmat(self):
        ppath = Path('.')
        filepath = Path(self.kmat_file)
        if filepath == Path(''):
            raise ValueError(f"kmat file is not listed in the input.")
        elif filepath.exists():
            param_dict = load_dict_from_hdf5(self.kmat_file)
        elif filepath.parent == ppath:
            modpath = Path(__file__).parent.parent
            sensor_folder = modpath.joinpath("config",'sensor_info')
            newkfile = sensor_folder.joinpath(self.kmat_file)
            param_dict = load_dict_from_hdf5(str(newkfile))

        else:
            raise ValueError(f"Could not find kmat file{str(filepath)}")
        kmat = param_dict['Params']['Kmat']
        bcodes = kmat[:,0]
        az = kmat[:,1]
        el = kmat[:,2]
        ksys = kmat[:,3]
        k_dict = {}
        for ib,ia,iel,ik in zip(bcodes,az,el,ksys):
            tempd = dict(az=ia,el=iel,ksys=ik)
            k_dict[int(ib)] = tempd
        return k_dict, kmat
    def get_closest(self,az,el):
        """
        """
        assert el>0, "Elevation must be in degrees > 0"
        az_a = self.kmat[:,1]
        el_a = self.kmat[:,2]

        az = az % 360

        min_ind = np.argmin((az_a-az)**2 + (el_a-el)**2)
        return int(self.kmat[min_ind,0])

    def get_angle_info(self, beamcodes):
        """
        """
        az_out =[]
        el_out = []
        k_out = []

        for ibeam in beamcodes:
            c_d = self.k_dict[ibeam]
            az_out.append(c_d['az'])
            el_out.append(c_d['el'])
            k_out.append(c_d['ksys'])

        az  = np.array(az_out)
        el = np.array(el_out)
        ksys = np.array(k_out)

        return az,el,ksys

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
    schema_dict = {"experiment":"experiment_schema.yaml","pulse": "pulse_schema.yaml", "radar": "radar_schema.yaml"}
    dirname = Path(__file__).expanduser().parent
    schemadir = dirname.joinpath("schema")
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
    if files is None:
        modpath = Path(__file__).expanduser().parent.parent
        files = str(modpath.joinpath("config","sensor_info"))

    if isinstance(files, str) or isinstance(files, Path):
        files = [files]

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
    """
    def __init__(self,name,id_code,pulsecodes,beamcodes,pulsefolders=[]):
        """Creates the sequence object.

        Parameters
        ----------
        name : str
            Name of the pulse.
        id_code : int
            Code for sequency type.
        pulsecodes : list
            Sequence of pulse codes.
        beamcodes : list
            Sequency of beamcodes.
        pulsefolders : list
            List of folders that have pulse yamls. Can overload with different pulse files.
        """
        self.name = name
        self.id_code = id_code
        if isinstance(pulsefolders,str):
            pulsefolders = [pulsefolders]
        pulsefolders = [Path(i) for i in pulsefolders]

        mod_path = Path(__file__).expanduser().parent.parent
        ppath = mod_path.joinpath('config','pulse_files')
        pulsefolders.insert(0,ppath)

        pdict_all = {}
        for ifold in pulsefolders:
            # Use both the yml or yaml extention
            plist = list(ifold.glob("*.y*ml"))

            for ifile in plist:
#                p_dict = read_from_yamle(str(ifile),str(schemafile))
                p_dict = read_config_yaml(str(ifile),"pulse")
                cur_pulse = PulseTime(**p_dict)
                pdict_all[cur_pulse.code] = cur_pulse
        self.pulseseq = {icode:pdict_all[icode] for icode in pulsecodes}
        if not hasattr(beamcodes,'__len__'):
            beamcodes = [beamcodes]
        if len(beamcodes)==1:
            beamcodes = beamcodes*len(self.pulseseq)
        self.beamcodes = beamcodes

    def get_pulse_codes(self):
        """Outputs the pulse codes in a list.
        """
        pcodes = list(self.pulseseq.keys())
        return pcodes

    def outforyaml(self):
        """Outputs a dictionary that should be easy to write as part of a yaml file.

        Returns
        -------
        outyaml : dict
            Dictionary that can be saved to yaml easily.

        """
        outyaml = self.__dict__
        outyaml['pulsecodes'] = self.get_pulse_codes()
        del outyaml["pulseseq"]
        return outyaml

    def get_pulse_iq(self,sr,interpmethod="repeat"):
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

class PulseTime(object):
    """Holds information for each pulse mainly timing, in nanoseconds. """
    def __init__(self, code, name, baudlen, nbauds, bauds, raster, pulsetype='alternatingcode'):
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
        """
        self.code = code
        self.name = name
        self.baudlen = baudlen
        self.nbauds = nbauds
        self.raster = raster
        self.pulsetype=pulsetype

        self.pulse_iq = np.empty(nbauds,dtype=np.complex64)

        baud_keys = list(bauds.keys())
        if 'real' in baud_keys:
            idata = np.array(bauds['real']).astype(np.float32)
            qdata = np.array(bauds['imag']).astype(np.float32)
            self.pulse_iq = idata + 1j*qdata
        elif 'phase' in baud_keys:
            ph = np.exp(2j*np.pi*bauds['phases'])
            if not 'amp' in baud_keys:
                bauds['amp'] = np.ones(nbauds)
            self.pulse_iq = bauds['amp']*ph
        elif 'pols' in baud_keys:
            tmp_dict = {'+':1,'-':-1}
            ph = np.array([np.exp(2j*np.pi*tmp_dict[i]) for i in bauds['pols']])
            if not 'amp' in baud_keys:
                bauds['amp'] = np.ones(nbauds)
            self.pulse_iq = bauds['amp']*ph

    def get_raster(self):
        """Outputs the raster dictionary

        Returns
        -------
        raster : dict
            Dictionary with format of different parts of the pulse.
        """
        return self.raster

    def get_iq(self,sr,interpmethod="repeat"):
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
        fns = Fraction(1000000000,1)
        tb = np.arange(self.nbauds)*self.baudlen

        orig_iq = self.pulse_iq

        samp_periodns = fns/sr
        plenns = self.nbauds*self.baudlen
        nbaudsus = plenns/samp_periodns
        ts = np.arange(int(nbaudsus))*int(samp_periodns)

        if interpmethod=='repeat':
            Xold,Ynew = np.meshgrid(tb,ts)
            Z = Xold-Ynew
            zterm = np.logical_and(Z<=0,Z>-self.baudlen)
            tin,tout = np.where(zterm)
            outiq = orig_iq[tout]
        elif 'linear':
            outiq = np.interp(ts,tb,orig_iq)
        else:
            raise ValueError("interpmethod can only be linear or repeat.")

        return outiq

    def write_yaml(self,folder):
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
        yamldict['bauds'] = bauds_dict
        del yamldict['pulse_iq']

        outpath=Path(folder)
        fname = outpath.joinpath(self.name+'.yaml')

        if fname.exists():
            fname.unlink()
        with open(str(fname),'w') as outfile:
            yaml.dump(yamldict,outfile,default_flow_style=False, sort_keys=False)
