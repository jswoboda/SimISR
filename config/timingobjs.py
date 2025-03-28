#!/usr/bin/env python
"""
timingobj.py
This module classes and functions for timing.
@author: John Swoboda
"""
import yamale
from pathlib import Path
import numpy as np
import yaml
from fractions import Fraction


class TimingSweep:
    """Timing infomation for a sweep object.

        For each sweep this holds the raw IQ in a list where each object is the
        IQ for each sampling for creating data and each is a numpy array.
        Attributes:
            pulse_iq (list): Description of `attr1`.
            raster (dict): Description of `attr2`.

    """
    def __init__(self, pulse_iq, raster, name):
        """ """
        self.pulse_iq = pulse_iq
        self.raster = raster
        self.name = name
    def get_iq(self, freq_chan):

        return self.pulse_iq[freq_chan]
    def get_nchans(self):
        return len(self.pulse_iq)


def timingsweep_dumper(dumper, tsweep):
    """Writing yaml with dump command for TimingSweep.

    """
    value = []
    value.append((u'name', tsweep.name))
    #namestr = u'name: {0}'.format(tsweep.name)
    pulse_iq = tsweep.pulse_iq
    iqreal = [i.real.tolist() for i in pulse_iq]
    iqimag = [i.imag.tolist() for i in pulse_iq]
    value.append((u'real',iqreal))
    value.append((u'imag', iqimag))
    value.append((u'raster',tsweep.raster))
    return dumper.represent_mapping('!timingsweep', value)
    # iq_str = u'real: {0}\nimag: {1}'.format(iqreal, iqimag)
    # rasterstr = u'raster: {0}'.format(tsweep.raster)
    # str_all = '\n'.join([namestr, iq_str, rasterstr])
    #
    #
    # return dumper.represent_scalar(u'!timingsweep', str_all)
def timingsweep_builder(loader, node):
    """Reading yaml with load command for TimingSweep.

    """
    values = loader.construct_mapping(node, deep=True)
    iqlist = [np.array(i)+1j*np.array(j) for i, j in zip(values['real'], values['imag'])]
    raster = values['raster']
    name = values['name']
    return TimingSweep(iqlist, raster, name)

class TimingMode:
    def __init__(self, orig_bws, cen_freqs, final_bw):
        """

        """
        self.sweep_dict = {}
        self.sweep_order = []
        self.bw_list = orig_bws
        self.cen_freqs = cen_freqs
        self.final_bw = final_bw
    def add_raster(self, name, sweep):
        self.sweep_dict[name] = sweep
    def set_order(self, order):
        self.sweep_order = order
    def get_pulse_list(self, freq_chan):

        out_list = [self.sweep_dict[i_sweep].pulse_iq[freq_chan] for i_sweep in self.sweep_order]
        return out_list
    def get_raster_list(self):

        outlist = []
        for i_sweep in self.sweep_order:
            outlist.append(self.sweep_dict[i_sweep].raster)
    def check_sweeps(self):
        first_time = True
        for ikey, i_sweep in self.sweep_dict.items():
            if first_time:
                nchans = i_sweep.get_nchans()
                pulse_iq = [i_sweep.get_iq(i) for i in range(nchans)]
                first_time = False
            else:
                if nchans != i_sweep.get_nchans():
                    return True
                cur_iq = [i_sweep.get_iq(i) for i in range(nchans)]
                for i in range(nchans):
                    if len(cur_iq[i]) != len(pulse_iq[i]):
                        return True
        return False


def timingmode_dumper(dumper, tmode,):
    """Writing yaml with dump command for TimingMode.
    """
    value = [(i, tmode.__dict__[i]) for i in tmode.__dict__]
    return dumper.represent_mapping('!timingmode', value)
def timingmode_builder(loader, node):
    """Reading yaml with load command for TimingMode.

    """

    values = loader.construct_mapping(node, deep=True)
    tmout = TimingMode(values['bw_list'], values['cen_freqs'], values['final_bw'])

    for ikey in values['sweep_dict']:
        tmout.add_raster(ikey, values['sweep_dict'][ikey])
    tmout.set_order(values['sweep_order'])
    return tmout

def write_config_file(tm1, filename):
    """Write config file function outside sees.
    """
    yaml.add_representer(TimingSweep, timingsweep_dumper)
    yaml.add_representer(TimingMode, timingmode_dumper)
    with open(filename, 'w') as ymlfile:
        yaml.dump(tm1, ymlfile)


def read_config_file(filename):
    """Read config file that outside sees.
    """
    yaml.add_constructor(u'!timingsweep', timingsweep_builder)
    yaml.add_constructor(u'!timingmode', timingmode_builder)

    with open(filename, 'r') as ymlfile:
        tm1 = yaml.full_load(ymlfile)
    return tm1





def read_from_yamle(inputfile,schemafile=None):

    dirname = Path(__file__).expanduser().parent
    if schemafile is None:
        schemafile = dirname / "pulse_schema.yaml"
    schema = yamale.make_schema(schemafile)
    data = yamale.make_data(inputfile)
    _ = yamale.validate(schema, data)

    return data[0][0]

def write_to_yamle(timingsw, outdir='.'):

    outpath = Path(outdir)
    if not outpath.exists():
        outpath.mkdir()

    sweep_dict = timingsw.sweep_dict

    for icode,isweep in sweep_dict.items():
        d1 = dict()

        d1['code'] = icode
        d1['name'] = isweep.name
        d1['baudlen'] = 1000

        # assume the single IQ term
        iqdata = isweep.pulse_iq[0]
        ch0 = []
        ch1 = []
        for inum in iqdata:
            ch0.append(int(inum.real))
            ch1.append(int(inum.imag))

        iqobj = {'real':ch0,'imag':ch1}

        d1['nbauds'] = len(ch0)
        raster = isweep.raster

        newrast = dict()
        for iras,ilims in raster.items():
            newrast[iras] = [ilims[0]*1000,ilims[1]*1000]

        d1['bauds'] = iqobj
        d1['raster'] = newrast
        fname = outpath.joinpath(isweep.name+'.yaml')
        if fname.exists():
            fname.unlink()

        print(f"Writing: {str(fname)}")
        with open(str(fname),'w') as outfile:
            yaml.dump(d1,outfile,default_flow_style=False, sort_keys=False)
