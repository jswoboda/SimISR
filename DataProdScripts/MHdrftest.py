#!/usr/bin/env python
"""

"""

from datetime import datetime
import calendar

import scipy as sp
import shutil
from pyglow.pyglow import Point
from SimISR.IonoContainer import IonoContainer
from SimISR.utilFunctions import getdefualtparams, getdefualtparams, makeconfigfile
from SimISR.analysisplots import analysisdump
from SimISR.runsim import main as runsimisr
from SimISR import Path

def pyglowinput(latlonalt=[65.1367, -147.4472, 250.00], dn_list=[datetime(2015, 3, 21, 8, 00), datetime(2015, 3, 21, 20, 00)], z=None):


    if z is None:
        z = sp.linspace(50., 1000., 200)
    dn_diff = sp.diff(dn_list)
    dn_diff_sec = dn_diff[-1].seconds
    timelist = sp.array([calendar.timegm(i.timetuple()) for i in dn_list])
    time_arr = sp.column_stack((timelist, sp.roll(timelist, -1)))
    time_arr[-1, -1] = time_arr[-1, 0]+dn_diff_sec

    v=[]
    coords = sp.column_stack((sp.zeros((len(z), 2), dtype=z.dtype), z))
    all_spec = ['O+', 'NO+', 'O2+', 'H+', 'HE+']
    Param_List = sp.zeros((len(z), len(dn_list),len(all_spec),2))
    for idn, dn in enumerate(dn_list):
        for iz, zcur in enumerate(z):
            latlonalt[2] = zcur
            pt = Point(dn, *latlonalt)
            pt.run_igrf()
            pt.run_msis()
            pt.run_iri()

            # so the zonal pt.u and meriodinal winds pt.v  will coorispond to x and y even though they are
            # supposed to be east west and north south. Pyglow does not seem to have
            # vertical winds.
            v.append([pt.u, pt.v, 0])

            for is1, ispec in enumerate(all_spec):
                Param_List[iz, idn, is1, 0] = pt.ni[ispec]*1e6

            Param_List[iz, idn, :, 1] = pt.Ti

            Param_List[iz, idn, -1, 0] = pt.ne*1e6
            Param_List[iz, idn, -1, 1] = pt.Te
    Param_sum = Param_List[:, :, :, 0].sum(0).sum(0)
    spec_keep = Param_sum > 0.
    species = sp.array(all_spec)[spec_keep[:-1]].tolist()
    species.append('e-')
    Param_List[:, :] = Param_List[:, :, spec_keep]
    Iono_out = IonoContainer(coords, Param_List, times = time_arr, species=species)
    return Iono_out
def main(npulse = 100 ,functlist = ['spectrums','radardata','fitting','analysis']):
    """ This function will call other functions to create the input data, config
        file and run the radar data sim. The path for the simulation will be
        created in the Testdata directory in the SimISR module. The new
        folder will be called BasicTest. The simulation is a long pulse simulation
        will the desired number of pulses from the user.
        Inputs
            npulse - Number of pulses for the integration period, default==100.
            functlist - The list of functions for the SimISR to do.
    """
    curloc = Path(__file__).resolve()
    testpath = curloc.parent.parent/'Testdata'/'BasicTest'
    if not testpath.is_dir():
        testpath.mkdir(parents=True)

    functlist_default = ['spectrums', 'radardata', 'fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run = sp.any(check_list)
    functlist_red = sp.array(functlist_default)[check_list].tolist()

    config = testpath.joinpath('stats.ini')
    if not config.exists():
        configfilesetup(str(testpath), npulse)

    (sensdict, simparams) = readconfigfile(str(config))
    makedata(testpath, simparams['Tint'])
    if check_run:
        runsim(functlist_red, str(testpath), config, True)
    if 'analysis' in functlist:
        analysisdump(str(testpath), config)

if __name__== '__main__':
    from argparse import ArgumentParser
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    PAR1 = ArgumentParser(description=descr)

    PAR1.add_argument("-p", "--path", help='Path.', type=str, default='')
    PAR1.add_argument('-f', '--funclist', help='Functions to be uses', nargs='+',
                      default=['spectrums', 'radardata', 'fitting', 'analysis'])
    PAR1 = PAR1.parse_args()
    main(PAR1.npulses, PAR1.funclist)
