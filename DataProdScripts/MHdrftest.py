#!/usr/bin/env python
"""

"""
import argparse
from datetime import datetime, timedelta
import dateutil.parser
import calendar

import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from SimISR.IonoContainer import IonoContainer
from SimISR.utilFunctions import getdefualtparams, getdefualtparams, makeconfigfile, readconfigfile
from SimISR.analysisplots import analysisdump
from SimISR.runsim import main as runsimisr
from SimISR import Path

def pyglowinput(latlonalt=None, dn_list=None, z=None, storm=False, spike=False):

    from pyglow.pyglow import Point
    if latlonalt is None:
        latlonalt = [42.61950, -71.4882, 250.00]
    if z is None:
        z = np.linspace(50., 1000., 200)
    if dn_list is None:
        dn_list = [datetime(2015, 3, 21, 8, 00), datetime(2015, 3, 21, 20, 00)]
    dn_diff = np.diff(dn_list)
    dn_diff_sec = dn_diff[-1].seconds
    timelist = np.array([calendar.timegm(i.timetuple()) for i in dn_list])
    time_arr = np.column_stack((timelist, np.roll(timelist, -1)))
    time_arr[-1, -1] = time_arr[-1, 0]+dn_diff_sec

    v = np.zeros((len(z), len(dn_list), 3))
    coords = np.column_stack((np.zeros((len(z), 2), dtype=z.dtype), z))
    all_spec = ['O+', 'NO+', 'O2+', 'H+'] #'HE+']
    Param_List = np.zeros((len(z), len(dn_list), len(all_spec)+1,2))

    # for velocities
    a = 150.
    b = 200.
    c = -200.
    d = -200.
    h_0 = 50
    for idn, dn in enumerate(dn_list):
        for iz, zcur in enumerate(z):
            latlonalt[2] = zcur
            pt = Point(dn, *latlonalt)
            pt.run_igrf()
            pt.run_msis()
            if storm:
                pt.run_iri(foF2_storm=True, foE_storm=True, hmF2_storm=True,
                           topside_storm=True)
            else:
                pt.run_iri()

            # so the zonal pt.u and meriodinal winds pt.v  will coorispond to x and y even though they are
            # supposed to be east west and north south. Pyglow does not seem to have
            # vertical winds.
            v0 = c* np.exp(-(zcur-h_0)/a)*np.sin(np.pi*(zcur-h_0)/b)+d
            v[iz, idn] = np.array([pt.u, pt.v, v0])

            for is1, ispec in enumerate(all_spec):
                Param_List[iz, idn, is1, 0] = pt.ni[ispec]*1e6

            Param_List[iz, idn, :-1, 1] = pt.Ti

            Param_List[iz, idn, -1, 0] = pt.ne*1e6
            Param_List[iz, idn, -1, 1] = pt.Te
    Param_sum = Param_List[:, :, :, 0].sum(0).sum(0)
    spec_keep = Param_sum > 0.
    v[np.isnan(v)] = 0.
    species = np.array(all_spec)[spec_keep[:-1]].tolist()
    species.append('e-')
    Param_List[:, :] = Param_List[:, :, spec_keep]
    if spike:
        ne_all = Param_List[:, :, -1, 0]
        maxarg = np.argmax(ne_all, axis=0)
        t_ar = np.arange(len(dn_list))
        Param_List[maxarg, t_ar, -1, 0] = 5.*ne_all[maxarg, t_ar]
    Iono_out = IonoContainer(coords, Param_List, times=time_arr, species=species, velocity=v)
    return Iono_out

def iriinput(latlonalt = [42.61950, -71.4882, 250.00], dn_start_stop =(datetime(2023, 1, 11, 8, 0), datetime(2023, 1, 11, 20, 0)),dn_diff= timedelta(minutes=10), altkmrange = [50,950,10]):
    """ """
    import iri2016.profile as iri

    dn_diff_sec = dn_diff.seconds

    altkmrange = [50,950,10]
    iri_data = iri.timeprofile(dn_start_stop,dn_diff,altkmrange,latlonalt[0],latlonalt[1])
    dn_list = pd.to_datetime(iri_data.time.to_series())
    timelist = np.array([calendar.timegm(i.timetuple()) for i in dn_list])
    time_arr = np.column_stack((timelist, np.roll(timelist, -1)))
    time_arr[-1, -1] = time_arr[-1, 0]+dn_diff_sec

    z = iri_data.alt_km.to_numpy()
    coords = np.column_stack((np.zeros((len(z), 2), dtype=z.dtype), z))

    all_spec = ['O+', 'NO+', 'O2+', 'H+', 'He+','e-']
    Param_List = np.zeros((len(z), len(dn_list),len(all_spec) ,2))
    v=[]

    for idn, dn in enumerate(dn_list):
        for iz, zcur in enumerate(z):
            latlonalt[2] = zcur

            # No winds from iri
            v.append([0, 0, 0])

            for is1, ispec in enumerate(all_spec[:-1]):
                Param_List[iz, idn, is1, 0] = iri_data.data_vars["n"+ispec][idn,iz]

            Param_List[iz, idn, :, 1] = iri_data.Ti[idn,iz]

            Param_List[iz, idn, -1, 0] = iri_data.ne[idn,iz]
            Param_List[iz, idn, -1, 1] = iri_data.Te[idn,iz]
    
    Param_sum = Param_List[:, :, :, 0].sum(0).sum(0)
    spec_keep = Param_sum > 0.
    species = np.array(all_spec)[spec_keep].tolist()
    Param_List[:, :] = Param_List[:, :, spec_keep]
    Iono_out = IonoContainer(coords, Param_List, times = time_arr, species=species)
    return Iono_out
def configfilesetup(testpath, config, simtime_mins=4):
    """ This will create the configureation file given the number of pulses for
        the test. This will make it so that there will be 12 integration periods
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses.
    """
    curloc = Path(__file__).resolve().parent
    defcon = curloc/config
    (sensdict, simparams) = readconfigfile(str(defcon))
    simparams['TimeLim'] = simtime_mins*60
    # tint = simparams['IPP']*npulses
    # ratio1 = tint/simparams['Tint']
    # simparams['Tint'] = ratio1*simparams['Tint']
    # simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    # simparams['TimeLim'] = 2*tint
    simparams['fitmode'] = 1
    simparams['startfile'] = 'startfile.h5'
    makeconfigfile(str(testpath/config), simparams['Beamlist'],
                   sensdict['Name'], simparams)


def plotiono(ionoin,fileprefix):

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    timevec = ionoin.Time_Vector

    for itn, itime in enumerate(timevec):
        t1 = itime[0]
        d_1 = datetime.utcfromtimestamp(t1)
        timestr = d_1.strftime("%Y-%m-%d %H:%M:%S")
        (figmplf, axmat) = plt.subplots(2, 2, figsize=(10, 10), facecolor='w', sharey=True)
        axvec = axmat.flatten()
        species = ionoin.Species
        zvec = ionoin.Cart_Coords[:, 2]
        params = ionoin.Param_List[:, itn]
        maxden = 10**np.ceil(np.log10(params[:, -1, 0].max()))
        for iplot, ispec in enumerate(species):
            axvec[0].plot(params[:, iplot, 0], zvec, label=ispec +'Density')


        axvec[1].plot(params[:, 0, 1], zvec, label='Ion Temperature')
        axvec[1].plot(params[:, -1, 1], zvec, label='Electron Temperature')
        axvec[0].set_title('Number Density')
        axvec[0].set_xscale('log')
        axvec[0].set_ylim([50, 800])
        axvec[0].set_xlim([maxden*1e-5, maxden])
        axvec[0].set_xlabel(r'Densities in m$^{-3}$')
        axvec[0].set_ylabel('Alt in km')
        axvec[0].legend()

        axvec[1].set_title('Temperature')
        axvec[1].set_xlim([100., 3500.])
        axvec[1].set_xlabel(r'Temp in $^{\circ}$ K')
        axvec[1].set_ylabel('Alt in km')
        axvec[1].legend()

        axvec[2].plot(ionoin.Velocity[:, itn, -1], zvec, label='Velocity')
        axvec[2].set_title('Velocity')
        axvec[2].set_xlim([-300., 300.])
        axvec[2].set_xlabel(r'Velocity m/s')
        axvec[2].set_ylabel('Alt in km')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        figmplf.suptitle('Parameters at Time: '+timestr, fontsize=12)

        figmplf.savefig(fileprefix+'_{0:0>3}.png'.format(itn), dpi=300)
        plt.close(figmplf)

def main(ARGS):
    """ This function will call other functions to create the input data, config
        file and run the radar data sim. The path for the simulation will be
        created in the Testdata directory in the SimISR module. The new
        folder will be called BasicTest. The simulation is a long pulse simulation
        will the desired number of pulses from the user.
        Inputs
            npulse - Number of pulses for the integration period, default==100.
            functlist - The list of functions for the SimISR to do.
    """
    curloc = Path(__file__).resolve().parent
    testpath = Path(ARGS.path)

    if not testpath.is_dir():
        testpath.mkdir(parents=True)
    functlist = ARGS.funclist
    functlist_default = ['spectrums', 'radardata', 'process', 'fitting']
    check_list = np.array([i in functlist for i in functlist_default])
    check_run = np.any(check_list)
    functlist_red = np.array(functlist_default)[check_list].tolist()

    configfilesetup(testpath, ARGS.config, ARGS.nminutes)
    config = str(testpath.joinpath(ARGS.config))

    inputpath = testpath.joinpath('Origparams')
    if ARGS.starttime is None:
        dtst0 = datetime(2015, 3, 21, 8, 00)
    else:
        dtst0 = dateutil.parser.parse(ARGS.starttime)

        print('Start time: %s' % (dtst0.strftime('%a %b %d %H:%M:%S %Y')))

    if ARGS.endtime is None:
        # default to the next 24 hours
        dtet0 = dtst0 + timedelta(1)
    else:
        dtet0 = dateutil.parser.parse(ARGS.endtime)
        print('End time: %s' % (dtet0.strftime('%a %b %d %H:%M:%S %Y')))

    #HACK May be should make this over the time period of the simulation?
    dtdiff = dtet0 - dtst0
    dn_secs = dtdiff.seconds/ARGS.ntimes
    dn_diff = timedelta(seconds=dn_secs)
    if not inputpath.is_dir():
        inputpath.mkdir()
    inputfile = inputpath.joinpath('0 stats.h5')
    if not inputfile.is_file() or ARGS.makeparams:
        dn_start_stop = (dtst0,dtet0)
        altkmrange = [50,1000,5]
        ionoout = iriinput(latlonalt = [42.61950, -71.4882, 250.00],dn_start_stop=dn_start_stop,dn_diff=dn_diff,altkmrange=altkmrange)
        #ionoout = pyglowinput(dn_list=dn_list, spike=ARGS.nespike)
        ionoout.saveh5(str(inputfile))
        ionoout.saveh5(str(testpath.joinpath('startfile.h5')))

    #make digitral rf directories
    drfdirone = drfdata = testpath/'drfdata'
    if drfdirone.exists() and 'radardata' in functlist_red:
        shutil.rmtree(str(drfdirone))
    if not drfdirone.exists():
        drfdata = testpath/'drfdata'/'rf_data'/'zenith-l'
        drfdata.mkdir(parents=True, exist_ok=True)
        drfdatatx = testpath/'drfdata'/'rf_data'/'tx-h'
        drfdatatx.mkdir(parents=True, exist_ok=True)
        dmddir = testpath/'drfdata'/'metadata'
        dmddir.mkdir(parents=True, exist_ok=True)

        acmdata = dmddir.joinpath('antenna_control_metadata')
        acmdata.mkdir(parents=True, exist_ok=True)

        iddir = dmddir.joinpath('id_metadata')
        iddir.mkdir(parents=True, exist_ok=True)

        pmdir = dmddir.joinpath('powermeter')
        pmdir.mkdir(parents=True, exist_ok=True)
    if check_run:
        runsimisr(functlist_red, str(testpath), config, True)

    if 'analysis' in functlist:
        analysisdump(str(testpath), config)
if __name__== '__main__':
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    PAR1 = argparse.ArgumentParser(description=descr)

    PAR1.add_argument("-p", "--path", help='Path.', type=str, default='')
    PAR1.add_argument("-c", "--config",
                      help='Name of config file in data prod scripts directory.',
                      type=str, default='MHsimple.yml')
    PAR1.add_argument('-t', '--ntimes', type=int, default=2,
                      help='''The number of iri data samples made in the time period.''')
    PAR1.add_argument('-j', "--nminutes", help='Number minutes of data created.',
                      type=int, default=4)
    PAR1.add_argument('-f', '--funclist', help='Functions to be used.', nargs='+',
                      default=['spectrums', 'radardata', 'process', 'fitting', 'analysis'])
    PAR1.add_argument('-s', '--starttime',  dest='starttime',
                      help='''Start time of IRI data created as datetime
                      (if in ISO8601 format: 2016-01-01T15:24:00Z)''')
    PAR1.add_argument('-e', '--endtime',  dest='endtime',
                      help='''End time of IRI data created as datetime (if in ISO8601 format
                      : 2016-01-01T15:24:00Z)''')
    PAR1.add_argument('-k', '--nespike', dest='nespike', action='store_true',
                      help='''Adds a spike in electron density at peak''',)
    PAR1.add_argument('-g', '--storms', dest='storms', action='store_true',
                      help="Storm time mode flag.")
    PAR1.add_argument('-m', '--makeparams', dest='makeparams', action='store_true',
                      help="Flag to remake the input parameters.")
    PAR1 = PAR1.parse_args()
    main(PAR1)
