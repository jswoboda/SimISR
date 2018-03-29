#!/usr/bin/env python
"""
Created on Fri Apr 15 15:00:12 2016
This script will be used to see if any changes will prevent the simulator from running.
@author: John Swoboda
"""
from SimISR import Path
import scipy as sp
from SimISR.utilFunctions import readconfigfile,makeconfigfile
from SimISR.IonoContainer import IonoContainer
from  SimISR.runsim import main as runsim
from SimISR.analysisplots import analysisdump


def configfilesetup(testpath, npulses, radar='PFISR'):
    """ This will create the configureation file given the number of pulses for
        the test. This will make it so that there will be 12 integration periods
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses.
    """
    testpath = Path(testpath).expanduser()
    curloc = Path(__file__).resolve().parent
    if radar.lower() == 'pfisr':
        defcon = curloc/'statsbase.ini'
    elif radar.lower() == 'millstonez':
        defcon = curloc/'statsbasemhz.yml'
    (sensdict, simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint'] = ratio1*simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = 2*tint
    simparams['fitmode'] = 1
    simparams['startfile'] = 'startfile.h5'
    makeconfigfile(str(testpath/'stats.yml'), simparams['Beamlist'],
sensdict['Name'], simparams)


def configfilesetup(testpath,npulses):
    """ This will create the configureation file given the number of pulses for
        the test. This will make it so that there will be 12 integration periods
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses.
    """
    testpath=Path(testpath).expanduser()
    curloc = Path(__file__).resolve().parent
    defcon = curloc/'statsbase.ini'

    (sensdict, simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint'] = ratio1*simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = 12*tint

    simparams['startfile']='startfile.h5'
    makeconfigfile(str(testpath/'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)

def makedata(testpath,tint):
    """ This will make the input data for the test case. The data will have cases
        where there will be enhancements in Ne, Ti and Te in one location. Each
        case will have 3 integration periods. The first 3 integration periods will
        be the default set of parameters Ne=Ne=1e11 and Te=Ti=2000.
        Inputs
            testpath - Directory that will hold the data.
            tint - The integration time in seconds.
    """
    testpath=Path(testpath).expanduser()
    finalpath = testpath.joinpath('Origparams')
    if not finalpath.is_dir():
        finalpath.mkdir()
    data = sp.array([[1e11,1100.],[1e11,2100.]])
    z = (50.+sp.arange(50)*10.)
    nz = len(z)
    params = sp.tile(data[sp.newaxis,sp.newaxis],(nz,1,1,1))
    epnt = range(20,22)
    p2 = sp.tile(params,(1,4,1,1))
    #enhancement in Ne
    p2[epnt,1,:,0]=5e11
    #enhancement in Ti
    p2[epnt,2,0,1]=2200.
    #enhancement in Te
    p2[epnt,3,1,1]=4200.
    coords = sp.column_stack((sp.ones(nz),sp.ones(nz),z))
    species=['O+','e-']
    times = sp.array([[0,1e3]])
    times2 = sp.column_stack((sp.arange(0,4),sp.arange(1,5)))*3*tint
    vel = sp.zeros((nz,1,3))
    vel2 = sp.zeros((nz,4,3))
    Icontstart = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
    Icont1 = IonoContainer(coordlist=coords,paramlist=p2,times = times2,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel2)

    finalfile = finalpath.joinpath('0 stats.h5')
    Icont1.saveh5(str(finalfile))
    Icontstart.saveh5(str(testpath.joinpath('startfile.h5')))


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

    PAR1.add_argument("-p", "--npulses", help='Number of pulses.', type=int, default=100)
    PAR1.add_argument('-f', '--funclist', help='Functions to be uses', nargs='+',
                      default=['spectrums', 'radardata', 'fitting', 'analysis'])
    PAR1 = PAR1.parse_args()
    main(PAR1.npulses, PAR1.funclist)
