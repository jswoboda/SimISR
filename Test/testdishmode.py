#!/usr/bin/env python
"""
Created on Mon Nov  9 13:13:02 2015

@author: John Swoboda
"""
from SimISR import Path
import sys
import scipy as sp
import matplotlib
matplotlib.use('Agg')
from SimISR.utilFunctions import makeconfigfile
from SimISR.IonoContainer import IonoContainer, MakeTestIonoclass
import SimISR.runsim as runsim
from SimISR.analysisplots import analysisdump

def configsetup(testpath):
    """This function will make a pickle file used to configure the simulation.
    Inputs
    testpath - A string for the path that this file will be saved."""
    # list of beams that will give a line of points between 70 and 80 deg el with az of 20 deg
    beamlist = [16626, 16987, 17348, 17709, 18070, 18431, 18792, 19153]#, 19514] # list of beams in
    radarname = 'millstone'# name of radar for parameters can either be pfisr or risr



#    pulse = sp.ones(14)# pulse
    rng_lims = [150,500]# limits of the range gates
    IPP = 1e-2#8.7e-3 #interpulse period in seconds
    NNs = 28 # number of noise samples per pulse
    NNp = 100 # number of noise pulses
    b_rate = 100
    intrate = 2.
    Tint=intrate*b_rate*IPP # integration time in seconds
    time_lim = len(beamlist)/intrate*4.0*Tint # simulation length in seconds
    fitter_int = Tint*len(beamlist)/intrate # time interval between fitted params
    simparams =   {'IPP':IPP, #interpulse period
                   'TimeLim':time_lim, # length of simulation
                   'RangeLims':rng_lims, # range swath limit
#                   'Pulse':pulse, # pulse shape
                   'Pulselength':280e-6,
                   'FitType' :'acf',
                   't_s': 20e-6,
                   'Pulsetype':'long', # type of pulse can be long or barker,
                   'Tint':Tint, #Integration time for each fitting
                   'Fitinter':fitter_int, # time interval between fitted params
                   'NNs': NNs,# number of noise samples per pulse
                   'NNp':NNp, # number of noise pulses
                   'dtype':sp.complex128, #type of numbers used for simulation
                   'ambupsamp':1, # up sampling factor for ambiguity function
                   'species':['O+','e-'], # type of ion species used in simulation
                   'numpoints':128, # number of points for each spectrum
                   'startfile': str(testpath/'startdata.h5'),# file used for starting points
                   'beamrate':b_rate,# the number of pulses each beam will output until it moves
                   'outangles':[sp.arange(i,i+intrate) for i in sp.arange(0,len(beamlist),intrate)]}
#                   'SUMRULE': sp.array([[-2,-3,-3,-4,-4,-5,-5,-6,-6,-7,-7,-8,-8,-9]
#                       ,[1,1,2,2,3,3,4,4,5,5,6,6,7,7]])}

    fname = testpath/'DishExample'

    makeconfigfile(fname.with_suffix('.ini'),beamlist,radarname,simparams)
def makeinputh5(Iono,basedir):
    """This will make a h5 file for the IonoContainer that can be used as starting
    points for the fitter. The ionocontainer taken will be average over the x and y dimensions
    of space to make an average value of the parameters for each altitude.
    Inputs
    Iono - An instance of the Ionocontainer class that will be averaged over so it can
    be used for fitter starting points.
    basdir - A string that holds the directory that the file will be saved to.
    """
    # Get the parameters from the original data
    Param_List = Iono.Param_List
    dataloc = Iono.Cart_Coords
    times = Iono.Time_Vector
    velocity = Iono.Velocity
    zlist,idx = sp.unique(dataloc[:,2],return_inverse=True)
    siz = list(Param_List.shape[1:])
    vsiz = list(velocity.shape[1:])

    datalocsave = sp.column_stack((sp.zeros_like(zlist),sp.zeros_like(zlist),zlist))
    outdata = sp.zeros([len(zlist)]+siz)
    outvel = sp.zeros([len(zlist)]+vsiz)
    #  Do the averaging across space
    for izn,iz in enumerate(zlist):
        arr = sp.argwhere(idx==izn)
        outdata[izn]=sp.mean(Param_List[arr],axis=0)
        outvel[izn]=sp.mean(velocity[arr],axis=0)

    Ionoout = IonoContainer(datalocsave,outdata,times,Iono.Sensor_loc,ver=0,
                            paramnames=Iono.Param_Names, species=Iono.Species,velocity=outvel)
    Ionoout.saveh5(basedir/'startdata.h5')

def main(funcnamelist):
    """This function will run the test simulation buy first making a simple set of
    ionospheric parameters based off of a Chapman function. Then it will create configuration
    and start files followed by running the simulation."""
    curpath = Path(__file__).parent
    testpath = curpath/'Testdata'/'DishMode'
    origparamsdir = testpath/'Origparams'

    testpath.mkdir(exist_ok=True,parents=True)
    
    origparamsdir.mkdir(exist_ok=True,parents=True)

    if 'configupdate' in funcnamelist:
        configsetup(testpath)
        funcnamelist.remove('configupdate')
    # clear everything out
    folddict = {'origdata':['Origparams'], 'spectrums':['Spectrums'], 'radardata':['ACF','Radardata'], 'fitting':['Fitted']}
    folderlist = []
    for i in funcnamelist:
        folderlist=folderlist+folddict[i]
#    folderlist = ['Origparams','Spectrums','Radardata','ACF','Fitted']
    for ifl in folderlist:
        flist = (testpath/ifl).glob('*.h5')
        for ifile in flist:
            ifile.unlink()
    # Now make stuff again


    if 'origdata' in funcnamelist:

        Icont1 = MakeTestIonoclass(testv=True,testtemp=True)
        makeinputh5(MakeTestIonoclass(testv=True,testtemp=False),testpath)
        Icont1.saveh5(origparamsdir/'0 testiono.h5')
        funcnamelist.remove('origdata')

#    funcnamelist=['spectrums','radardata','fitting']
    failflag=runsim.main(funcnamelist,testpath,testpath/'DishExample.ini',True)
    if not failflag:
        analysisdump(testpath,testpath/'DishExample.ini')
        
if __name__== '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Planeproc.py -f <function: configupdate, origdata, spectrums, radardata, fitting or all>')
    p.add_argument('-f','--funcnamelist',nargs='+')
    p.add_argument('-r','--re',action='store_true')
    p = p.parse_args()
    
    
    curpath = Path(__file__).parent

    remakealldata = p.re
    
    if p.funcnamelist is None or 'all' in p.funcnamelist:
        funcnamelist=['configupdate','origdata','spectrums','radardata','fitting']
        
    main(funcnamelist)
