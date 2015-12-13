#!/usr/bin/env python
"""
Created on Tue May  5 17:16:51 2015

@author: John Swoboda
"""


import os, inspect, glob
import scipy as sp
from RadarDataSim.utilFunctions import makepicklefile, GenBarker
from RadarDataSim.IonoContainer import IonoContainer, MakeTestIonoclass
import RadarDataSim.runsim as runsim
from RadarDataSim.analysisplots import analysisdump

def makeconfigfile(testpath):
    beamlist = [64094,64091,64088,64085,64082,64238,64286,64070,64061,64058,64055,64052,
                64049,64046,64043,64067,64040,64037,64034]
    radarname = 'pfisr'

    Tint=4.0*60.0
    time_lim = 3.0*Tint
    pulse = GenBarker(7)
    rng_lims = [75.0,250.0]
    IPP = .0087
    NNs = 28
    NNp = 100
    simparams =   {'IPP':IPP,
                   'TimeLim':time_lim,
                   'RangeLims':rng_lims,
                   'Pulse':pulse,
                   'Pulsetype':'barker',
                   'Tint':Tint,
                   'Fitinter':Tint,
                   'NNs': NNs,
                   'NNp':NNp,
                   'dtype':sp.complex128,
                   'ambupsamp':30,
                   'species':['O+','e-'],
                   'numpoints':128,
                   'startfile':os.path.join(testpath,'startdata.h5')}

    fname = os.path.join(testpath,'PFISRExample')

    makepicklefile(fname+'.pickle',beamlist,radarname,simparams)

def makeinputh5(Iono,basedir):
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

    for izn,iz in enumerate(zlist):
        arr = sp.argwhere(idx==izn)
        outdata[izn]=sp.mean(Param_List[arr],axis=0)
        outvel[izn]=sp.mean(velocity[arr],axis=0)

    Ionoout = IonoContainer(datalocsave,outdata,times,Iono.Sensor_loc,ver=0,
                            paramnames=Iono.Param_Names, species=Iono.Species,velocity=outvel)
    Ionoout.saveh5(os.path.join(basedir,'startdata.h5'))

def main():
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Testdata','Barker')
    origparamsdir = os.path.join(testpath,'Origparams')
    if not os.path.exists(testpath):
        os.mkdir(testpath)
        print "Making a path for testdata at "+testpath
    if not os.path.exists(origparamsdir):
        os.mkdir(origparamsdir)
        print "Making a path for testdata at "+origparamsdir
#    makeconfigfile(testpath)


    Icont1 = MakeTestIonoclass(testv=True,testtemp=False,N_0=1e12,z_0=150.0,H_0=50.0)
    makeinputh5(Icont1,testpath)
    Icont1.saveh5(os.path.join(origparamsdir,'0 testiono.h5'))
    funcnamelist=['spectrums','radardata','fitting']

    failflag=runsim.main(funcnamelist,testpath,os.path.join(testpath,'PFISRExample.pickle'),True)
    if not failflag:
        analysisdump(testpath,os.path.join(testpath,'PFISRExample.pickle'))
if __name__== '__main__':

    main()