#!/usr/bin/env python
"""
Created on Tue May  5 17:16:51 2015

@author: John Swoboda
"""
import pdb
import argparse
from SimISR import Path
import scipy as sp
from SimISR.utilFunctions import makeconfigfile, GenBarker#,readconfigfile
from SimISR.IonoContainer import IonoContainer, MakeTestIonoclass
from SimISR import runsimisr
from SimISR.analysisplots import analysisdump

def makeconfigfilebarker(testpath):
    testpath = Path(testpath).expanduser()

    beamlist = [64094,64091,64088,64085,64082,64238,64286,64070,64061,64058,64055,64052,
                64049,64046,64043,64067,64040,64037,64034]
    radarname = 'pfisr'

    Tint = 4.0*60.0
    time_lim = 3.0*Tint
    pulse = GenBarker(7)
    rng_lims = [75.0,250.0]
    IPP = .0087
    NNs = 28
    NNp = 100
    t_s=2e-5
    Pulselength=len(pulse)*t_s
    simparams =   {'IPP':IPP,
                   'TimeLim':time_lim,
                   'RangeLims':rng_lims,
                   'Pulse':pulse,
                   'Pulsetype':'barker',
                   'Pulselength':Pulselength,
                   't_s':t_s,
                   'Tint':Tint,
                   'Fitinter':Tint,
                   'NNs': NNs,
                   'NNp':NNp,
                   'dtype':sp.complex128,
                   'ambupsamp':30,
                   'species':['O+','e-'],
                   'numpoints':128,
                   'FitType':'acf',
                   'startfile':str(testpath.joinpath('startdata.h5'))}

    fn = testpath/'barkertest.yml'

    makeconfigfile(fn,beamlist,radarname,simparams)


def makeinputh5(Iono,basedir):
    basedir = Path(basedir).expanduser()

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


    ofn = basedir/'startdata.h5'
    print('writing {}'.format(ofn))
    Ionoout.saveh5(str(ofn))
def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    # if str_input is None:
    parser = argparse.ArgumentParser()
    # else:
    #     parser = argparse.ArgumentParser(str_input)
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="prints debug output and additional detail.")
    parser.add_argument('-p', "--path", dest='path',
                        default=None, help='Path to the Digital RF files and meta data.')
    parser.add_argument('-d', "--drawplots", default=False, dest='drawplots', action="store_true",
                        help="Bool to determine if plots will be made and saved.")
    parser.add_argument('-m', default=True, dest='makeorigdata', action="store_false",
                        help="Bool to determine the input data will be made and saved.")
    parser.add_argument('-f', "--funclist", dest='funclist', nargs='*', type=str,
                        default=['spectrums','radardata','fitting'],
                        help='Strings representing the type of SimISR calculations.')


    if str_input is None:
        return parser.parse_args()
    else:
        if isinstance(str_input, str):
            str_input.split()
        return parser.parse_args(str_input)

def main(input_str=None):
    args_commd = parse_command_line(input_str)
    if args_commd.path is None:
        curloc = Path(__file__).resolve()
        testpath = curloc.parent.parent/'Testdata'/'Barker'
    else:
        testpath = Path(args_commd.path)

    origparamsdir = testpath/'Origparams'
    testpath.mkdir(exist_ok=True,parents=True)
    print("Making a path for testdata at {}".format(testpath))

    origparamsdir.mkdir(exist_ok=True,parents=True)
    print("Making a path for testdata at {}".format(origparamsdir))
    makeconfigfilebarker(testpath)

    if args_commd.makeorigdata:
        Icont1 = MakeTestIonoclass(testv=True,testtemp=False,N_0=1e12,z_0=150.0,H_0=50.0)
        makeinputh5(Icont1,str(testpath))
        Icont1.saveh5(str(origparamsdir.joinpath('0 testiono.h5')))

    funcnamelist = args_commd.funclist

    failflag=runsimisr(funcnamelist,testpath,str(testpath.joinpath('barkertest.yml')),True)
    if not failflag:
        analysisdump(testpath,testpath.joinpath('barkertest.yml'))
if __name__== '__main__':

    main()
