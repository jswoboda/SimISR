#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a set of data 
@author: John Swoboda
"""
import os,inspect
import scipy as sp
import scipy.fftpack as scfft
import scipy.interpolate as spinterp
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from RadarDataSim.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf,makeconfigfile
from ISRSpectrum.ISRSpectrum import ISRSpectrum

def configsetup(testpath):
    """This function will make a pickle file used to configure the simulation.
    Inputs
    testpath - A string for the path that this file will be saved."""
    beamlist = [64016] # list of beams in
    radarname = 'pfisr'# name of radar for parameters can either be pfisr or risr

    Tint=60.0 # integration time in seconds
    time_lim = 4.0*Tint # simulation length in seconds
    fitter_int = 60.0 # time interval between fitted params
#    pulse = sp.ones(14)# pulse
    rng_lims = [150,500]# limits of the range gates
    IPP = .0087 #interpulse period in seconds
    NNs = 28 # number of noise samples per pulse
    NNp = 100 # number of noise pulses
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
                   'startfile':os.path.join(testpath,'startdata.h5')}# file used for starting points
#                   'SUMRULE': sp.array([[-2,-3,-3,-4,-4,-5,-5,-6,-6,-7,-7,-8,-8,-9]
#                       ,[1,1,2,2,3,3,4,4,5,5,6,6,7,7]])}

    fname = os.path.join(testpath,'PFISRExample')

    makeconfigfile(fname+'.ini',beamlist,radarname,simparams)
def configfilesetup(testpath,npulses):
    
    
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    defcon = os.path.join(curloc,'statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = ratio1 * simparams['TimeLim']
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(os.path.join(testpath,'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    