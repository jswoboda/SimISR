#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""

#imported basic modules
import os, inspect, time
import pdb
# Imported scipy and matplotlib modules
import scipy as sp
import scipy.optimize
from matplotlib import rc
import matplotlib.pylab as plt
# My modules
from IonoContainer import IonoContainer


def defaultparamsfunc(curlag,amb_dict,sensdict,npts,numtype):
    return(curlag,amb_dict,sensdict,npts,numtype)

class Fitterionoconainer(object):
    def __init__(self,Ionocont,sensdict,simparams):
        """ The init function for the fitter take the inputs for the fitter programs.

            Inputs:
            DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            sensdict: The dictionary that holds the sensor info.
            simparams: The dictionary that hold the specific simulation params"""

        self.Iono = Ionocont
        self.sensdict = sensdict
        self.simparams = simparams
    def fitNE(self,Tratio = 1):
        """ This funtction will fit electron density assuming Te/Ti is constant
        thus only the zero lag will be needed.
        Inputs:
        Tratio: Optional  a scaler for the Te/Ti.
        Outputs:
        Ne: A numpy array that is NtxNbxNrg, Nt is number of times, Nb is number
        of beams and Nrg is number of range gates."""

        Ne = sp.absolute(self.Iono.Param_List[:,:,0]*(1.0+Tratio))
        return Ne
    def fitdata(self,fitfunc,startvalfunc,npts=64,numtype = sp.complex128, d_funcfunc=defaultparamsfunc,exinputs=[]):
        """ """

        # get intial guess for NE
        Ne_start =self.fitNE()
        # get the data nd noise lags
        lagsData= self.Iono.Param_List
        (Nloc,Nt,Nlags) = lagsData.shape

        # normalized out parameters

        Pulse_shape = self.simparams['Pulse']

        print('\nData Now being fit.')
        first_lag = True
        x_0all = startvalfunc(Ne_start,self.Iono.Cart_Coords,self.Iono.Time_Vector,exinputs)
        for itime in range(Nt):
            print('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt))
            for iloc in range(Nloc):
                print('\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} now being fit.'.format(itime,Nt,iloc,Nloc))
               # self.simparams['Rangegatesfinal'][irngnew] = sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1])
                curlag = lagsData[iloc,itime]
                d_func = d_funcfunc(curlag, self.simparams['amb_dict'],self.sensdict,npts,numtype)
                x_0 = x_0all[iloc,itime]
                if first_lag:
                    first_lag = False
                    nparams = len(x_0)
                    fittedarray = sp.zeros((Nloc,Nt,nparams))
                    fittederror = sp.zeros((Nloc,Nt,nparams,nparams))
                try:
                    (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,x0=x_0,args=d_func,full_output=True)

                    fittedarray[iloc,itime] = x
                    if cov_x == None:
                        fittederror[iloc,itime] = sp.ones((len(x_0),len(x_0)))*float('nan')
                    else:
                        fittederror[iloc,itime] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                except TypeError:
                    pdb.set_trace()

            print('\t\tData for Location {0:d} of {1:d} fitted.'.format(iloc,Nloc))
        return(fittedarray,fittederror)

