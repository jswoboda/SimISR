#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""

#imported basic modules
import os, inspect, time
import pdb
# Imported scipy modules
import scipy as sp
import scipy.optimize
# My modules
from IonoContainer import IonoContainer
from utilFunctions import readconfigfile
from RadarDataSim.specfunctions import ISRSfitfunction


def defaultparamsfunc(curlag,sensdict,simparams):
    return(curlag,sensdict,simparams)

class Fitterionoconainer(object):
    def __init__(self,Ionocont,Ionosig,inifile):
        """ The init function for the fitter take the inputs for the fitter programs.

            Inputs:
            DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            sensdict: The dictionary that holds the sensor info.
            simparams: The dictionary that hold the specific simulation params"""
        (self.sensdict,self.simparams) = readconfigfile(inifile)
        self.Iono = Ionocont
        self.sig = Ionosig
    def fitNE(self,Tratio = 1):
        """ This funtction will fit electron density assuming Te/Ti is constant
        thus only the zero lag will be needed.
        Inputs:
        Tratio: Optional  a scaler for the Te/Ti.
        Outputs:
        Ne: A numpy array that is NtxNbxNrg, Nt is number of times, Nb is number
        of beams and Nrg is number of range gates."""

        Ne = sp.absolute(self.Iono.Param_List[:,:,0]*(1.0+Tratio))
        Nesig = sp.absolute(self.sig.Param_List[:,:,0]*(1.0+Tratio))
        return (Ne,Nesig)
    def fitdata(self,fitfunc,startvalfunc,d_funcfunc=defaultparamsfunc,exinputs=[]):
        """This funcition is used to fit data given in terms of lags """

        # get intial guess for NE
        Ne_start,Ne_sig =self.fitNE()
        if self.simparams['Pulsetype'].lower()=='barker':
            return(Ne_start[:,:,sp.newaxis],Ne_sig[:,:,sp.newaxis])
        # get the data and noise lags
        lagsData= self.Iono.Param_List.copy()
        (Nloc,Nt,Nlags) = lagsData.shape


        print('\nData Now being fit.')
        first_lag = True
        x_0all = startvalfunc(Ne_start,self.Iono.Cart_Coords,self.Iono.Time_Vector,exinputs)
        nparams=x_0all.shape[-1]+1
        for itime in range(Nt):
            print('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt))
            for iloc in range(Nloc):
                print('\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} now being fit.'.format(itime,Nt,iloc,Nloc))
                curlag = lagsData[iloc,itime]
                d_func = d_funcfunc(curlag,self.sensdict,self.simparams)
                x_0 = x_0all[iloc,itime]
                #XXX Added random noise to start points
                # add some random noise so we don't just go to the desired value right away
                x_rand = sp.random.standard_normal(x_0.shape)*sp.sqrt(.1)*x_0
                x_0 =x_0+x_rand

                if first_lag:
                    first_lag = False
                    fittedarray = sp.zeros((Nloc,Nt,nparams))
                    fittederror = sp.zeros((Nloc,Nt,nparams,nparams))
                # get uncertianties
                if self.simparams['FitType'].lower()=='acf':
                    # this is calculated from a formula
                    d_func = d_func+(self.sig.Param_List[iloc,itime],)
                elif self.simparams['FitType'].lower()=='spectrum':
                    # these uncertianties are derived from the acf variances.
                    acfvar = self.sig.Param_List[iloc,itime]**2
                    Nspec = self.simparams['numpoints']
                    #XXX when creating these variences I'm assuming the lags are independent
                    # this isn't true and I should use the ambiguity function to fix this.
                    acfvarmat = sp.diag(acfvar)
                    # calculate uncertianties by applying the FFT to the columns and the
                    # ifft to the rows. Then multiply by the constant to deal with the different size ffts
                    specmat = sp.ifft(sp.fft(acfvarmat,n=Nspec,axis=0),n=Nspec,axis=1)*Nspec**2/Nlags
                    specsig = sp.sqrt(sp.diag(specmat.real))
                    d_func = d_func+(specsig,)

                (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,
                    x0=x_0,args=d_func,full_output=True)


                fittedarray[iloc,itime] = sp.append(x,Ne_start[iloc,itime])
                if cov_x is None:

                    fittederror[iloc,itime,:-1,:-1] = sp.ones((len(x_0),len(x_0)))*float('nan')
                else:
                    fittederror[iloc,itime,:-1,:-1] = sp.sqrt(sp.absolute(cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))))
                fittederror[iloc,itime,-1,-1] = Ne_sig[iloc,itime]

            print('\t\tData for Location {0:d} of {1:d} fitted.'.format(iloc,Nloc))
        return(fittedarray,fittederror)



#%% fit function stuff
def simpstart(Ne_init, loc,time,exinputs):
    """ """

    xarray = sp.zeros((loc.shape[0],len(time),5))
    xarray[:,:,0] = Ne_init
    xarray[:,:,2] = Ne_init
    xarray[:,:,1] = 1e3
    xarray[:,:,3] = 1e3
    xarray[:,:,4] = 0.0
    return xarray
#%% Testing
def main():
    """ Test function for the RadarData class."""
    t1 = time.time()
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Testdata')
    Ionoin=IonoContainer.readh5(os.path.join(testpath,'lags.h5'))
    inifile = os.path.join(testpath,'PFISRExample.pickle')

    fitterone = Fitterionoconainer(Ionoin,inifile)
    (fitteddata,fittederror) = fitterone.fitdata(ISRSfitfunction,simpstart)
    (Nloc,Ntimes,nparams)=fitteddata.shape
    fittederronly = fittederror[:,:,range(nparams),range(nparams)]
    paramlist = sp.concatenate((fitteddata,fittederronly),axis=2)
    paramnames = []
    species = fitterone.simparams['species']
    for isp in species[:-1]:
        paramnames.append('Ni_'+isp)
        paramnames.append('Ti_'+isp)
    paramnames = paramnames+['Ne','Te','Vi']
    paramnamese = ['n'+ip for ip in paramnames]
    paranamsf = sp.array(paramnames+paramnamese)


    Ionoout=IonoContainer(Ionoin.Sphere_Coords,paramlist,Ionoin.Time_Vector,ver =1,
                          coordvecs = Ionoin.Coord_Vecs, paramnames=paranamsf,species=species)

    Ionoout.saveh5(os.path.join(testpath,'fittedtestdata.h5'))
    t2 = time.time()
    print(t2-t1)
if __name__== '__main__':
    main()