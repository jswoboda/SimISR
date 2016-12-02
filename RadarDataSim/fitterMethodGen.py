#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""

#imported basic modules
import os
# Imported scipy modules
import scipy as sp
import scipy.optimize
# My modules
from .IonoContainer import IonoContainer#, makeionocombined
from .utilFunctions import readconfigfile
from RadarDataSim.specfunctions import ISRSfitfunction

def defaultparamsfunc(curlag,sensdict,simparams):
    return(curlag,sensdict,simparams)

class Fitterionoconainer(object):
    def __init__(self,Ionocont,Ionosig = None,inifile='default.ini'):
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
        if self.sig is None:
            Nesig = None
        else:
            if self.sig.Param_List.ndim==4:
                Nesig = sp.absolute(self.sig.Param_List[:,:,0,0]*(1.0+Tratio)**2)
            elif self.sig.Param_List.ndim==3:
                Nesig = sp.absolute(self.sig.Param_List[:,:,0]*(1.0+Tratio)**2)
        return (Ne,Nesig)
    def fitdata(self,fitfunc,startinputs,fittimes=None):
        """This funcition is used to fit data given in terms of lags """

        # get intial guess for NE
        Ne_start,Ne_sig =self.fitNE()
        if self.simparams['Pulsetype'].lower()=='barker':
            if Ne_sig is None:
                 return (Ne_start[:,:,sp.newaxis],None,None)
            else:
                 return(Ne_start[:,:,sp.newaxis],Ne_sig[:,:,sp.newaxis],None)
        # get the data and noise lags
        lagsData= self.Iono.Param_List.copy()
        (Nloc,Nt,Nlags) = lagsData.shape
        # Need list of times to save time

        if fittimes is None:
            fittimes = range(Nt)
        else:
            if len(fittimes)==0:
                fittimes=range(Nt)
            else:
                Nt=len(fittimes)


        print('\nData Now being fit.')
        first_lag = True
        x_0all = startvalfunc(Ne_start,self.Iono.Cart_Coords,self.Iono.Time_Vector[:,0],startinputs)
        nparams=x_0all.shape[-1]
        x_0_red = sp.zeros(4)
        specs = self.simparams['species']
        nspecs = len(specs)
        ni = nspecs-1
        sigexist = not self.sig is None
        L = self.sensdict['taurg']*2
        dof = 2*L-4
        if dof<=0:
            dof=1
        for itn,itime in enumerate(fittimes):
            print('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt))
            for iloc in range(Nloc):
                print('\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} now being fit.'.format(itime,Nt,iloc,Nloc))
                curlag = lagsData[iloc,itime]
                if sp.any(sp.isnan(curlag)) or sp.all(curlag==0):
                    print('\t\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} is NaN, skipping.'.format(itime,Nt,iloc,Nloc))
                    continue
                x_0 = x_0all[iloc,itime]
                Niratio = x_0[0:2*ni:2]/x_0[2*ni]
                Ti = (Niratio*x_0[1:2*ni:2]).sum()

                if sp.any(sp.isnan(x_0)):
                    print('\t\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} is NaN, skipping.'.format(itime,Nt,iloc,Nloc))
                    continue
                d_func = (curlag,self.sensdict,self.simparams,Niratio)
                if first_lag:
                    first_lag = False
                    fittedarray = sp.zeros((Nloc,Nt,nparams+1))*sp.nan
                    fittederror = sp.zeros((Nloc,Nt,nparams+1))*sp.nan
                    funcevals = sp.zeros((Nloc,Nt))
                # get uncertianties
                if sigexist:
                    cursig = self.sig.Param_List[iloc,itime]
                    if cursig.ndim<2:
                        sigscov=sp.diag(cursig**2)
                    else:
                        sigscov=cursig
                    if self.simparams['FitType'].lower()=='spectrum':
                        # these uncertianties are derived from the acf variances.
                        Nspec = self.simparams['numpoints']
                        #XXX when creating these variences I'm assuming the lags are independent
                        # this isn't true and I should use the ambiguity function to fix this.
                        acfvarmat = sp.diag(sigscov)
                        # calculate uncertianties by applying the FFT to the columns and the
                        # ifft to the rows. Then multiply by the constant to deal with the different size ffts
                        sigscov = sp.ifft(sp.fft(acfvarmat,n=int(Nspec),axis=0),n=int(Nspec),axis=1)*Nspec**2/Nlags

                # Only fit Ti, Te, Ne and Vi
                x_0_red[0]=Ti
                x_0_red[1:] = x_0[2*ni:]
                # Perform the fitting
                optresults = scipy.optimize.least_squares(fun=ISRSfitfunction,x0=x_0_red,method='lm',verbose=0,args=d_func)
                x=optresults.x
                # Derive data for the ions using output from the fitter and ion species ratios which are assumed to be given.
                ionstuff = sp.zeros(ni*2-1)
                ionstuff[:2*ni:2]=x[1]*Niratio
                ionstuff[1:2*ni-1:2] = x[0]
                fittedarray[iloc,itn] = sp.append(ionstuff,sp.append(x,Ne_start[iloc,itime]))
#                fittedarray[iloc,itime] = sp.append(optresults.x,Ne_start[iloc,itime])
                resid = optresults.cost
                jac=optresults.jac
                jacc = jac[0::2]+jac[1::2]
                try:
                    # Derive covariances for the ions using output from the fitter and ion species ratios which are assumed to be given.
                    #covf = sp.linalg.inv(sp.dot(jac.transpose(),jac))*resid/dof

                    if sigexist:
                        covf = sp.linalg.inv(sp.dot(sp.dot(jacc.transpose(),sp.linalg.inv(sigscov)),jacc))
                    else:
                        covf = sp.linalg.inv(sp.dot(jac.transpose(),jac))*resid/dof
                    vars_vec = sp.diag(covf).real
                    ionstuff = sp.zeros(ni*2-1)
                    ionstuff[:2*ni:2]=vars_vec[1]*Niratio
                    ionstuff[1:2*ni-1:2] = vars_vec[0]
                    vars_vec = sp.append(ionstuff,vars_vec)
                except:
                    vars_vec = sp.ones(nparams)*float('nan')

#
                if len(vars_vec)<fittederror.shape[-1]-1:
                    pdb.set_trace()
                fittederror[iloc,itn,:-1]=vars_vec

                if not self.sig is None:
                    fittederror[iloc,itn,-1] = Ne_sig[iloc,itime]

            print('\t\tData for Location {0:d} of {1:d} fitted.'.format(iloc,Nloc))
        return(fittedarray,fittederror,funcevals)

#%% start values for the fit function
def startvalfunc(Ne_init, loc,time,inputs):
    """ This is a method to determine the start values for the fitter.
    Inputs
        Ne_init - A nloc x nt numpy array of the initial estimate of electron density. Basically
        the zeroth lag of the ACF.
        loc - A nloc x 3 numpy array of cartisian coordinates.
        time - A nt x 2 numpy array of times in seconds
        exinputs - A list of extra inputs allowed for by the fitter class. It only
            has one element and its the name of the ionocontainer file holding the
            rest of the start parameters.
    Outputs
        xarray - This is a numpy arrya of starting values for the fitter parmaeters."""
    if isinstance(inputs,str):
        if os.path.splitext(inputs)[-1]=='.h5':
            Ionoin = IonoContainer.readh5(inputs)
        elif os.path.splitext(inputs)[-1]=='.mat':
            Ionoin = IonoContainer.readmat(inputs)
        elif os.path.splitext(inputs)[-1]=='':
            Ionoin = IonoContainer.makeionocombined(inputs)
    elif isinstance(inputs,list):
        Ionoin = IonoContainer.makeionocombined(inputs)
    else:
        Ionoin = inputs

    numel =sp.prod(Ionoin.Param_List.shape[-2:]) +1

    xarray = sp.zeros((loc.shape[0],len(time),numel))
    for ilocn, iloc in enumerate(loc):
        (datast,vel)=Ionoin.getclosest(iloc,time)[:2]
        datast[:,-1,0] = Ne_init[ilocn,:]
        ionoden =datast[:,:-1,0]
        ionodensum = sp.repeat(sp.sum(ionoden,axis=-1)[:,sp.newaxis],ionoden.shape[-1],axis=-1)
        ionoden = sp.repeat(Ne_init[ilocn,:,sp.newaxis],ionoden.shape[-1],axis=-1)*ionoden/ionodensum
        datast[:,:-1,0] = ionoden
        xarray[ilocn,:,:-1]=sp.reshape(datast,(len(time),numel-1))
        locmag = sp.sqrt(sp.sum(iloc*iloc))
        ilocmat = sp.repeat(iloc[sp.newaxis,:],len(time),axis=0)
        xarray[ilocn,:,-1] = sp.sum(vel*ilocmat)/locmag
    return xarray

