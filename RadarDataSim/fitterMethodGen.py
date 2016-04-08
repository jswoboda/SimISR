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
from RadarDataSim.specfunctions import ISRSfitfunction,ISRSfitfunction_lmfit
from lmfit import Parameters, Minimizer

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
            Nesig = sp.absolute(self.sig.Param_List[:,:,0]*(1.0+Tratio))
        return (Ne,Nesig)
    def fitdata(self,fitfunc,startvalfunc,d_funcfunc=defaultparamsfunc,exinputs=[]):
        """This funcition is used to fit data given in terms of lags """

        # get intial guess for NE
        Ne_start,Ne_sig =self.fitNE()
        if self.simparams['Pulsetype'].lower()=='barker':
            if Ne_sig is None:
                 return Ne_start[:,:,sp.newaxis]
            else:
                 return(Ne_start[:,:,sp.newaxis],Ne_sig[:,:,sp.newaxis])
        # get the data and noise lags
        lagsData= self.Iono.Param_List.copy()
        (Nloc,Nt,Nlags) = lagsData.shape


        print('\nData Now being fit.')
        first_lag = True
        x_0all = startvalfunc(Ne_start,self.Iono.Cart_Coords,self.Iono.Time_Vector[:,0],exinputs)
        nparams=x_0all.shape[-1]+1
        
        
            
        for itime in range(Nt):
            print('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt))
            for iloc in range(Nloc):
                print('\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} now being fit.'.format(itime,Nt,iloc,Nloc))
                curlag = lagsData[iloc,itime]
                d_func = d_funcfunc(curlag,self.sensdict,self.simparams)
                x_0 = x_0all[iloc,itime]
                
                if first_lag:
                    first_lag = False
                    fittedarray = sp.zeros((Nloc,Nt,nparams))
                    fittederror = sp.zeros((Nloc,Nt,nparams,nparams))
                # get uncertianties
                if not self.sig is None:
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
                        specmat = sp.ifft(sp.fft(acfvarmat,n=int(Nspec),axis=0),n=int(Nspec),axis=1)*Nspec**2/Nlags
                        specsig = sp.sqrt(sp.diag(specmat.real))
                        d_func = d_func+(specsig,)
                        
                if fitfunc==ISRSfitfunction:
                    (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,
                        x0=x_0,args=d_func,full_output=True)


                    fittedarray[iloc,itime] = sp.append(x,Ne_start[iloc,itime])
                    if cov_x is None:
    
                        fittederror[iloc,itime,:-1,:-1] = sp.ones((len(x_0),len(x_0)))*float('nan')
                    else:
                        fittederror[iloc,itime,:-1,:-1] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                    if not self.sig is None:
                        fittederror[iloc,itime,-1,-1] = Ne_sig[iloc,itime]
                elif fitfunc == ISRSfitfunction_lmfit:
                    pset = x2params(x_0)
                    M1 = Minimizer(ISRSfitfunction_lmfit,pset,d_func)
                    out = M1.minimize()
                    x,fittederror[iloc,itime,:-1,:-1] = minimizer2x(out)
                    fittedarray[iloc,itime]=sp.append(x,Ne_start[iloc,itime])
                    if not self.sig is None:
                        fittederror[iloc,itime,-1,-1] = Ne_sig[iloc,itime]**2
                    
            print('\t\tData for Location {0:d} of {1:d} fitted.'.format(iloc,Nloc))
        return(fittedarray,fittederror)

def x2params(xvec):
    p1 = Parameters()
    nx = xvec.shape[0]
    ni = (nx-1)/2-1
    plist = []
    nestr = 'Ni0'
    for i1 in range(ni):
        plist.append(('Ni{}'.format(i1),xvec[i1*2],True,0.,1e15,None))
        plist.append(('Ti{}'.format(i1),xvec[i1*2+1],True,0.,1e5,None))
        if i1>0:
            nestr = nestr+'+Ni{}'.format(i1)
    plist.append(('Ne',xvec[-3],True,0.,1e15,nestr))
    plist.append(('Te',xvec[-2],True,0.,1e5,None))
    plist.append(('Vi',xvec[-1],True,-1e6,1e6,None))

    p1.add_many(*plist)
    return p1
    
def minimizer2x(out):
    p1 = out.params
    nx = len(p1.keys())
    ni = (nx-1)/2-1
    varnames = sp.array(out.var_names)
    xvec = sp.zeros(nx)
    new_order = sp.zeros(nx)
    for i1 in range(ni):
        niname = 'Ni{}'.format(i1)
        tiname = 'Ti{}'.format(i1)
        xvec[2*i1]=p1.get(niname).value
        xvec[2*i1+1]=p1.get(tiname).value
        
        new_order[2*i1]=sp.argwhere(niname==varnames)[0][0]
        new_order[2*i1+1]=sp.argwhere(tiname==varnames)[0][0]
    xvec[2*ni] = p1.get('Ne').value
    xvec[2*ni+1] = p1.get('Te').value
    xvec[2*ni+2] = p1.get('Vi').value
    covx = out.covar
    
    new_order[2*ni]=nx-1
    new_order[2*ni+1]=sp.argwhere('Te'==varnames)[0][0]
    new_order[2*ni+2]=sp.argwhere('Vi'==varnames)[0][0]
    
    if covx is None:
        return (xvec,sp.ones((len(xvec),len(xvec)))*float('nan'))
    covx2 = sp.zeros((nx,nx))
    covx2[:nx-1,:nx-1] = covx
    Neerr = out.params.get('Ne').stderr
    covf = sp.array([[covx2[i][j] for j in new_order] for i in new_order])
    covf[2*ni,2*ni]=Neerr*Neerr
    return (xvec,covf)
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