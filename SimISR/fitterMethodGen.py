#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""

#imported basic modules
import os
import ipdb
# Imported scipy modules
import scipy as sp
import scipy.constants as sc
import scipy.optimize
# My modules
from SimISR.IonoContainer import IonoContainer, makeionocombined
from SimISR.utilFunctions import readconfigfile, update_progress

def defaultparamsfunc(curlag, sensdict, simparams):
    """
        Just a pass through function.
    """
    return(curlag, sensdict, simparams)

class Fitterionoconainer(object):
    """
        This is fitter class for SimISR
    """
    def __init__(self, ionoacf, Ionosig=None, inifile='default.ini'):
        """ The init function for the fitter take the inputs for the fitter programs.

            Inputs:
            ionoacf: The lags in ionocontainer format.
            Ionosig:
            sensdict: The dictionary that holds the sensor info.
            simparams: The dictionary that hold the specific simulation params"""
        (self.sensdict, self.simparams) = readconfigfile(inifile, make_amb=True)
        self.acf = ionoacf
        self.sig = Ionosig
    def fitNE(self, Tratio=1):
        """ This funtction will fit electron density assuming Te/Ti is constant
        thus only the zero lag will be needed.
        Inputs:
        Tratio: Optional  a scaler for the Te/Ti.
        Outputs:
        Ne: A numpy array that is NtxNbxNrg, Nt is number of times, Nb is number
        of beams and Nrg is number of range gates."""

        Ne = sp.absolute(self.acf.Param_List[:, :, 0]*(1.0+Tratio))
        if self.sig is None:
            Nesig = None
        else:
            if self.sig.Param_List.ndim == 4:
                Nesig = sp.absolute(self.sig.Param_List[:, :, 0, 0]*(1.0+Tratio)**2)
            elif self.sig.Param_List.ndim == 3:
                Nesig = sp.absolute(self.sig.Param_List[:,:,0]*(1.0+Tratio)**2)
        return (Ne,Nesig)

    def fitdata(self,fitfunc,startinputs,fittimes=None,printlines=True):
        """This funcition is used to fit data given in terms of lags """

        # get intial guess for NE
        Ne_start, Ne_sig = self.fitNE()
        if self.simparams['Pulsetype'].lower() == 'barker':
            if Ne_sig is None:
                return (Ne_start[:, :, sp.newaxis], None, None, None)
            else:
                return(Ne_start[:, :, sp.newaxis], Ne_sig[:, :, sp.newaxis], None, None)
        # get the data and noise lags
        lagsData= self.acf.Param_List.copy()
        (Nloc, Nt, Nlags) = lagsData.shape
        # Need list of times to save time
        if fittimes is None:
            fittimes = range(Nt)
        else:
            if len(fittimes) ==0:
                fittimes = range(Nt)
            else:
                Nt = len(fittimes)

        # Fit mode
        if 'fitmode' in self.simparams.keys():
            fitmode = self.simparams['fitmode']
        else:
            fitmode = 0
        print('\nData Now being fit.')
        # HACK taking ambiugty into account with start values
        ds_fac = sp.prod(self.simparams['declist'])
        p_samps = len(self.simparams['Pulse'])/ds_fac
        pvec = sp.arange(-sp.ceil((p_samps-1)/2.), sp.floor((p_samps-1)/2.)+1)/p_samps
        amblen = .5*sc.c*self.simparams['Pulselength']*pvec*1e-3
        first_lag = True
        x_0all = startvalfunc(Ne_start, self.acf.Cart_Coords, self.acf.Sphere_Coords,
                              self.acf.Time_Vector[:, 0], startinputs, amblen)
        nparams = x_0all.shape[-1]
        x_0_red = sp.zeros(4)
        specs = self.simparams['species']
        nspecs = len(specs)
        ni = nspecs-1
        sigexist = not self.sig is None
        L = self.sensdict['taurg']*2
        dof = 2*L-4
        if dof <= 0:
            dof = 1
        for itn, itime in enumerate(fittimes):
            if printlines:
                prgstr = 'Data for time {0:d} of {1:d} now being fit.'.format(itime, Nt)
                update_progress(float(itime)/Nt, prgstr)
            for iloc in range(Nloc):
                if printlines:
                    prgstr = '\t Time:{0:d} of {1:d} Location:{2:d} of {3:d} now being fit.'.format(itime, Nt, iloc, Nloc)
                    update_progress(float(itime)/Nt + float(iloc)/Nt/Nloc, prgstr)
                curlag = lagsData[iloc, itime]
                if sp.any(sp.isnan(curlag)) or sp.all(curlag == 0):
                    if printlines:
                        prgstr = 'Time:{0:d} of {1:d} Location:{2:d} of {3:d} is NaN, skipping.'.format(itime, Nt, iloc, Nloc)
                        update_progress(float(itime)/Nt + float(iloc)/Nt/Nloc, prgstr)
                    continue
                x_0 = x_0all[iloc, itime]
                Niratio = x_0[0:2*ni:2]/x_0[2*ni]
                Ti = (Niratio*x_0[1:2*ni:2]).sum()

                if sp.any(sp.isnan(x_0)):
                    if printlines:
                        prgstr = 'Time:{0:d} of {1:d} Location:{2:d} of {3:d} is NaN, skipping.'.format(itime, Nt, iloc, Nloc)
                        update_progress(float(itime)/Nt + float(iloc)/Nt/Nloc,
                                        prgstr)
                    continue
                d_func = (curlag, self.sensdict, self.simparams, Niratio)
                if first_lag:
                    first_lag = False
                    fittedarray = sp.zeros((Nloc, Nt, nparams+1))*sp.nan
                    fittederror = sp.zeros((Nloc, Nt, nparams+1))*sp.nan
                    fittedcov = sp.zeros((Nloc, Nt, 4, 4))*sp.nan
                    funcevals = sp.zeros((Nloc, Nt))
                # get uncertianties
                if sigexist:
                    cursig = self.sig.Param_List[iloc, itime]
                    if cursig.ndim < 2:
                        sigscov = sp.diag(cursig**2)
                    else:
                        sigscov = cursig
                    if self.simparams['FitType'].lower() == 'spectrum':
                        # these uncertianties are derived from the acf variances.
                        ds_fac = sp.prod(self.simparams['declist'])
                        Nspec = self.simparams['numpoints']/ds_fac
                        #XXX when creating these variences I'm assuming the lags are independent
                        # this isn't true and I should use the ambiguity function to fix this.
                        acfvarmat = sp.diag(sigscov)
                        # calculate uncertianties by applying the FFT to the columns and the
                        # ifft to the rows. Then multiply by the constant to deal with the different size ffts
                        sigscov = sp.ifft(sp.fft(acfvarmat, n=int(Nspec), axis=0),
                                          n=int(Nspec), axis=1)*Nspec**2/Nlags

                # Only fit Ti, Ne, Te and Vi
                x_0_red[0] = Ti
                x_0_red[1:] = x_0[2*ni:]
                # change variables because of new fit mode
                if fitmode == 1:
                    x_0_red[2] = x_0_red[2]/Ti
                elif fitmode == 2:
                    x_0_red[2] = x_0_red[2]/Ti
                    x_0_red[1] = x_0_red[1]/(1+x_0_red[2])
                # Perform the fitting
                optresults = scipy.optimize.least_squares(fun=fitfunc, x0=x_0_red,
                                                          method='lm', verbose=0, args=d_func)
                x_res = optresults.x.real
                # Derive data for the ions using output from the fitter and ion species ratios which are assumed to be given.
                ionstuff = sp.zeros(ni*2-1)
                ionstuff[:2*ni:2] = x_res[1]*Niratio
                ionstuff[1:2*ni-1:2] = x_res[0]
                # change variables because of new fit mode
                if fitmode == 1:
                    x_res[2] = x_res[2]*x_res[0]
                elif fitmode == 2:
                    x_res[1] = x_res[1]*(1+x_res[2])
                    x_res[2] = x_res[2]*x_res[0]

                fittedarray[iloc, itn] = sp.append(ionstuff,
                                                   sp.append(x_res, Ne_start[iloc, itime]))
                funcevals[iloc, itn] = optresults.nfev
#                fittedarray[iloc,itime] = sp.append(optresults.x,Ne_start[iloc,itime])
                resid = optresults.cost
                jac = optresults.jac
                # combine the rows because of the complex conjugates
                jacc = jac[0::2]+jac[1::2]
                try:
                    # Derive covariances for the ions using output from the fitter and ion species ratios which are assumed to be given.
                    #covf = sp.linalg.inv(sp.dot(jac.transpose(),jac))*resid/dof

                    if sigexist:
                        covf = sp.linalg.inv(sp.dot(sp.dot(jacc.transpose(),
                                                           sp.linalg.inv(sigscov)), jacc))
                    else:
                        covf = sp.linalg.inv(sp.dot(jac.transpose(), jac))*resid/dof
                    # change variables because of new fit mode
                    if fitmode == 1:
                        covf[2] = covf[2]*x_res[0]**2
                        covf[:, 2] = covf[:, 2]*x_res[0]**2
                    elif fitmode == 2:
                        # is this right?
                        covf[1] = covf[1]*(1+x_res[2]/x_res[0])**2
                        covf[:, 1] = covf[:, 1]*(1+x_res[2]/x_res[0])**2
                        covf[2] = covf[2]*x_res[0]**2
                        covf[:, 2] = covf[:, 2]*x_res[0]**2

                    vars_vec = sp.diag(covf).real
                    ionstuff = sp.zeros(ni*2-1)
                    ionstuff[:2*ni:2] = vars_vec[1]*Niratio
                    ionstuff[1:2*ni-1:2] = vars_vec[0]
                    vars_vec = sp.append(ionstuff, vars_vec)
                    fittedcov[iloc, itn] = covf
                except:#sp.linalg.LinAlgError('singular matrix'):
                    vars_vec = sp.ones(nparams)*float('nan')

#                if len(vars_vec)<fittederror.shape[-1]-1:
#                    pdb.set_trace()
                fittederror[iloc, itn, :-1] = vars_vec

                if not self.sig is None:
                    fittederror[iloc, itn, -1] = Ne_sig[iloc, itime]

                #print('\t\tData for Location {0:d} of {1:d} fitted.'.format(iloc, Nloc))
        return(fittedarray, fittederror, funcevals, fittedcov)

#%% start values for the fit function
def startvalfunc(Ne_init, loc, locsp, time, inputs, ambinfo = [0]):
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
        xarray - This is a numpy array of starting values for the fitter parameters."""

    if isinstance(inputs, str):
        if os.path.splitext(inputs)[-1] == '.h5':
            ionoin = IonoContainer.readh5(inputs)
        elif os.path.splitext(inputs)[-1] == '.mat':
            ionoin = IonoContainer.readmat(inputs)
        elif os.path.splitext(inputs)[-1] == '':
            ionoin = makeionocombined(inputs)
    elif isinstance(inputs, list):
        ionoin = makeionocombined(inputs)
    else:
        ionoin = inputs

    numel = sp.prod(ionoin.Param_List.shape[-2:]) +1

    xarray = sp.zeros((loc.shape[0], len(time), numel))
    for ilocn, iloc in enumerate(loc):
        #for iamb in ambinfo:
        newlocsp = locsp[ilocn]
        #newlocsp[0] += iamb
        (datast, vel) = ionoin.getclosestsphere(newlocsp, time)[:2]
        datast[:, -1, 0] = Ne_init[ilocn, :]
        # get the ion densities
        ionoden = datast[:, :-1, 0]
        # find the right normalization for the ion species
        ionodensum = sp.repeat(sp.sum(ionoden, axis=-1)[:, sp.newaxis],
                               ionoden.shape[-1], axis=-1)
        # renormalized to the right level
        ionoden = sp.repeat(Ne_init[ilocn, :, sp.newaxis], ionoden.shape[-1],
                            axis=-1)*ionoden/ionodensum
        datast[:, :-1, 0] = ionoden
        xarray[ilocn, :, :-1] = sp.reshape(datast, (len(time), numel-1))#/float(len(ambinfo))
        locmag = sp.sqrt(sp.sum(iloc*iloc))
        ilocmat = sp.repeat(iloc[sp.newaxis, :], len(time), axis=0)
        xarray[ilocn, :, -1] = sp.sum(vel*ilocmat)/locmag#/float(len(ambinfo))
    return xarray
