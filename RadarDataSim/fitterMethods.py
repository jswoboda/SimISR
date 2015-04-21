#!/usr/bin/env python
"""
fitterMethods.py

@author: John Swoboda
Holds class that applies the fitter.
"""
from __future__ import absolute_import
#imported basic modules
import os, inspect, time
import pdb
# Imported scipy and matplotlib modules
import scipy as sp
import scipy.fftpack as scfft
import scipy.optimize,scipy.interpolate
from matplotlib import rc
import matplotlib.pylab as plt
# My modules
from .IonoContainer import MakeTestIonoclass
from .radarData import RadarData
from .ISSpectrum import ISSpectrum
import const.sensorConstants as sensconst
from .utilFunctions import make_amb, spect2acf

class FitterBasic(object):
    """ This is a basic fitter to take data created by the RadarData Class and
    make fitted Data."""
    def __init__(self,DataLags,NoiseLags,sensdict,simparams):
        """ The init function for the fitter take the inputs for the fitter programs.

            Inputs:
            DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' for
            the returned value from the RadarData class function processdata.
            sensdict: The dictionary that holds the sensor info.
            simparams: The dictionary that hold the specific simulation params"""

        self.DataDict = DataLags
        self.NoiseDict = NoiseLags
        self.sensdict = sensdict
        self.simparams = simparams
    def fitNE(self,Tratio=1.0):
        """ This funtction will fit electron density assuming Te/Ti is constant
        thus only the zero lag will be needed.
        Inputs:
        Tratio: Optional  a scaler for the Te/Ti.
        Outputs:
        Ne: A numpy array that is NtxNbxNrg, Nt is number of times, Nb is number
        of beams and Nrg is number of range gates."""
        pulsewidth = self.sensdict['taurg']*self.sensdict['t_s']
        txpower = self.sensdict['Pt']
        DataLags = self.DataDict
        NoiseLags = self.NoiseDict
        (Nt,Nbeams,Nrng) = DataLags['Pow'].shape
        power = DataLags['Pow']
        pulses = sp.repeat(DataLags['Pulses'][:,:,None],Nrng,axis=-1)
        power = power/pulses
        #noise power set up
        noisepower = sp.mean(NoiseLags['Pow'],axis=2)
        npulses = NoiseLags['Pulses']
        noisepower = noisepower/npulses
        power = power-sp.repeat(noisepower[:,:,None],Nrng,axis=-1)
        rng_vec = self.sensdict['RG']*1e3
        rng3d = sp.tile(rng_vec,(Nt,Nbeams,1))
        Ksysvec = self.sensdict['Ksys'] # Beam shape and physcial constants
        ksys3d = sp.tile(Ksysvec[None,:,None],(Nt,1,Nrng))

        Ne = power*rng3d*rng3d/(pulsewidth*txpower*ksys3d)*(1.0+Tratio)
        return Ne

    def fitdata2(self,npts=64,numtype = sp.complex128,
                 startvalfunc=defstart,d_funcfunc = default_fit_func2,
                 fitfunc=deffitfunc2):
        """ """

        # get intial guess for NE
        Ne_start =self.fitNE()

        sumrule = self.simparams['SUMRULE']
        minrg = -sp.min(sumrule[0])
        maxrg = len(self.sensdict['RG'])-sp.max(sumrule[1])
        Nrng2 = maxrg-minrg;
        # get the data nd noise lags
        lagsData= self.DataDict['ACF']
        (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
        pulses = sp.tile(self.DataDict['Pulses'][:,:,None,None],(1,1,Nrng,Nlags))

        # average by the number of pulses
        lagsData = lagsData/pulses
        lagsNoise=self.NoiseDict['ACF']
        lagsNoise = sp.mean(lagsNoise,axis=2)
        pulsesnoise = sp.tile(self.NoiseDict['Pulses'][:,:,None],(1,1,Nlags))
        lagsNoise = lagsNoise/pulsesnoise
        lagsNoise = sp.tile(lagsNoise[:,:,None,:],(1,1,Nrng,1))
        # subtract out noise lags
        lagsData = lagsData-lagsNoise

        # normalized out parameters
        pulsewidth = self.sensdict['taurg']*self.sensdict['t_s']
        txpower = self.sensdict['Pt']
        rng_vec = self.sensdict['RG']*1e3
        rng3d = sp.tile(rng_vec[None,None,:,None],(Nt,Nbeams,1,Nlags))
        Ksysvec = self.sensdict['Ksys']
        ksys3d = sp.tile(Ksysvec[None,:,None,None],(Nt,1,Nrng,Nlags))
        lagsData = lagsData*rng3d*rng3d/(pulsewidth*txpower*ksys3d)
        Pulse_shape = self.simparams['Pulse']
        fittedarray = sp.zeros((Nt,Nbeams,Nrng2,nparams))
        fittederror = sp.zeros((Nt,Nbeams,Nrng2,nparams,nparams))
        #self.simparams['Rangegatesfinal'] = sp.zeros(Nrng2)
        self.simparams['Rangegatesfinal'] = sp.array([ sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1]) for irng in sp.arange(minrg,maxrg)])
        print('\nData Now being fit.')
        for itime in sp.arange(Nt):
            print(('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt)))
            for ibeam in sp.arange(Nbeams):
                for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):

                   # self.simparams['Rangegatesfinal'][irngnew] = sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1])
                    curlag = sp.array([sp.mean(lagsData[itime,ibeam,irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag]) for ilag in sp.arange(Nlags)])#/sumreg
                    d_func = d_funcfunc(curlag, Pulse_shape,self.simparams['amb_dict'],self.sensdict,numtype)
                    x_0 = startvalfunc(Ne_start[itime,ibeam,irng])

                    try:
                        (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,x0=x_0,args=d_func,full_output=True)

                        fittedarray[itime,ibeam,irngnew] = x
                        if cov_x == None:
                            fittederror[itime,ibeam,irngnew] = sp.ones((len(x_0),len(x_0)))*float('nan')
                        else:
                            fittederror[itime,ibeam,irngnew] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                    except TypeError:
                        pdb.set_trace()

                print(('\t\tData for Beam {0:d} of {1:d} fitted.'.format(ibeam,Nbeams)))
        return(fittedarray,fittederror)
    def fitdata(self,nparams=4,npnts = 64,numtype=sp.complex128):
        """ This function will fit the electron density Ti and Te for the ISR data.
        Currenly set up to use the Haystack spectrum model.
        Outputs:
        fittedarray: A numpy array that is NtxNbxNrgfxNp that holds the fitted values.
        fittederror: A numpy array that is NtxNbxNrgfxNp that holds the std of fitted values."""

        # Inputs for phils spectrum program. Either needs to become a function input
        # at some point or changed if moving to new spectrum model.
        rm1 = 16# atomic weight of species 1
        rm2 = 1# atomic weight os species 2
        p2 =0 #

        # get intial guess for NE
        Ne_start =self.fitNE()

        sumrule = self.simparams['SUMRULE']
        minrg = -sp.min(sumrule[0])
        maxrg = len(self.sensdict['RG'])-sp.max(sumrule[1])
        Nrng2 = maxrg-minrg;
        # get the data nd noise lags
        lagsData= self.DataDict['ACF']
        (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
        pulses = sp.tile(self.DataDict['Pulses'][:,:,None,None],(1,1,Nrng,Nlags))

        # average by the number of pulses
        lagsData = lagsData/pulses
        lagsNoise=self.NoiseDict['ACF']
        lagsNoise = sp.mean(lagsNoise,axis=2)
        pulsesnoise = sp.tile(self.NoiseDict['Pulses'][:,:,None],(1,1,Nlags))
        lagsNoise = lagsNoise/pulsesnoise
        lagsNoise = sp.tile(lagsNoise[:,:,None,:],(1,1,Nrng,1))
        # subtract out noise lags
        lagsData = lagsData-lagsNoise

        # normalized out parameters
        pulsewidth = self.sensdict['taurg']*self.sensdict['t_s']
        txpower = self.sensdict['Pt']
        rng_vec = self.sensdict['RG']*1e3
        rng3d = sp.tile(rng_vec[None,None,:,None],(Nt,Nbeams,1,Nlags))
        Ksysvec = self.sensdict['Ksys']
        ksys3d = sp.tile(Ksysvec[None,:,None,None],(Nt,1,Nrng,Nlags))
        lagsData = lagsData*rng3d*rng3d/(pulsewidth*txpower*ksys3d)
        Pulse_shape = self.simparams['Pulse']
        fittedarray = sp.zeros((Nt,Nbeams,Nrng2,nparams))
        fittederror = sp.zeros((Nt,Nbeams,Nrng2,nparams,nparams))
        #self.simparams['Rangegatesfinal'] = sp.zeros(Nrng2)
        self.simparams['Rangegatesfinal'] = sp.array([ sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1]) for irng in sp.arange(minrg,maxrg)])
        print('\nData Now being fit.')
        for itime in sp.arange(Nt):
            print(('\tData for time {0:d} of {1:d} now being fit.'.format(itime,Nt)))
            for ibeam in sp.arange(Nbeams):
                for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):

                   # self.simparams['Rangegatesfinal'][irngnew] = sp.mean(self.sensdict['RG'][irng+sumrule[0,0]:irng+sumrule[1,0]+1])
                    curlag = sp.array([sp.mean(lagsData[itime,ibeam,irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag]) for ilag in sp.arange(Nlags)])#/sumreg
                    d_func = (curlag, Pulse_shape,self.simparams['amb_dict'],self.sensdict,rm1,rm2,p2,npnts,numtype)
                    x_0 = sp.array([1000,1000,Ne_start[itime,ibeam,irng],0.0])[:nparams]

                    try:
                        (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=default_fit_func,x0=x_0,args=d_func,full_output=True)

                        fittedarray[itime,ibeam,irngnew] = x
                        if cov_x == None:
                            fittederror[itime,ibeam,irngnew] = sp.ones((len(x_0),len(x_0)))*float('nan')
                        else:
                            fittederror[itime,ibeam,irngnew] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                    except TypeError:
                        pdb.set_trace()

                print(('\t\tData for Beam {0:d} of {1:d} fitted.'.format(ibeam,Nbeams)))
        return(fittedarray,fittederror)

#    def fitdataiono(self,nparams=4,filename=None,npnts = 64,numtype=sp.complex128):
#        '''This function will output an ionocontainer class after the fitting is done. '''
#    def fitNEiono(self, filename=None,Tratio = 1):

    def plotbeams(self,beamnum,radardata,fittedarray,fittederror,timenum = 0,figsdir = None):
        """ This function will plot the fitted data along with error bars and the input
        vs altitude which will be the y axis.
        Input:
        beamnum: List of beams that are desired to be ploted.
        radardata: The an instance of the RadarData class.
        fittedarray: Numpy array of fitted data from the method fitdata.
        fittederror: Numpy array of errors from the method fitdata.
        timenum: The time point that is desired to be shown.
        figsdir: The directory the figure is to be saved."""

        rc('text',usetex=True)
        rc('font',**{'family':'serif','serif':['Computer Modern']})
        d2r = sp.pi/180.0
        angles = self.simparams['angles']
        Range_gates = self.simparams['Rangegates']
        Range_gates2 = self.simparams['Rangegatesfinal']
        ang = angles[beamnum]
        params = radardata.paramdict[ang]
        if type(timenum) is int:
            myfsize = 18
            titfsize = 22
            rng_arr = sp.zeros(0)
            Ne_true = sp.zeros(0)
            Te_true = sp.zeros(0)
            Ti_true = sp.zeros(0)
            Vi_true = sp.zeros(0)
            # get all of the original data into range bins
            for irng,rng in enumerate(Range_gates):
                #pdb.set_trace()
                cur_params = params[rng][timenum,:]
                cur_rgates = 1
                Ne_true = sp.append(Ne_true,10.0**cur_params[2])
                Te_true = sp.append(Te_true,cur_params[0]*cur_params[1])
                Ti_true = sp.append(Ti_true,cur_params[0])
                Vi_true = sp.append(Vi_true,cur_params[3])
                rng_arr = sp.append(rng_arr,sp.ones(cur_rgates)*rng)
            cur_fit = fittedarray[timenum,beamnum]
            cur_cov = fittederror[timenum,beamnum]

            Ne_fit = cur_fit[:,2]
            Te_fit = cur_fit[:,1]
            Ti_fit = cur_fit[:,0]
            Vi_fit = cur_fit[:,3]

            Ne_error = sp.sqrt(cur_cov[:,2,2])
            Te_error = sp.sqrt(cur_cov[:,1,1])
            Ti_error = sp.sqrt(cur_cov[:,0,0])
            Vi_error = sp.sqrt(cur_cov[:,3,3])

            altfit = Range_gates2*sp.sin(ang[1]*d2r)
            altori = rng_arr*sp.sin(ang[1]*d2r)
            fig = plt.figure(figsize=(15.5,8))
            #Ne plot
            plt.subplot(1,4,1)
            plt.errorbar(Ne_fit,altfit,xerr=Ne_error,fmt='bo',label=r'Fitted')
            plt.hold(True)
            plt.plot(Ne_true,altori,'go',label=r'Original')
            plt.xscale('log')
            plt.xlabel(r'Log $N_e$',fontsize=myfsize)
            plt.ylabel(r'Alt km',fontsize=myfsize)
            plt.grid(True)
            plt.legend(loc='upper right')
            # Te plot
            plt.subplot(1,4,2)
            plt.errorbar(Te_fit,altfit,xerr=Te_error,fmt='bo',label=r'Fitted')
            plt.hold(True)
            plt.plot(Te_true,altori,'go',label=r'Original')
            plt.grid(True)
            plt.xlabel(r'$T_e$ in K',fontsize=myfsize)
            plt.legend(loc='upper right')
            plt.xlim(0,4000)
            # Ti plot
            plt.subplot(1,4,3)
            plt.errorbar(Ti_fit,altfit,xerr=Vi_error,fmt='bo',label=r'Fitted')
            plt.hold(True)
            plt.plot(Ti_true,altori,'go',label=r'Original')
            plt.grid(True)
            plt.xlabel(r'$T_i$ in K',fontsize=myfsize)
            plt.xlim(0,4000)
            plt.legend(loc='upper right')
            # Vi plot
            plt.subplot(1,4,4)
            plt.errorbar(Vi_fit,altfit,xerr=Ti_error,fmt='bo',label=r'Fitted')
            plt.hold(True)
            plt.plot(Vi_true,altori,'go',label=r'Original')
            plt.grid(True)
            plt.xlabel(r'$V_i$ in K',fontsize=myfsize)
            plt.xlim(-5e2,5e2)
            plt.legend(loc='upper right')


            plt.suptitle(r'Fitted and Actual Parameters for Beam Az {0:.2f} El {1:.2f}'.format(ang[0],ang[1]),fontsize=titfsize)
            figname = 'Beam{0:d}.png'.format(beamnum)
            if figsdir==None:
                plt.savefig(figname)
            else:
                plt.savefig(os.path.join(figsdir,figname))


def default_fit_func(x,y_acf,amb_func,amb_dict,sensdict, rm1,rm2,p2,npts,numtype):
    """Fitter function that takes the difference between the given data
    and the spectrum given the current paramters. Used with scipy.optimize.leastsq."""
    ti = x[0]
    te = x[1]
    Ne = x[2]

    if len(x)>3:
        Vi = x[3]
    else:
        Vi = 0.0

    if te<0:
        te=-te
    if ti <0:
        ti=-ti
    if Ne<=0:
        Ne=-Ne
    po = sp.log10(Ne)
    tr = te/ti
    myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
    (omeg,cur_spec) = myspec.getSpectrum(ti, tr, po, rm1, rm2, p2)

    # Add Doppler
    Fd = -2.0*Vi/sensdict['lamb']
    omegnew = omeg-Fd
    omegnew = omeg-Fd
    fillspot = sp.argmax(omeg)
    fillval = cur_spec[fillspot]
    cur_spec =scipy.interpolate.interp1d(omeg,cur_spec,bounds_error=0,fill_value=fillval)(omegnew)
    # Create spectrum guess
    (tau,acf) = spect2acf(omeg,cur_spec)


    # apply ambiguity function
    tauint = amb_dict['Delay']
    acfinterp = sp.zeros(len(tauint),dtype=numtype)
    acfinterp.real =scipy.interpolate.interp1d(tau,acf.real,bounds_error=0)(tauint)
    acfinterp.imag =scipy.interpolate.interp1d(tau,acf.imag,bounds_error=0)(tauint)
    # Apply the lag ambiguity function to the data
    guess_acf = sp.zeros(amb_dict['Wlag'].shape[0],dtype=sp.complex128)
    for i in range(amb_dict['Wlag'].shape[0]):
        guess_acf[i] = sp.sum(acfinterp*amb_dict['Wlag'][i])
    # scale the guess acf
#    if (te==1e3)and (ti==1e3):
#        pdb.set_trace()

    guess_acf = guess_acf*Ne/((1+tr))/guess_acf[0].real
    # fit to spectrums
    spec_interm = scfft.fft(guess_acf,n=len(cur_spec))
    spec_final = spec_interm.real
    y_interm = scfft.fft(y_acf,n=len(spec_final))
    y = y_interm.real
    return y-spec_final




if __name__== '__main__':
    """ This is a test for the fitter class"""
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Test')
    t1 = time.time()
    IPP = .0087
#    angles = [(90,85),(90,84),(90,83),(90,82),(90,81)]
    angles = [(90,85)]
    ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
    t_int = 8.7*len(angles)
    pulse = sp.ones(14)
    rng_lims = [200,550]
    ioncont = MakeTestIonoclass(testv=True,testtemp=True)
    time_lim = t_int
    sensdict = sensconst.getConst('risr',ang_data)
    sensdict['Tsys']=0.1#reduce noise
    radardata = RadarData(ioncont,sensdict,angles,IPP,t_int,time_lim,pulse,rng_lims)
    timearr = sp.linspace(0,t_int,10)
    curint_time = IPP*100*len(angles)
    (DataLags,NoiseLags) = radardata.processdata(timearr,curint_time)
    simparams = radardata.simparams.copy()

    simparams['SUMRULE'] = sp.array([[-2,-3,-3,-4,-4,-5,-5,-6,-6,-7,-7,-8,-8,-9],[1,1,2,2,3,3,4,4,5,5,6,6,7,7]])
    simparams['amb_dict'] = make_amb(sensdict['fs'],30,sensdict['t_s']*len(pulse),len(pulse))
    curfitter =  FitterBasic(DataLags,NoiseLags,radardata.sensdict,simparams)
    Ne = curfitter.fitNE()
    (fittedarray,fittederror) = curfitter.fitdata()
    curfitter.plotbeams(0,radardata,fittedarray,fittederror,figsdir =testpath )
    plt.show(False)
    t2 = time.time()
    print((t2-t1))