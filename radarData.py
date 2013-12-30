#!/usr/bin/env python
"""
Created on Mon Dec 16 23:32:43 2013

@author: Bodangles
"""

import numpy as np
import scipy as sp
import scipy.optimize
import pdb
import time

from IonoContainer import IonoContainer, MakeTestIonoclass
from ISSpectrum import ISSpectrum
from physConstants import *
from sensorConstants import *
class RadarData(object):
    """ This class will will take the ionosphere class and create radar data both
    at the IQ and fitted level.  
    
    Variables
    simparams - A dictionary that holds simulation parameters the keys are the 
        following
        'angles': Angles (in degrees) for the simulation.  The az and el angles
            are stored in a list of tuples [(az_0,el_0),(az_1,el_1)...] 
        'IPP' : Interpulse period in seconds, this is the IPP from position to 
            position.  In other words if a IPP is 10 ms and there are 10 beams it will
            take 100 ms to go through all of the beams.
        'Tint'  This is the integration time in seconds.
        'Timevec': This is the time vector for each frame.  It is referenced to
            the begining of the frame.  Also in seconds
        'Pulse': The shape of the pulse.  This is a numpy array.
        'TimeLim': The length of time that the experiment will run, which cooresponds 
            to how many pulses per position are used.
    sensdict - A dictionary that holds the sensor parameters.
    datadict: This is dictionary that holds the data.  The dictionary keys are 
        ints based off of the positions of the beams based of of the list of 
        angles in the simparams dict.  The data is going to be in pulse vs 
        sample array.
    fittedarray: This is a numpy array, it is aranged NtxNbxNrxNpr, a Nt is 
        number of times, Nb is number of beams, Nr is number of range gates and 
        Npr is number of parameters.
    fittederror: This is the fitted error and is a similar set up to fittedarray.
        It's is a numpy array, it is aranged NtxNbxNrxNprxNpr, a Nt is 
        number of times, Nb is number of beams, Nr is number of range gates and 
        Npr is number of parameters.
        
    """
    def __init__(self,ionocont,sensdict,angles,IPP,Tint,time_lim, pulse,rng_lims):
        """This function will create an instance of the RadarData class.  It will
        take in the values and create the class and make raw IQ data.
        Inputs:
            sensdict - A dictionary of sensor parameters
            angles - A list of tuples which the first position is the az angle
                and the second position is the el angle.
            IPP - The interpulse period in seconds represented as a float.
            Tint - The integration time in seconds as a float.  This will be the 
            integration time of all of the beams.
            time_lim - The length of time of the simulation the number of time points
                will be calculated.
            pulse - A numpy array that represents the pulse shape.
            rng_lims - A numpy array of length 2 that holds the min and max range
                that the radar will cover."""
        # Initial params
        rng_gates = np.arange(rng_lims[0],rng_lims[1],sensdict['t_s']*v_C_0*1e-3)
        self.Ionocont = ionocont
        self.simparams =   {'IPP':IPP,'angles':angles,'TimeLim':time_lim,'Pulse':pulse,\
            'Timevec':np.arange(0,time_lim,Tint),'Tint':Tint,'Rangegates':rng_gates}
        N_times = len(self.simparams['Timevec'])
        N_params = 3
        N_range = len(rng_gates) 
        N_angles = len(angles)
        sensdict['RG'] = rng_gates
        self.sensdict = sensdict
        #self.datadict = dict()
        self.paramdict = dict()
        self.fittedarray = np.zeros((N_times,N_angles,N_range,N_params))
        self.fittederror = np.zeros((N_times,N_angles,N_range,N_params,N_params))
        firstcode = True
        for icode in np.arange(N_angles):
            outdata = self.__makeData__(icode)
            if firstcode:
                (Nr,Np) = outdata.shape
                self.datadict = np.zeros((N_angles,Nr,Np),dtype=np.complex128)
                
            self.datadict[icode]=outdata
            #[fitted,error] = self.fitdata()
            #self.fitteddict[beamangles[icode]]
            
    def __makeData__(self,beamcode):
        """This is an internal method that is used by the constructor function to
        create I\Q data for each beam.
        Inputs:
            self - The RadarData object.
            beamcode - This is simply an integer to select which beam will have
                data created for it.
        Outputs:
            Outdata - This will be the raw I\Q data
            """
        range_gates = self.simparams['Rangegates']
        centangles = self.simparams['angles'][beamcode]
        beamwidths = self.sensdict['BeamWidth']
        pulse = self.simparams['Pulse']
        sensdict = self.sensdict
        (omeg,specs_dict,params_dict)= self.Ionocont.makespectrums(range_gates,centangles,beamwidths,self.sensdict)
        # This is the time vector the data is changing its parameters under.        
        data_time_vec = self.Ionocont.Time_Vector
        self.paramdict[centangles] = params_dict
        # This is the IPP for each position, this will determine howmany pulses will be avalible for that position
        PIPP = len(self.simparams['angles'])*self.simparams['IPP']
        
        pulse_add = beamcode*self.simparams['IPP']
        time_lim = self.simparams['TimeLim']
        
        # determine the number of pulses per time period that will be running
        pulse_times = np.arange(pulse_add,time_lim,PIPP)
        N_pulses = sp.zeros(data_time_vec.shape)
        time_lim_vec = np.append(data_time_vec,time_lim)
        time_lim_vec = time_lim_vec[1:]
        for itime in np.arange(len(N_pulses)):
            pulse_log = pulse_times<time_lim_vec[itime]
            N_pulses[itime] = pulse_log.sum()
            pulse_times = pulse_times[~pulse_log]
            
        
        lp_pnts = len(pulse)
        samp_num = np.arange(lp_pnts)
        isamp = 0
        N_rg = len(range_gates)# take the size 
        N_samps = N_rg +lp_pnts-1
        Np = N_pulses.sum()
#        out_data = sp.zeros((N_samps,Np))+1j*sp.zeros((N_samps,Np))   
        out_data = sp.zeros((N_samps,Np),dtype=sp.complex128)
        # go through the spectrums at each range gate
        for isamp in np.arange(len(range_gates)):
            range = range_gates[isamp]
            cur_pnts = samp_num+isamp
            spect_ar = specs_dict[range]
            params_ar = params_dict[range]
            range_m = range*1e3
            if spect_ar.ndim ==3:
                (Nloc,Ndtime,speclen) = spect_ar.shape
                ipulse = 0
                for itimed in np.arange(Ndtime):
                   # pdb.set_trace()
                    for i_curtime in np.arange(N_pulses[itimed]): 
                        for iloc in np.arange(Nloc):
                            cur_spec = spect_ar[iloc,itimed]
                            cur_filt = np.sqrt(np.fft.ifftshift(cur_spec))
                            cur_params = params_ar[iloc,itimed]
                            
                            # get the plasma parameters
                            Ti = cur_params[0]
                            Tr = cur_params[1]
                            Te = Ti*Tr
                            N_e = 10**cur_params[2]                        
                            #calculated the power at each point
                            
                            debyel = np.sqrt(v_epsilon0*v_Boltz*Te/(v_epsilon0**2*N_e))
                            pow_num = sensdict['Pt']*sensdict['G']*v_C_0*sensdict['lamb']**2
                            pow_den = 2*16*np.pi**2*range_m**2
                            rcs = v_electron_rcs*N_e/((1+sensdict['k']**2*debyel**2)*(1+sensdict['k']**2*debyel**2+Tr))
                            pow_all = pow_num*rcs/pow_den
                            cur_mag = np.sqrt(pow_all)
                            cur_pulse_data = MakePulseData(pulse,cur_filt)
                            # find the varience of the complex data
                            cur_var = np.sum(np.abs(cur_pulse_data)**2)/len(cur_pulse_data)
                            cur_pulse_data = (cur_mag/np.sqrt(cur_var))*cur_pulse_data
                            
                            # This needs to be changed to weighting from the beam pattern
                            cur_pulse_data = cur_pulse_data/Nloc
                            out_data[cur_pnts,ipulse] = cur_pulse_data+out_data[cur_pnts,ipulse]
                        ipulse+=1
        return(out_data)               
            
    def fitdata(self,icode):
        """ """
        angles = self.simparams['angles']
        n_beams = len(angles)
        rawIQ = self.datadict[icode]
        IPP = self.simparams['IPP']
        timelim = self.simparams['TimeLim']
        PIPP = n_beams*IPP
        beg_time = icode*IPP
        Pulse_shape =  self.simparams['Pulse']
        plen = len(Pulse_shape)
        sensdict = self.sensdict
        npnts = 64
        # need to change this should not reference assumptions
        rm1 = 16# atomic weight of species 1
        rm2 = 1# atomic weight os species 2
        p2 =0 #
        
        # loop to figure out all of the pulses in a time period
        pulse_times = np.arange(beg_time,timelim,PIPP)
        
        int_timesbeg = self.simparams['Timevec']
        int_times = np.append(int_timesbeg,timelim)
        int_timesend = int_times[1:]
        numpast = 0
        Pulselims = np.zeros((len(int_timesbeg),2))
        N_pulses = np.zeros_like(int_timesbeg)
        # loop to determine what pulses are kep for each period         
        for itime in np.arange(len(int_timesbeg)):
            Pulselims[itime,0] = numpast
            keep_pulses = (pulse_times>=int_timesbeg[itime]) & (pulse_times<int_timesend[itime])
            N_pulses[itime] = keep_pulses.sum()
            numpast+= N_pulses[itime]
            Pulselims[itime,1] = numpast
            # loop for fitting
            cur_raw = rawIQ[:,Pulselims[itime,0]:Pulselims[itime,1]]
            out_lags = CenteredLagProduct(cur_raw,plen)
            (N_gates,N_lags) = out_lags.shape
            for irng in np.arange(N_gates):
                rangem=sensdict['RG'][irng]*1e3
                
                d_func = (out_lags[irng], Pulse_shape,rm1,rm2,p2,sensdict,rangem,npnts,N_pulses[itime]) 
                x_0 = Init_vales()
                try:                
                    (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fit_fun,x0=x_0,args=d_func,full_output=True)
                                
                    self.fittedarray[itime,icode,irng] = x
                    if cov_x == None:
                        self.fittederror[itime,icode,irng] = np.ones((len(x_0),len(x_0)))*float('nan')
                    else:
                        self.fittederror[itime,icode,irng] = cov_x*(infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(x_0))
                except TypeError:
                    pdb.set_trace()
                
    def fitalldata(self):
        """ """
        for ibeam in np.arange(len(self.simparams['angles'])):
            self.fitdata(ibeam)
            
    def reconstructdata(self):       
        """ """
def MakePulseData(pulse_shape, filt_freq, delay=16):
    """ This function will create a pulse width of data shaped by the filter that who's frequency
        response is passed as the parameter filt_freq.  The pulse shape is delayed by the parameter
        delay into the data
    """
    npts = len(filt_freq)
    #noise_vec = np.random.randn(npts)+1j*np.random.randn(npts);# make a noise vector
    noise_vec = np.random.randn(npts).astype(complex)
    mult_freq = filt_freq*noise_vec
    data = np.fft.ifft(mult_freq)
    data_out = pulse_shape*data[delay:(delay+len(pulse_shape))]
    return data_out

def CenteredLagProduct(rawbeams,N =14):
    
    # It will be assumed the data will be range vs pulses    
    (Nr,Np) = rawbeams.shape    
    
    # Make masks for each piece of data
    arex = np.arange(0,N/2.0,0.5);
    arback = np.array([-np.int(np.floor(k)) for k in arex]);
    arfor = np.array([np.int(np.ceil(k)) for k in arex]) ;
    
    # figure out how much range space will be kept
    sp = np.max(abs(arback));
    ep = Nr- np.max(arfor);
    rng_ar_all = np.arange(sp,ep);
    #acf_cent = np.zeros((ep-sp,N))*(1+1j)
    acf_cent = np.zeros((ep-sp,N),dtype=np.complex128)
    for irng in  np.arange(len(rng_ar_all)):
        rng_ar1 =np.int(rng_ar_all[irng]) + arback
        rng_ar2 = np.int(rng_ar_all[irng]) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = np.conj(rawbeams[rng_ar1,:])*rawbeams[rng_ar2,:]
        acf_ave = np.sum(acf_tmp,1)
        acf_cent[irng,:] = acf_ave# might need to transpose this
    return acf_cent    
    
def Init_vales():
    return np.array([1000.0,1000.0,10**11])
    
def fit_fun(x,y_acf,amb_func,rm1,rm2,p2,sensdict,range,npts,Npulses):
    
    ti = x[0]
    te = x[1]
    Ne = x[2]    
    if te<0:
        te=-te
    if ti <0:
        ti=-ti
    po = np.log10(Ne)
    tr = te/ti
    myspec = ISSpectrum(nspec = npts-1,sampfreq=sensdict['fs'])
    (omeg,cur_spec) = myspec.getSpectrum(ti, tr, po, rm1, rm2, p2)    
    
    # form the acf
    guessacf = np.fft.ifft(np.fft.ifftshift(cur_spec))
    L_amb = len(amb_func)
    full_amb = np.concatenate((amb_func,np.zeros(len(guessacf)-L_amb)))
    acf_mult = guessacf*full_amb
    # Add the change in power from the sensor
    pow_num = sensdict['Pt']*sensdict['G']*v_C_0*sensdict['lamb']**2*Npulses
    pow_den = 2*16*np.pi**2*range**2
    rcs = v_electron_rcs*Ne/((1+tr))    
    # multiply the power
    acf_mult = (acf_mult/np.abs(acf_mult[0]))*rcs*pow_num/pow_den
    spec_interm = np.fft.fft(acf_mult)
    spec_final = spec_interm.real
    y_interm = np.fft.fft(y_acf,n=len(spec_final))
    y = y_interm.real    
    #y = y/np.sum(y)
    return y-spec_final
    
if __name__== '__main__':
    t1 = time.time()
    IPP = .0087
    #angles = [(5,85),(5,84),(5,83),(5,82),(5,81)]
    angles = [(5,85)]    
    t_int = 8.7*len(angles)
    pulse = np.ones(14)
    rng_lims = [250,500]
    ioncont = MakeTestIonoclass()
    time_lim = t_int
    
    radardata = RadarData(ioncont,AMISR,angles,IPP,t_int,time_lim,pulse,rng_lims)
    radardata.fitalldata()
    t2 = time.time()
    print(t2-t1)