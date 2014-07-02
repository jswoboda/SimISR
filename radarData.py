#!/usr/bin/env python
"""
radarData.py
This file holds the RadarData class that hold the radar data and processes it.

@author: Bodangles
"""

import numpy as np
import scipy as sp
import time
import pdb
from IonoContainer import IonoContainer, MakeTestIonoclass
from const.physConstants import *
import const.sensorConstants as sensconst

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
    rawdata: This is a NbxNpxNr numpy array that holds the raw IQ data. 
    rawnoise: This is a NbxNpxNr numpy array that holds the raw noise IQ data. 
       
    """
    def __init__(self,ionocont,sensdict,angles,IPP,Tint,time_lim, pulse,rng_lims,noisesamples =28,noisepulses=100):
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
            'Timevec':np.arange(0,time_lim,Tint),'Tint':Tint,'Rangegates':rng_gates,\
            'Noisesamples': noisesamples,'Noisepulses':noisepulses}
        N_times = len(self.simparams['Timevec'])
        N_params = 3
        N_range = len(rng_gates) 
        N_angles = len(angles)
        sensdict['RG'] = rng_gates
        self.sensdict = sensdict
        #self.datadict = dict()
        self.paramdict = dict()
        self.lagarray = np.zeros((N_times,N_angles,N_range,len(pulse)),dtype=np.complex128)
        self.noiselag = np.zeros((N_times,N_angles,len(pulse)),dtype=np.complex128)
        self.fittedarray = np.zeros((N_times,N_angles,N_range,N_params))
        self.fittederror = np.zeros((N_times,N_angles,N_range,N_params,N_params))
        firstcode = True
        print('\nData Now being created.')
        for icode in np.arange(N_angles):
            (outdata,noisedata) = self.__makeData__(icode)
            if firstcode:
                (Np,Nr) = outdata.shape
                (NNP,NNr) = noisedata.shape
                self.rawdata = np.zeros((N_angles,Np,Nr),dtype=np.complex128)
                self.rawnoise = np.zeros((N_angles,NNP,NNr),dtype=np.complex128)
                firstcode = False
            self.rawdata[icode]=outdata
            self.rawnoise[icode] = noisedata
            print('\tData for Beam {0:d} of {1:d} created.'.format(icode,N_angles))
            
            
    def __makeData__(self,beamcode):
        """This is an internal method that is used by the constructor function to
        create I\Q data for each beam.
        Inputs:
            self - The RadarData object.
            beamcode - This is simply an integer to select which beam will have
                data created for it.
        Outputs:
            Outdata - This will be the raw I\Q data.  It will be a numpy array 
                of size NbxNrxNpu, Nb is number of beams, Nr is number of range
                gates and Npu is number of pulses.
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
        # This is the IPP for each position, this will determine how many pulses will be avalible for that position
        PIPP = len(self.simparams['angles'])*self.simparams['IPP']
        
        pulse_add = beamcode*self.simparams['IPP']
        time_lim = self.simparams['TimeLim']
        # check for last pulse of all data and truncate number of pulses for sim
        # to this maximum.
        Nangle = len(self.simparams['angles'])
        pulse_add_last = (Nangle-1)*self.simparams['IPP']
        maxpall = np.floor((time_lim-pulse_add_last)/PIPP)
        
        NNs = self.simparams['Noisesamples']
        NNp = self.simparams['Noisepulses']
        # determine the number of pulses per time period that will be running
        pulse_times = np.arange(pulse_add,time_lim,PIPP)
        pulse_times = pulse_times[:maxpall]
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
        out_data = sp.zeros((Np,N_samps),dtype=sp.complex128)
        # go through the spectrums at each range gate
        for isamp in np.arange(len(range_gates)):
            range = range_gates[isamp]
            cur_pnts = samp_num+isamp
            spect_ar = specs_dict[range]
            range_m = range*1e3
            if spect_ar.ndim ==3:
                (Nloc,Ndtime,speclen) = spect_ar.shape
                ipulse = 0
                for itimed in np.arange(Ndtime):
                    for i_curtime in np.arange(N_pulses[itimed]): 
                        for iloc in np.arange(Nloc):
                            cur_spec = spect_ar[iloc,itimed]
                            cur_filt = np.sqrt(np.fft.ifftshift(cur_spec))                          
                       
                            #calculated the power at each point                            
                            pow_num = sensdict['Pt']*sensdict['Ksys'][beamcode]*sensdict['t_s'] # based off new way of calculating
                            pow_den = range_m**2
                            # create data
                            cur_pulse_data = MakePulseData(pulse,cur_filt)
                            
                            # find the varience of the complex data
                            cur_pulse_data = cur_pulse_data*np.sqrt(pow_num/pow_den)
                            # This needs to be changed to weighting from the beam pattern
                            cur_pulse_data = cur_pulse_data/np.sqrt(Nloc)
                            out_data[ipulse,cur_pnts] = cur_pulse_data+out_data[ipulse,cur_pnts]
                        ipulse+=1
        Noisepwr =  v_Boltz*sensdict['Tsys']*sensdict['BandWidth']
        Noise = np.sqrt(Noisepwr/2)*(np.random.randn(Np,N_samps).astype(complex)+1j*np.random.randn(Np,N_samps).astype(complex))
        noisesamples = np.sqrt(Noisepwr/2)*(np.random.randn(NNp,NNs).astype(complex) + 1j*np.random.randn(NNp,NNs).astype(complex))

        return(out_data +Noise,noisesamples)               
    
    def processdata(self,timevec,inttime):
        """ This will perform the the data processing and create the ACF estimates 
        for both the data and noise.
        Inputs:
        timevec - A numpy array of times in seconds where the integration will begin.
        inttime - The integration time in seconds.
        Outputs:
        DataLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' that holds
        the numpy arrays of the data.
        NoiseLags: A dictionary with keys 'Power' 'ACF','RG','Pulses' that holds
        the numpy arrays of the data.
        """
        #angles and number of angles        
        (Nbeams,Np,Ns) = self.rawdata.shape
        NNs = self.rawnoise.shape[2]
        IPP = self.simparams['IPP']
        PIPP = Nbeams*IPP
        rng = self.sensdict['RG']
        Nrange = len(rng)
        pulse = self.simparams['Pulse']
        Nlag = len(pulse)
        npperint = np.ceil(inttime/PIPP)
        
        # find the start periods for the pulses        
        startvec = np.floor(timevec/PIPP)
        endvec = startvec+npperint
        
        timelogic = endvec<=Np
        startvec = startvec[timelogic]
        endvec = endvec[timelogic]
        Ntime = len(startvec)        
        timemat = np.zeros((Ntime,2))
        timemat[:,0] = startvec*PIPP
        timemat[:,1] = endvec*PIPP
                
        # create outdata will be Ntime x Nbeams x Nrange x Nlag
        outdata = np.zeros((Ntime,Nbeams,Nrange,Nlag),dtype=np.complex128)
        outnoise = np.zeros((Ntime,Nbeams,NNs-Nlag+1,Nlag),dtype=np.complex128)
        pulses = np.zeros((Ntime,Nbeams))
        pulsesN = np.zeros((Ntime,Nbeams))
        for ibeam in np.arange(Nbeams):
            for inum, istart in enumerate(startvec):
                iend = endvec[inum]
                curdata = self.rawdata[ibeam,istart:iend]
                curnoisedata = self.rawnoise[ibeam,istart:iend]
                pulses[inum,ibeam] = iend-istart;
                pulsesN[inum,ibeam] = iend-istart;
                outdata[inum,ibeam] = CenteredLagProduct(curdata.transpose(),Nlag)
                outnoise[inum,ibeam] = CenteredLagProduct(curnoisedata.transpose(),Nlag)
        DataLags = {'ACF':outdata,'Pow':outdata[:,:,:,0].real,'Pulses':pulses,'Time':timemat}  
        NoiseLags = {'ACF':outnoise,'Pow':outnoise[:,:,:,0].real,'Pulses':pulsesN,'Time':timemat}     
        return(DataLags,NoiseLags)
        
        
# Utility functions
        
def MakePulseData(pulse_shape, filt_freq, delay=16):
    """ This function will create a pulse width of data shaped by the filter that who's frequency
        response is passed as the parameter filt_freq.  The pulse shape is delayed by the parameter
        delay into the data. The noise vector that will be multiplied by the filter's frequency
        response will be zero mean complex white Gaussian noise with a power of 1. The user
        then will need to scale their filter to get the desired power out.
        Inputs:
            pulse_shape: A numpy array that holds the shape of the single pulse.
            filt_freq - a numpy array that holds the complex frequency response of the filter
            that will be used to shape the noise data.
            delay - The number of samples that the pulse will be delayed into the 
            array of noise data to avoid any problems with filter overlap.
    """
    npts = len(filt_freq)
    
    noise_vec = (np.random.randn(npts)+1j*np.random.randn(npts))/np.sqrt(2.0)# make a noise vector
    mult_freq = filt_freq*noise_vec
    data = np.fft.ifft(mult_freq)
    data_out = pulse_shape*data[delay:(delay+len(pulse_shape))]
    return data_out

def CenteredLagProduct(rawbeams,N =14):
    """ This function will create a centered lag product for each range using the
    raw IQ given to it.  It will form each lag for each pulse and then integrate 
    all of the pulses.
    Inputs: 
        rawbeams - This is a NsxNpu complex numpy array where Ns is number of 
        samples per pulse and Npu is number of pulses
        N - The number of lags that will be created, default is 14
    Output:
        acf_cent - This is a NrxNl complex numpy array where Nr is number of 
        range gate and Nl is number of lags.
    """
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
    return np.array([1000.0,1500.0,10**11])
    
# Main function test
    
if __name__== '__main__':
    """ Test function for the RadarData class."""
    t1 = time.time()
    IPP = .0087
    angles = [(90,85),(90,84),(90,83),(90,82),(90,81)]
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    t_int = 8.7*len(angles)
    pulse = np.ones(14)
    rng_lims = [250,500]
    ioncont = MakeTestIonoclass()
    time_lim = t_int
    sensdict = sensconst.getConst('risr',ang_data)
    radardata = RadarData(ioncont,sensdict,angles,IPP,t_int,time_lim,pulse,rng_lims)
    timearr = np.linspace(0,t_int,10)
    curint_time = IPP*100*len(angles)
    (DataLags,NoiseLags) = radardata.processdata(timearr,curint_time)

    t2 = time.time()
    print(t2-t1)