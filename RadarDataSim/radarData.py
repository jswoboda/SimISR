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
from const.physConstants import v_C_0, v_Boltz
import const.sensorConstants as sensconst
from utilFunctions import CenteredLagProduct, MakePulseData
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
    
    def __init__(self,ionocont,sensdict,angles,IPP,Tint,time_lim, pulse,rng_lims,noisesamples =28,noisepulses=100,npts = 128,type=0):
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
        self.paramdict = dict()
        self.lagarray = np.zeros((N_times,N_angles,N_range,len(pulse)),dtype=np.complex128)
        self.noiselag = np.zeros((N_times,N_angles,len(pulse)),dtype=np.complex128)
        self.fittedarray = np.zeros((N_times,N_angles,N_range,N_params))
        self.fittederror = np.zeros((N_times,N_angles,N_range,N_params,N_params))
        # determine the type
        if type ==0:
            print "All spectrums being created"
            (omeg,self.allspecs,npts) = self.Ionocont.makeallspectrums(sensdict,npts)
        elif type ==1:
            print "All spectrums created already"
            omeg = self.Ionocont.Param_Names
            self.allspecs = self.Ionocont.Param_List
            npts = len(omeg)
            
        firstcode = True
        print('\nData Now being created.')
        for icode in np.arange(N_angles):
            (outdata,noisedata) = self.__makeBeam__(icode,omeg,npts)
            if firstcode:
                (Np,Nr) = outdata.shape
                (NNP,NNr) = noisedata.shape
                self.rawdata = np.zeros((N_angles,Np,Nr),dtype=np.complex128)
                self.rawnoise = np.zeros((N_angles,NNP,NNr),dtype=np.complex128)
                firstcode = False
            self.rawdata[icode]=outdata
            self.rawnoise[icode] = noisedata
            print('\tData for Beam {0:d} of {1:d} created.'.format(icode,N_angles))
            
    def __makeBeam__(self,beamcode,omeg,npts):

        range_gates = self.simparams['Rangegates']
        centangles = self.simparams['angles'][beamcode]
        #beamwidths = self.sensdict['BeamWidth']
        pulse = self.simparams['Pulse']
        sensdict = self.sensdict
        # This is the time vector the data is changing its parameters under.        
        data_time_vec = self.Ionocont.Time_Vector
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
        rho = self.Ionocont.Sphere_Coords[:,0]
        Az = self.Ionocont.Sphere_Coords[:,1]
        El = self.Ionocont.Sphere_Coords[:,2]
        rng_len=self.sensdict['t_s']*v_C_0/1000.0
        (Nloc,Ndtime,speclen) = self.allspecs.shape
        weight = self.sensdict['ArrayFunc'](Az,El,centangles[0],centangles[1],sensdict['Angleoffset'])
        params_dict = {}
        # go through the spectrums at each range gate
        for isamp in np.arange(len(range_gates)):
            range = range_gates[isamp]
            
            range_m = range*1e3
            rnglims = [range-rng_len/2.0,range+rng_len/2.0]
            rangelog = (rho>=rnglims[0])&(rho<rnglims[1])
            cur_pnts = samp_num+isamp
                        
            if np.sum(rangelog)==0:
                pdb.set_trace()
            #create the weights and weight location based on the beams pattern.
            weight_cur =weight[rangelog]
           # pdb.set_trace()
            weight_cur = weight_cur/weight_cur.sum()
            
            specsinrng = self.allspecs[rangelog]
            specsinrng = specsinrng*np.tile(weight_cur[:,np.newaxis,np.newaxis],(1,Ndtime,speclen))            
            specsinrng = specsinrng.sum(0)
            ipulse = 0
            #pdb.set_trace()
            for itimed in np.arange(Ndtime):       
                for i_curtime in np.arange(N_pulses[itimed]):
                    cur_spec = specsinrng[itimed]
                    
                    cur_filt = np.sqrt(np.fft.ifftshift(cur_spec))                          
                    
                    #calculated the power at each point                            
                    pow_num = sensdict['Pt']*sensdict['Ksys'][beamcode]*sensdict['t_s'] # based off new way of calculating
                    pow_den = range_m**2
                    # create data
                    cur_pulse_data = MakePulseData(pulse,cur_filt)
                    
                    # find the varience of the complex data
                    cur_pulse_data = cur_pulse_data*np.sqrt(pow_num/pow_den)
                    cur_pulse_data = cur_pulse_data
                    out_data[ipulse,cur_pnts] = cur_pulse_data+out_data[ipulse,cur_pnts]
                    ipulse+=1
            params_dict[range] = self.Ionocont.getclosestsphere(np.array([range,centangles[0],centangles[1]]))[0]
        # Noise spectrums
        Noisepwr =  v_Boltz*sensdict['Tsys']*sensdict['BandWidth']
        Noise = np.sqrt(Noisepwr/2)*(np.random.randn(Np,N_samps).astype(complex)+1j*np.random.randn(Np,N_samps).astype(complex))
        noisesamples = np.sqrt(Noisepwr/2)*(np.random.randn(NNp,NNs).astype(complex) + 1j*np.random.randn(NNp,NNs).astype(complex))
        self.paramdict[centangles] = params_dict
        return(out_data +Noise,noisesamples)   
    #%% Trash this    
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
   #%% Processing 
    def processdata(self,timevec,inttime,lagfunc=CenteredLagProduct):
        """ This will perform the the data processing and create the ACF estimates 
        for both the data and noise.
        Inputs:
        timevec - A numpy array of times in seconds where the integration will begin.
        inttime - The integration time in seconds.
        lagfunc - A function that will make the desired lag products.
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
                outdata[inum,ibeam] = lagfunc(curdata,Nlag)
                outnoise[inum,ibeam] = lagfunc(curnoisedata,Nlag)
        DataLags = {'ACF':outdata,'Pow':outdata[:,:,:,0].real,'Pulses':pulses,'Time':timemat}  
        NoiseLags = {'ACF':outnoise,'Pow':outnoise[:,:,:,0].real,'Pulses':pulsesN,'Time':timemat}     
        return(DataLags,NoiseLags)
        
           
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