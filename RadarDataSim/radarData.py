#!/usr/bin/env python
"""
radarData.py
This file holds the RadarData class that hold the radar data and processes it.

@author: John Swoboda
"""

import os, time,inspect
import scipy.fftpack as scfft
import scipy as sp
import tables
import pdb
# My modules
from IonoContainer import IonoContainer
from const.physConstants import v_C_0, v_Boltz
from utilFunctions import CenteredLagProduct, MakePulseData,MakePulseDataRep, dict2h5
from makeConfigFiles import readconfigfile
import specfunctions
class RadarDataFile(object):
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
    def __init__(self,Ionodict,inifile, outdir,outfilelist=None):
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
       (sensdict,simparams) = readconfigfile(inifile)
       self.simparams = simparams
       N_angles = len(self.simparams['angles'])
       sensdict['RG'] = self.simparams['Rangegates']

       NNs = self.simparams['NNs']
       self.sensdict = sensdict
       Npall = sp.floor(self.simparams['TimeLim']/self.simparams['IPP'])
       Npall = sp.floor(Npall/N_angles)*N_angles
       Np = Npall/N_angles

       print "All spectrums created already"
       filetimes = Ionodict.keys()
       filetimes.sort()
       ftimes = sp.array(filetimes)
       simdtype = self.simparams['dtype']
       pulsetimes = sp.arange(Npall)*self.simparams['IPP']
       pulsefile = sp.array([sp.where(itimes-ftimes>=0)[0][-1] for itimes in pulsetimes])
       beams = sp.tile(sp.arange(N_angles),Npall/N_angles)

       pulsen = sp.repeat(sp.arange(Np),N_angles)

       if outfilelist is None:
            print('\nData Now being created.')

            Noisepwr =  v_Boltz*sensdict['Tsys']*sensdict['BandWidth']
            self.outfilelist = []
            for ifn, ifilet in enumerate(filetimes):
                outdict = {}
                ifile = Ionodict[ifilet]
                print('\tData from {0:d} of {1:d} being processed Name: {2:s}.'.format(ifn,len(filetimes),os.path.split(ifile)[1]))
                curcontainer = IonoContainer.readh5(ifile)
                pnts = pulsefile==ifn
                pt =pulsetimes[pnts]
                pb = beams[pnts]
                pn = pulsen[pnts].astype(int)
                outdict['RawData']= self.__makeTime__(pt,curcontainer.Time_Vector,
                    curcontainer.Sphere_Coords, curcontainer.Param_List,pb)
                outdict['NoiseData'] = sp.sqrt(Noisepwr/2)*(sp.random.randn(len(pn),NNs).astype(simdtype)+
                    1j*sp.random.randn(len(pn),NNs).astype(simdtype))
                outdict['Pulses']=pn
                outdict['Beams']=pb
                outdict['Time'] = pt
                newfn = os.path.join(outdir,'{0:d} RawData.h5'.format(ifn))
                self.outfilelist.append(newfn)
                dict2h5(newfn,outdict)
       else:
           self.outfilelist=outfilelist

#%% Make functions
    def __makeTime__(self,pulsetimes,spectime,Sphere_Coords,allspecs,beamcodes):

        range_gates = self.simparams['Rangegates']
        #beamwidths = self.sensdict['BeamWidth']
        pulse = self.simparams['Pulse']
        sensdict = self.sensdict
        pulse2spec = sp.array([sp.where(itimes-spectime>=0)[0][-1] for itimes in pulsetimes])
        Np = len(pulse2spec)
        lp_pnts = len(pulse)
        samp_num = sp.arange(lp_pnts)
        isamp = 0
        N_rg = len(range_gates)# take the size
        N_samps = N_rg +lp_pnts-1
        angles = self.simparams['angles']
        Nbeams = len(angles)
        rho = Sphere_Coords[:,0]
        Az = Sphere_Coords[:,1]
        El = Sphere_Coords[:,2]
        rng_len=self.sensdict['t_s']*v_C_0/1000.0
        (Nloc,Ndtime,speclen) = allspecs.shape
        simdtype = self.simparams['dtype']
        out_data = sp.zeros((Np,N_samps),dtype=simdtype)
        weights = {ibn:self.sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}


        specsused = sp.zeros((Ndtime,Nbeams,N_rg,speclen),dtype=allspecs.dtype)
        for istn, ist in enumerate(spectime):
            for ibn in range(Nbeams):
                print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
                weight = weights[ibn]
                for isamp in sp.arange(len(range_gates)):
                    range_g = range_gates[isamp]
                    range_m = range_g*1e3
                    rnglims = [range_g-rng_len/2.0,range_g+rng_len/2.0]
                    rangelog = (rho>=rnglims[0])&(rho<rnglims[1])
                    cur_pnts = samp_num+isamp

                    if sp.sum(rangelog)==0:
                        pdb.set_trace()
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()

                    specsinrng = allspecs[rangelog][istn]
                    specsinrng = specsinrng*sp.tile(weight_cur[:,sp.newaxis],(1,speclen))
                    cur_spec = specsinrng.sum(0)
                    specsused[istn,ibn,isamp] = cur_spec
                    cur_filt = sp.sqrt(scfft.ifftshift(cur_spec))
                    pow_num = sensdict['Pt']*sensdict['Ksys'][ibn]*sensdict['t_s'] # based off new way of calculating
                    pow_den = range_m**2
                    curdataloc = sp.where((pulse2spec==istn)&(beamcodes==ibn))[0]
                    # create data
                    cur_pulse_data = MakePulseDataRep(pulse,cur_filt,rep=len(curdataloc),numtype = simdtype)
                    cur_pulse_data = cur_pulse_data*sp.sqrt(pow_num/pow_den)
                    # the double slicing is a clever way to get this to work. curdataloc takes
                    # the points that are the same spectrum. Then slice for the range dimension with : operator
                    for idatn,idat in enumerate(curdataloc):
                        out_data[idat,cur_pnts] = cur_pulse_data[idatn]+out_data[idat,cur_pnts]
        # Noise spectrums
        Noisepwr =  v_Boltz*sensdict['Tsys']*sensdict['BandWidth']
        Noise = sp.sqrt(Noisepwr/2)*(sp.random.randn(Np,N_samps).astype(complex)+
            1j*sp.random.randn(Np,N_samps).astype(complex))
        return out_data +Noise
        #%% Processing
    def processdataiono(self,lagfunc=CenteredLagProduct):
        """ This will perform the the data processing and create the ACF estimates
        for both the data and noise but put it in an Ionocontainer.
        Inputs:
        timevec - A numpy array of times in seconds where the integration will begin.
        inttime - The integration time in seconds.
        lagfunc - A function that will make the desired lag products.
        Outputs:
        Ionocontainer- This is an instance of the ionocontainer class that will hold the acfs.
        """
        (DataLags,NoiseLags) = self.processdata(lagfunc)
        return lagdict2ionocont(DataLags,NoiseLags,self.sensdict,self.simparams,self.simparams['Timevec'])

    def processdata(self,lagfunc=CenteredLagProduct):
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
        timevec = self.simparams['Timevec']
        inttime = self.simparams['Tint']
        # Get array sizes
        file_list = self.outfilelist
        NNs = self.simparams['NNs']
        range_gates = self.simparams['Rangegates']
        N_rg = len(range_gates)# take the size
        pulse = self.simparams['Pulse']
        Nlag = len(pulse)
        N_samps = N_rg +Nlag-1
        simdtype = self.simparams['dtype']
        Ntime=len(timevec)
        Nbeams = len(self.simparams['angles'])

        # initialize output arrays
        outdata = sp.zeros((Ntime,Nbeams,N_rg,Nlag),dtype=simdtype)
        outnoise = sp.zeros((Ntime,Nbeams,NNs-Nlag+1,Nlag),dtype=simdtype)
        pulses = sp.zeros((Ntime,Nbeams))
        pulsesN = sp.zeros((Ntime,Nbeams))
        timemat = sp.zeros((Ntime,2))

        # initalize lists for stuff
        pulsen_list = []
        beamn_list = []
        time_list = []
        file_loclist = []

        # read in times
        for ifn, ifile in enumerate(file_list):
            h5file=tables.openFile(ifile)
            pulsen_list.append(h5file.get_node('/Pulses').read())
            beamn_list.append(h5file.get_node('/Beams').read())
            time_list.append(h5file.get_node('/Time').read())
            file_loclist.append(ifn*sp.ones(len(pulsen_list[-1])))
            h5file.close()

        pulsen = sp.hstack(pulsen_list).astype(int)
        beamn = sp.hstack(beamn_list).astype(int)
        ptimevec = sp.hstack(time_list).astype(int)
        file_loc = sp.hstack(file_loclist).astype(int)
        # run the time loop
        print("Forming ACF estimates")
        for itn,it in enumerate(timevec):
            print("\tTime {0:d} of {0:d}".format(itn,Ntime))
            # do the book keeping to determine locations of data within the files
            cur_tlim = (it,it+inttime)
            curcases = sp.logical_and(ptimevec>=cur_tlim[0],ptimevec<cur_tlim[1])
            if  not sp.any(curcases):
                print("\tNo pulses for time {0:d} of {0:d}, lagdata adjusted accordinly".format(itn,Ntime))
                outdata = outdata[:itn]
                outnoise = outnoise[:itn]
                pulses=pulses[:itn]
                pulsesN=pulsesN[:itn]
                timemat=timemat[:itn]
                continue
            pulseset = set(pulsen[curcases])
            poslist = [sp.where(pulsen==item)[0] for item in pulseset ]
            pos_all = sp.hstack(poslist)
            try:
                pos_all = sp.hstack(poslist)
                curfileloc = file_loc[pos_all]
            except:
                pdb.set_trace()
            curfiles = set(curfileloc)
            beamlocs = beamn[pos_all]

            timemat[itn,0] = ptimevec[pos_all].min()
            timemat[itn,1]=ptimevec[pos_all].max()
            curdata = sp.zeros((len(pos_all),N_samps),dtype = simdtype)
            curnoise = sp.zeros((len(pos_all),NNs),dtype = simdtype)
            # Open files and get required data
            # XXX come up with way to get open up new files not have to reread in data that is already in memory
            for ifn in curfiles:
                curfileit =  [sp.where(pulsen_list[ifn]==item)[0] for item in pulseset ]
                curfileitvec = sp.hstack(curfileit)
                ifile = file_list[ifn]
                h5file=tables.openFile(ifile)
                file_arlocs = sp.where(curfileloc==ifn)[0]
                curdata[file_arlocs] = h5file.get_node('/RawData').read().astype(simdtype)[curfileitvec]
                curnoise[file_arlocs] = h5file.get_node('/NoiseData').read().astype(simdtype)[curfileitvec]
                h5file.close()
            # After data is read in form lags for each beam
            for ibeam in range(Nbeams):
                print("\t\tBeam {0:d} of {0:d}".format(ibeam,Nbeams))
                beamlocstmp = sp.where(beamlocs==ibeam)[0]
                pulses[itn,ibeam] = len(beamlocstmp)
                pulsesN[itn,ibeam] = len(beamlocstmp)
                outdata[itn,ibeam] = lagfunc(curdata[beamlocstmp],Nlag)
                outnoise[itn,ibeam] = lagfunc(curnoise[beamlocstmp],Nlag)
        # Create output dictionaries and output data
        DataLags = {'ACF':outdata,'Pow':outdata[:,:,:,0].real,'Pulses':pulses,'Time':timemat}
        NoiseLags = {'ACF':outnoise,'Pow':outnoise[:,:,:,0].real,'Pulses':pulsesN,'Time':timemat}
        return(DataLags,NoiseLags)

#%% Make Lag dict to an iono container
def lagdict2ionocont(DataLags,NoiseLags,sensdict,simparams,time_vec):
    """This function will take the data and noise lags and create an instance of the
    Ionocontanier class. This function will also apply the summation rule to the lags.
    Inputs
    DataLags - A dictionary """
    # Pull in Location Data
    angles = simparams['angles']
    ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
    rng_vec = sensdict['RG']
    # pull in other data
    pulsewidth = sensdict['taurg']*sensdict['t_s']
    txpower = sensdict['Pt']
    Ksysvec = sensdict['Ksys']
    sumrule = simparams['SUMRULE']
    minrg = -sumrule[0].min()
    maxrg = len(rng_vec)-sumrule[1].max()
    Nrng2 = maxrg-minrg;
    rng_vec2 = sp.array([ sp.mean(rng_vec[irng+sumrule[0,0]:irng+sumrule[1,0]+1]) for irng in range(minrg,maxrg)])
    # Set up Coordinate list
    angtile = sp.tile(ang_data,(Nrng2,1))
    rng_rep = sp.repeat(rng_vec2,ang_data.shape[0],axis=0)
    coordlist=sp.zeros((len(rng_rep),3))
    [coordlist[:,0],coordlist[:,1:]] = [rng_rep,angtile]
    # set up the lags
    lagsData= DataLags['ACF']
    (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
    pulses = sp.tile(DataLags['Pulses'][:,:,sp.newaxis,sp.newaxis],(1,1,Nrng,Nlags))
    time_vec = time_vec[:Nt]

    # average by the number of pulses
    lagsData = lagsData/pulses
    lagsNoise=NoiseLags['ACF']
    lagsNoise = sp.mean(lagsNoise,axis=2)
    pulsesnoise = sp.tile(NoiseLags['Pulses'][:,:,sp.newaxis],(1,1,Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = sp.tile(lagsNoise[:,:,sp.newaxis,:],(1,1,Nrng,1))
    # subtract out noise lags
    lagsData = lagsData-lagsNoise
    rng3d = sp.tile(rng_vec[sp.newaxis,sp.newaxis,:,sp.newaxis],(Nt,Nbeams,1,Nlags)) *1e3
    ksys3d = sp.tile(Ksysvec[sp.newaxis,:,sp.newaxis,sp.newaxis],(Nt,1,Nrng,Nlags))
    lagsData = lagsData*rng3d*rng3d/(pulsewidth*txpower*ksys3d)

    # Apply summation rule
    # lags transposed from (time,beams,range,lag)to (range,lag,time,beams)
    lagsData = sp.transpose(lagsData,axes=(2,3,0,1))
    lagsDatasum = sp.zeros((Nrng2,Nlags,Nt,Nbeams),dtype=lagsData.dtype)

    for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
        for ilag in range(Nlags):
            lagsDatasum[irngnew,ilag] = lagsData[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)

    # Put everything in a parameter list
    Paramdata = sp.zeros((Nbeams*Nrng2,Nt,Nlags),dtype=lagsData.dtype)
    # transpose from (range,lag,time,beams) to (beams,range,time,lag)
    lagsDataSum = sp.transpose(lagsDatasum,axes=(3,0,2,1))
    curloc = 0
    for irng in range(Nrng2):
        for ibeam in range(Nbeams):
            Paramdata[curloc] = lagsDataSum[ibeam,irng]
            curloc+=1

    return IonoContainer(coordlist,Paramdata,times = time_vec,ver =1, paramnames=sp.arange(Nlags)*sensdict['t_s'])
#%% Testing

if __name__== '__main__':
    """ Test function for the RadarData class."""
    t1 = time.time()
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Test')
    inifile = os.path.join(testpath,'PFISRExample.pickle')
    (sensdict,simparams) = readconfigfile(inifile)
    testh5 = os.path.join(testpath,'testiono.h5')
    ioncont = IonoContainer.readh5(testh5)
    outfile = os.path.join(testpath,'testionospec.h5')

    ioncont.makespectruminstanceopen(specfunctions.ISRSspecmake,sensdict,simparams['numpoints']).saveh5(outfile)
    radardata = RadarDataFile({0.0:outfile},inifile,testpath)

    ionoout = radardata.processdataiono()
    ionoout.saveh5(os.path.join(testpath,'lags.h5'))
    t2 = time.time()
    print(t2-t1)