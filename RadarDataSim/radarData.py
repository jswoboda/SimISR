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
from utilFunctions import CenteredLagProduct,MakePulseDataRep, dict2h5,h52dict,readconfigfile, BarkerLag
import specfunctions
from analysisplots import plotspecsgen
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

       # differentiate between phased arrays and dish antennas
       if sensdict['Name'].lower() in ['risr','pfisr']:
           beams = sp.tile(sp.arange(N_angles),Npall/N_angles)
       else:
           # for dish arrays
            brate = simparams['beamrate']
            beams2 = sp.repeat(sp.arange(N_angles),brate)
            beam3 = sp.concatenate((beams2,beams2[:-1,::-1]))
            beams = sp.tile(beam3,Npall/len(beam3))

       pulsen = sp.repeat(sp.arange(Np),N_angles)
       pt_list = []
       pb_list = []
       pn_list = []
       fname_list = []
       self.datadir = outdir
       self.maindir = os.path.dirname(os.path.abspath(outdir))
       self.procdir =os.path.join(self.maindir,'ACF')
       self.procdir
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
                (rawdata,outdict['SpecsUsed'])= self.__makeTime__(pt,curcontainer.Time_Vector,
                    curcontainer.Sphere_Coords, curcontainer.Param_List,pb)
                Noise = sp.sqrt(Noisepwr/2)*(sp.random.randn(*rawdata.shape).astype(simdtype)+
                    1j*sp.random.randn(*rawdata.shape).astype(simdtype))
                outdict['AddedNoise'] =Noise
                outdict['RawData'] = rawdata+Noise
                outdict['RawDatanonoise'] = rawdata
                outdict['NoiseData'] = sp.sqrt(Noisepwr/2)*(sp.random.randn(len(pn),NNs).astype(simdtype)+
                    1j*sp.random.randn(len(pn),NNs).astype(simdtype))
                outdict['Pulses']=pn
                outdict['Beams']=pb
                outdict['Time'] = pt
                fname = '{0:d} RawData.h5'.format(ifn)
                newfn = os.path.join(self.datadir,fname)
                self.outfilelist.append(newfn)
                dict2h5(newfn,outdict)

                #Listing info
                pt_list.append(pt)
                pb_list.append(pb)
                pn_list.append(pn)
                fname_list.append(fname)
            infodict = {'Files':fname_list,'Time':pt_list,'Beams':pb_list,'Pulses':pn_list}
            dict2h5(os.path.join(outdir,'INFO.h5'),infodict)

       else:
           self.outfilelist=outfilelist


#%% Make functions
    def __makeTime__(self,pulsetimes,spectime,Sphere_Coords,allspecs,beamcodes):
        """This is will make raw radar data for a given time period and set of
        spectrums. This is an internal function called by __init__.
        Inputs-
        self - RadarDataFile object.
        pulsetimes - The time the pulses are sent out in reference to the spectrums.
        spectime - The times for the spectrums.
        Sphere_Coords - An Nlx3 array that holds the spherical coordinates of the spectrums.
        allspecs - An NlxNdtimexNspec array that holds the spectrums.
        beamcodes - A NBx4 array that holds the beam codes along with the beam location
        in az el along with a system constant in the array. """

        range_gates = self.simparams['Rangegates']
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

                    # This is a nearest neighbors interpolation for the spectrums in the range domain
                    if sp.sum(rangelog)==0:
                        minrng = sp.argmin(sp.absolute(range_g-rho))
                        rangelog[minrng] = True
                        #pdb.set_trace()
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()

                    specsinrng = allspecs[rangelog]
                    if specsinrng.ndim==3:
                        specsinrng=specsinrng[:,istn]
                    elif specsinrng.ndim==2:
                        specsinrng=specsinrng[istn]
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

                    for idatn,idat in enumerate(curdataloc):
                        out_data[idat,cur_pnts] = cur_pulse_data[idatn]+out_data[idat,cur_pnts]

        return (out_data,specsused)
        #%% Processing
    def processdataiono(self):
        """ This will perform the the data processing and create the ACF estimates
        for both the data and noise but put it in an Ionocontainer.
        Inputs:

        Outputs:
        Ionocontainer- This is an instance of the ionocontainer class that will hold the acfs.
        """
        (DataLags,NoiseLags) = self.processdata()
        return lagdict2ionocont(DataLags,NoiseLags,self.sensdict,self.simparams,DataLags['Time'])

    def processdata(self):
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

        NNs = self.simparams['NNs']
        range_gates = self.simparams['Rangegates']
        N_rg = len(range_gates)# take the size
        pulse = self.simparams['Pulse']
        Pulselen = len(pulse)
        N_samps = N_rg +Pulselen-1
        simdtype = self.simparams['dtype']
        Ntime=len(timevec)
        Nbeams = len(self.simparams['angles'])

        # Choose type of processing
        if self.simparams['Pulsetype'].lower() == 'barker':
            lagfunc=BarkerLag
            Nlag=1
        else:
            lagfunc=CenteredLagProduct
            Nlag=Pulselen
        # initialize output arrays
        outdata = sp.zeros((Ntime,Nbeams,N_rg,Nlag),dtype=simdtype)
        outaddednoise = sp.zeros((Ntime,Nbeams,N_rg,Nlag),dtype=simdtype)
        outnoise = sp.zeros((Ntime,Nbeams,NNs-Pulselen+1,Nlag),dtype=simdtype)
        pulses = sp.zeros((Ntime,Nbeams))
        pulsesN = sp.zeros((Ntime,Nbeams))
        timemat = sp.zeros((Ntime,2))

        # set up arrays that hold the location of pulses that are to be processed together
        infoname = os.path.join(self.datadir,'INFO.h5')
        if os.path.isfile(infoname):
            infodict =h52dict(infoname)
            flist =  infodict['Files']
            file_list = [os.path.join(self.datadir,i) for i in flist]
            pulsen_list = infodict['Pulses']
            beamn_list = infodict['Beams']
            time_list = infodict['Time']
            file_loclist = [ifn*sp.ones(len(ifl)) for ifn,ifl in enumerate(beamn_list)]
        else:
            file_list = self.outfilelist
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
        firstfile = True
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
            curaddednoise = sp.zeros((len(pos_all),N_samps),dtype = simdtype)
            curnoise = sp.zeros((len(pos_all),NNs),dtype = simdtype)
            # Open files and get required data
            # XXX come up with way to get open up new files not have to reread in data that is already in memory
            for ifn in curfiles:
                curfileit =  [sp.where(pulsen_list[ifn]==item)[0] for item in pulseset ]
                curfileitvec = sp.hstack(curfileit)
                ifile = file_list[ifn]
                h5file=tables.openFile(ifile)
                file_arlocs = sp.where(curfileloc==ifn)[0]
                curdata[file_arlocs] = h5file.get_node('/RawData').read().astype(simdtype)[curfileitvec].copy()
                if firstfile:
                    specsused = h5file.get_node('/SpecsUsed').read().astype(simdtype)
                    firstfile=False
                else:
                    specsusedtmp = h5file.get_node('/SpecsUsed').read().astype(simdtype)
                    specsused = sp.concatenate((specsused,specsusedtmp),axis=0)

                curaddednoise[file_arlocs] = h5file.get_node('/AddedNoise').read().astype(simdtype)[curfileitvec].copy()
                curnoise[file_arlocs] = h5file.get_node('/NoiseData').read().astype(simdtype)[curfileitvec].copy()
                h5file.close()
            # After data is read in form lags for each beam
            for ibeam in range(Nbeams):
                print("\t\tBeam {0:d} of {0:d}".format(ibeam,Nbeams))
                beamlocstmp = sp.where(beamlocs==ibeam)[0]
                pulses[itn,ibeam] = len(beamlocstmp)
                pulsesN[itn,ibeam] = len(beamlocstmp)
                outdata[itn,ibeam] = lagfunc(curdata[beamlocstmp].copy(),numtype=self.simparams['dtype'], pulse=pulse)
                outnoise[itn,ibeam] = lagfunc(curnoise[beamlocstmp].copy(),numtype=self.simparams['dtype'], pulse=pulse)
                outaddednoise[itn,ibeam] = lagfunc(curaddednoise[beamlocstmp].copy(),numtype=self.simparams['dtype'], pulse=pulse)
        # Create output dictionaries and output data
        DataLags = {'ACF':outdata,'Pow':outdata[:,:,:,0].real,'Pulses':pulses,'Time':timemat,'AddedNoiseACF':outaddednoise,'SpecsUsed':specsused}
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
    rng_vec = simparams['Rangegates']
    # pull in other data
    pulsewidth = len(simparams['Pulse'])*sensdict['t_s']
    txpower = sensdict['Pt']
    Ksysvec = sensdict['Ksys']
    sumrule = simparams['SUMRULE']
    rng_vec2 = simparams['Rangegatesfinal']
    minrg = -sumrule[0].min()
    maxrg = len(rng_vec)-sumrule[1].max()
    Nrng2 = len(rng_vec2)
    # Set up Coordinate list
    angtile = sp.tile(ang_data,(Nrng2,1))
    rng_rep = sp.repeat(rng_vec2,ang_data.shape[0],axis=0)
    coordlist=sp.zeros((len(rng_rep),3))
    [coordlist[:,0],coordlist[:,1:]] = [rng_rep,angtile]
    # set up the lags
    lagsData= DataLags['ACF'].copy()
    (Nt,Nbeams,Nrng,Nlags) = lagsData.shape
    rng3d = sp.tile(rng_vec[sp.newaxis,sp.newaxis,:,sp.newaxis],(Nt,Nbeams,1,Nlags)) *1e3
    ksys3d = sp.tile(Ksysvec[sp.newaxis,:,sp.newaxis,sp.newaxis],(Nt,1,Nrng,Nlags))
    radar2acfmult = rng3d*rng3d/(pulsewidth*txpower*ksys3d)
    pulses = sp.tile(DataLags['Pulses'][:,:,sp.newaxis,sp.newaxis],(1,1,Nrng,Nlags))
    time_vec = time_vec[:Nt]
    # average by the number of pulses
    lagsData = lagsData/pulses
    lagsNoise=NoiseLags['ACF'].copy()
    lagsNoise = sp.mean(lagsNoise,axis=2)
    pulsesnoise = sp.tile(NoiseLags['Pulses'][:,:,sp.newaxis],(1,1,Nlags))
    lagsNoise = lagsNoise/pulsesnoise
    lagsNoise = sp.tile(lagsNoise[:,:,sp.newaxis,:],(1,1,Nrng,1))

    # output debug plots

    if 'debug' in simparams.keys():
        debuginfo = simparams['debug']
        plotdir =debuginfo['plotdir']
        if debuginfo['PlotNoise']:
            addednoise = DataLags['AddedNoiseACF']/pulses
            addednoise = sp.mean(addednoise,axis=2)

            noiseacf = [addednoise[0,0],lagsNoise[0,0,0]]
            taun = sp.arange(Nlags)*sensdict['t_s']
            noisepltfname = os.path.join(plotdir,'NoiseACF.png')
            plotspecsgen(taun,noiseacf,[True,True],['Actual Noise','Noise Est'],noisepltfname,simparams['numpoints'])
        if debuginfo['PlotDatawon']:
            addednoise = DataLags['AddedNoiseACF']/pulses
            lagsDatapnt = lagsData-addednoise
            lagsDatapnt = lagsDatapnt*radar2acfmult
            beamnumn = debuginfo['beam']
            rnggaten = debuginfo['range']
            realspec = DataLags['SpecsUsed'].mean(0)
            lagsDatapnt = sp.transpose(lagsDatapnt,axes=(2,3,0,1))
            lagsDatasumpnt = sp.zeros((Nrng2,Nlags,Nt,Nbeams),dtype=lagsData.dtype)

            for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
                for ilag in range(Nlags):
                    lagsDatasumpnt[irngnew,ilag] = lagsDatapnt[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)
            lagsDatasumpnt = sp.transpose(lagsDatasumpnt,axes=(3,0,2,1))
            n= simparams['numpoints']
            fs = 1./sensdict['t_s']
            omegn = sp.arange(-sp.ceil((n-1.0)/2.0),sp.floor((n-1.0)/2.0+1))*(fs/(2*sp.ceil((n-1.0)/2.0)))
            taun = sp.arange(Nlags)*sensdict['t_s']
            baslist = [omegn,taun]
            acspec = realspec[beamnumn,rnggaten]
            acspec = acspec*simparams['numpoints']*lagsDatasumpnt[beamnumn,rnggaten,0,0].real/acspec.real.sum()
            specacf = [acspec,lagsDatasumpnt[beamnumn,rnggaten,0]]
            specpltfname = os.path.join(plotdir,'Specacfwon.png')
            plotspecsgen(baslist,specacf,[False,True],['Input Spec','Spec Est'],specpltfname,simparams['numpoints'])
        if debuginfo['PlotDatawn']:

            lagsDatapnt = lagsData.copy()
            lagsDatapnt = lagsDatapnt*radar2acfmult
            beamnumn = debuginfo['beam']
            rnggaten = debuginfo['range']
            realspec = DataLags['SpecsUsed'].mean(0)
            lagsDatapnt = sp.transpose(lagsDatapnt,axes=(2,3,0,1))
            lagsDatasumpnt = sp.zeros((Nrng2,Nlags,Nt,Nbeams),dtype=lagsData.dtype)

            for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
                for ilag in range(Nlags):
                    lagsDatasumpnt[irngnew,ilag] = lagsDatapnt[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)
            lagsDatasumpnt = sp.transpose(lagsDatasumpnt,axes=(3,0,2,1))
            n= simparams['numpoints']
            fs = 1./sensdict['t_s']
            omegn = sp.arange(-sp.ceil((n-1.0)/2.0),sp.floor((n-1.0)/2.0+1))*(fs/(2*sp.ceil((n-1.0)/2.0)))
            taun = sp.arange(Nlags)*sensdict['t_s']
            baslist = [omegn,taun]
            acspec = realspec[beamnumn,rnggaten]
            acspec = acspec*simparams['numpoints']*lagsDatasumpnt[beamnumn,rnggaten,0,0].real/acspec.real.sum()
            specacf = [acspec,lagsDatasumpnt[beamnumn,rnggaten,0]]
            specpltfname = os.path.join(plotdir,'Specacfon.png')
            plotspecsgen(baslist,specacf,[False,True],['Input Spec','Spec Est'],specpltfname,simparams['numpoints'])
    # subtract out noise lags
    lagsData = lagsData-lagsNoise
    # Use a basic formula to find the std of the lagss
    sigS = (sp.absolute(lagsData)+sp.absolute(lagsNoise))/sp.sqrt(pulses)
    sigS = sp.sqrt(varacf(lagsData)+sp.absolute(lagsNoise)**2)/sp.sqrt(pulses)
    # multiply the data and the sigma by inverse of the scaling from the radar
    lagsData = lagsData*radar2acfmult
    sigS = sigS*radar2acfmult

    # Apply summation rule
    # lags transposed from (time,beams,range,lag)to (range,lag,time,beams)
    lagsData = sp.transpose(lagsData,axes=(2,3,0,1))
    sigS = sp.transpose(sigS,axes=(2,3,0,1))
    lagsDatasum = sp.zeros((Nrng2,Nlags,Nt,Nbeams),dtype=lagsData.dtype)
    sigsSsum = sp.zeros((Nrng2,Nlags,Nt,Nbeams),dtype=sigS.dtype)
    for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
        for ilag in range(Nlags):
            lagsDatasum[irngnew,ilag] = lagsData[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)
            sigsSsum[irngnew,ilag] = sp.sqrt(sp.power(sigS[irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag],2).mean(axis=0))
    # Put everything in a parameter list
    Paramdata = sp.zeros((Nbeams*Nrng2,Nt,Nlags),dtype=lagsData.dtype)
    # Put everything in a parameter list
    Paramdatasig = sp.zeros((Nbeams*Nrng2,Nt,Nlags),dtype=sigsSsum.dtype)
    # transpose from (range,lag,time,beams) to (beams,range,time,lag)
    lagsDataSum = sp.transpose(lagsDatasum,axes=(3,0,2,1))
    sigsSsum = sp.transpose(sigsSsum,axes=(3,0,2,1))
    curloc = 0
    for irng in range(Nrng2):
        for ibeam in range(Nbeams):
            Paramdata[curloc] = lagsDataSum[ibeam,irng].copy()
            Paramdatasig[curloc] = sigsSsum[ibeam,irng].copy()
            curloc+=1
    ionodata = IonoContainer(coordlist,Paramdata,times = time_vec,ver =1, paramnames=sp.arange(Nlags)*sensdict['t_s'])
    ionosigs = IonoContainer(coordlist,Paramdatasig,times = time_vec,ver =1, paramnames=sp.arange(Nlags)*sensdict['t_s'])
    return (ionodata,ionosigs)

def varacf(acf):
    """This calculates the variances of an ACF. """
    axisvec =sp.roll( sp.arange(acf.ndim),1)
    acftrans = sp.transpose(acf,tuple(axisvec))
    acfvar = sp.zeros_like(acftrans)
    axisvec = sp.roll(axisvec,-2)
    N= acftrans.shape[0]

    for l in range(N):
        for m in range(-(N+l)+1,N-l-1):
            constterm = (N-sp.absolute(m)+l)
            # for r1, Rxx(m)
            if sp.absolute(m)>=N-1:
                r1=sp.zeros_like(acftrans[0])
            elif m>=0:
                r1 = acftrans[m]
            elif m<0:
                r1=acftrans[-m]

            # for r2, Rxx(m+1)
            m2=m+l
            if sp.absolute(m2)>=N-1:
                r2=sp.zeros_like(acftrans[0])
            elif m2>=0:
                r2 = acftrans[m2]
            elif m2<0:
                r2=acftrans[-m2]

            # for r3, Rxx(m-1)
            m3=m+l
            if sp.absolute(m3)>=N-1:
                r3=sp.zeros_like(acftrans[0])
            elif m3>=0:
                r3 = acftrans[m3]
            elif m3<0:
                r3=acftrans[-m3]

            acfvar[l]=acfvar[l]+constterm*(sp.absolute(r1)**2+sp.absolute(r2*r3))

    return sp.transpose(acfvar/N,tuple(axisvec))
#%% Testing
def main():
    """Testing function"""
    t1 = time.time()
    curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curpath)[0],'Testdata')
    inifile = os.path.join(testpath,'PFISRExample.pickle')
    (sensdict,simparams) = readconfigfile(inifile)
    testh5 = os.path.join(testpath,'testiono.h5')
    ioncont = IonoContainer.readh5(testh5)
    outfile = os.path.join(testpath,'testionospec.h5')

    ioncont.makespectruminstanceopen(specfunctions.ISRSspecmake,sensdict,
                                     simparams['numpoints']).saveh5(outfile)
    radardata = RadarDataFile({0.0:outfile},inifile,testpath)

    ionoout = radardata.processdataiono()
    ionoout.saveh5(os.path.join(testpath,'lags.h5'))
    t2 = time.time()
    print(t2-t1)
if __name__== '__main__':
    main()
    
