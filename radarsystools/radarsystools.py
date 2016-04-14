#!/usr/bin/env python
"""
radarsystools
This set of functions can be used to calculate expected SNR and RMS error of ISR
data.

@author: John Swoboda
"""
import sys, getopt
import numpy as np

from RadarDataSim.const.physConstants import v_C_0, v_Boltz, v_electron_rcs, v_epsilon0,v_elemcharge
import RadarDataSim.const.sensorConstants as sensconst
import pdb

class RadarSys(object):
    """ This is a class for the radar system object. This object can be used to
    determine the expected SNR and variance of the return. The class will be able
    to ouput a print out or a matrix of data. The variables for the class are
    the following which will be input in the constructor.
    Inputs
    sysdict - A dictionary that holds the information on the radar system.
    rng - A numpy array that holds the ranges the user wants to see the SNR or
    variances.
    Kpulse - The number of pulses used to integrate."""
    def __init__(self,sysdict,rng = np.arange(100.0,600.0,100.0),Kpulse = 1,beamnum=0):
        """Constructor for the RadarSys class see above for inputs."""
        self.sysdict = sysdict
        self.rng = rng
        self.Kpulse = Kpulse
        self.noisepow = noisepow(sysdict['Tsys'],sysdict['BandWidth'])
        self.beamnum =beamnum

    def snr(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function will return SNR esimtates of ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the SNR, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.
        Output
        SNRdata - An array the same shape as ne that holds the SNR'''
        #%% Assersions
        assert type(te) is np.ndarray, "te is not an numpy array."
        assert type(ne) is np.ndarray, "ne is not an numpy array."
        assert type(ti) is np.ndarray,  "ti is not an numpy array."
        #%% Get parameters from class
        Kpulse = self.Kpulse
        #%% Do the SNR calculation
        powdata = self.powcalc(ne,te,ti)
        SNRdata = np.sqrt(Kpulse)*powdata/self.noisepow
        return (SNRdata)

    def printsnr(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function will print out the SNR esimtates of ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the SNR, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.'''
        #%% Assersions
        assert type(te) is np.ndarray, "te is not an numpy array."
        assert type(ne) is np.ndarray, "ne is not an numpy array."
        assert type(ti) is np.ndarray,  "ti is not an numpy array."
        #%% Get parameters from class
        rng= self.rng
        Kpulse = self.Kpulse
        #%% Do the SNR calculation
        SNRdata = self.snr(ne,te,ti)
        SNRdb = pow2db(SNRdata)

        #%% Set up and Make print out
        print "SNR for the follwing Ne with Number of pulses = {0:d}".format(Kpulse)
        # make strings for each Ne
        nestrs = ['{:.2g}m^-3 '.format(i) for i in ne]
        nestrs.insert(0,'Range ')
        lstoflst = [nestrs]
        # make a list of lists for each SNR
        for irng,vrng in enumerate(rng):
            rngstr = "{0:.2f}km ".format(vrng)
            snrliststr = ['{:.2f}dB '.format(i) for i in SNRdb[irng]]
            snrliststr.insert(0,rngstr)
            lstoflst.append(snrliststr)

        printtable(lstoflst)

    def rms(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function calculate the rms error of the  ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the variance, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.'''
        #%% Assersions
        assert type(te) is np.ndarray, "te is not an numpy array."
        assert type(ne) is np.ndarray, "ne is not an numpy array."
        assert type(ti) is np.ndarray,  "ti is not an numpy array."
        #%% Do the SNR calculation
        powdata = self.powcalc(ne,te,ti)
        vardata = (powdata+self.noisepow)**2/self.Kpulse
        rmsdata = np.sqrt(vardata)
        return rmsdata
    def printrms(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function will print out the rms esimtates of ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the variance, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.'''
        rmsdata = self.rms(ne,te,ti)
        #%% Set up and Make print out
        print "RMS for the follwing Ne with Number of pulses = {0:d}".format(self.Kpulse)
        # make strings for each Ne
        nestrs = ['{:.2g}m^-3 '.format(i) for i in ne]
        nestrs.insert(0,'Range ')
        lstoflst = [nestrs]
        # make a list of lists for each SNR
        for irng,vrng in enumerate(self.rng):
            rngstr = "{0:.2e}km ".format(vrng)
            rmsliststr = ['{:.2e}W '.format(i) for i in rmsdata[irng]]
            rmsliststr.insert(0,rngstr)
            lstoflst.append(rmsliststr)
        printtable(lstoflst)


    def printfracrms(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function will print out the Fractional RMS esimtates of ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the variance, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.'''
        #%% Assersions
        assert type(te) is np.ndarray, "te is not an numpy array."
        assert type(ne) is np.ndarray, "ne is not an numpy array."
        assert type(ti) is np.ndarray,  "ti is not an numpy array."
        #%% Get parameters from class
        rng= self.rng
        Kpulse = self.Kpulse
        #%% Do the SNR calculation
        powdata = self.powcalc(ne,te,ti)
        rmsdata = (powdata+self.noisepow)/(powdata*np.sqrt(self.Kpulse))

        #%% Set up and Make print out
        print "RMS fractional error for the follwing Ne with Number of pulses = {0:d}".format(Kpulse)
        # make strings for each Ne
        nestrs = ['{:.2g}m^-3 '.format(i) for i in ne]
        nestrs.insert(0,'Range ')
        lstoflst = [nestrs]
        # make a list of lists for each SNR
        for irng,vrng in enumerate(rng):
            rngstr = "{0:.2f}km ".format(vrng)
            rmsliststr = ['{:.2f} '.format(i) for i in rmsdata[irng]]
            rmsliststr.insert(0,rngstr)
            lstoflst.append(rmsliststr)

        printtable(lstoflst)

    def powcalc(self,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
        '''This function will print out the returned power esimtates of ISR data
        with the input plasma parameters and system prameters. The inputs are
        the basic plasma parameters the impact the variance, the electron density and
        the electron and ion tempretures.
        Inputs
        ne - A one dimensional array of electron densities.
        te = A one dimensional array of electron tempretures.
        ti = A one dimensional array of ion tempretures.'''
        #%% Assersions
        assert type(te) is np.ndarray, "te is not an numpy array."
        assert type(ne) is np.ndarray, "ne is not an numpy array."
        assert type(ti) is np.ndarray,  "ti is not an numpy array."
        assert ti.shape==te.shape and te.shape == ne.shape, "te, ti and ne must be the same shape."

        #%% Get parameters from class
        rng= self.rng
        sysdict = self.sysdict

        Pt =sysdict['Pt']
        taup = sysdict['t_s']*sysdict['taurg']
        k = sysdict['k']
        antconst = 0.4;

        (NE,RNG) = np.meshgrid(ne,rng*1e3)
        TE = np.meshgrid(te,rng)[0]
        TI = np.meshgrid(ti,rng)[0]
        Tr = TE/TI
        debyel = np.sqrt(v_epsilon0*v_Boltz*TE/(NE*v_elemcharge))

        pt2 = Pt*taup/RNG**2

        if sysdict['Ksys'] == None:
            G = sysdict['G']
            pt1 = antconst*v_C_0*G/(8*k**2)
            rcs =v_electron_rcs* NE/((1.0+k**2*debyel**2)*(1.0+k**2*debyel**2+Tr))
        else:
            pt1 = sysdict['Ksys'][self.beamnum]
            rcs =NE/((1.0+k**2*debyel**2)*(1.0+k**2*debyel**2+Tr))
        Pr = pt1*pt2*rcs
        return Pr

#%% Utility functions
def noisepow(Tsys,BW):
    return Tsys*BW*v_Boltz
def pow2db(x):
    """Returns in dB x."""
    return 10.0*np.log10(x)
def mag2db(x):
    """Returns in dB the magnitude in x."""
    return 20.0*np.log10(np.abs(x))
def printtable(rows):
    cols = zip(*rows)
    col_widths = [ max(len(value) for value in col) for col in cols ]
    format = ' '.join(['%%-%ds' % width for width in col_widths ])
    for row in rows:
        print format % tuple(row)

def main(argv):

    kinput=1
    try:
      opts, args = getopt.getopt(argv,"hp:o:",["pulses=","ofile="])
    except getopt.GetoptError:
      print 'radarsystools.py -p <number of pulses> -o <outputfile>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'radarsystools.py -p <number of pulses> -o <outputfile>'
         sys.exit()
      elif opt in ("-p", "--pulses"):
         kinput = int(arg)
      elif opt in ("-o", "--ofile"):
          print "Still have to add output file feature. I'm working on it... may be..."

    print "Without angle"
    sensdict = sensconst.getConst('risr')
    sensdict['tau'] = 320e-6
    r_sys = RadarSys(sensdict,Kpulse = kinput)
    r_sys.printsnr(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    r_sys.printrms(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    r_sys.printfracrms(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    print "\n\n"


    angles = [(90,85)]
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    sensdict = sensconst.getConst('risr',ang_data)
    sensdict['tau'] = 320e-6

    print "With Angle"
    r_sys = RadarSys(sensdict,Kpulse = kinput)
    r_sys.printsnr(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    r_sys.printrms(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    r_sys.printfracrms(ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))

#%% Test function
if __name__== '__main__':
    main(sys.argv[1:])

