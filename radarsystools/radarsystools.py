#!/usr/bin/env python
"""
radarsystools
This set of functions can be used to calculate snr and variance of data

@author: Bodangles
"""


import numpy as np

from RadarDataSim.const.physConstants import v_C_0, v_Boltz, v_electron_rcs, v_epsilon0,v_elemcharge
import RadarDataSim.const.sensorConstants as sensconst
import pdb

def printsnr(sysdict,rng = np.arange(100.0,600.0,100.0),Kpulse = 1,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
    assert type(te) is np.ndarray, "te is not an numpy array."
    assert type(ne) is np.ndarray, "ne is not an numpy array."
    assert type(ti) is np.ndarray,  "ti is not an numpy array."
    SNRdata = snrcalc(sysdict,rng,Kpulse,ne,te,ti)
    SNRdb = pow2db(SNRdata)
    
    print "SNR for the follwing electron densities"
    
    nestrs = ['{:.2g}m^-3 '.format(i) for i in ne]
    nestrs.insert(0,'Range ')
    lstoflst = [nestrs]
    for irng,vrng in enumerate(rng):
        rngstr = "{0:.2f}km ".format(vrng)
        snrliststr = ['{:.2f}dB '.format(i) for i in SNRdb[irng]]
        snrliststr.insert(0,rngstr)
        lstoflst.append(snrliststr)
        
    cols = zip(*lstoflst)
    col_widths = [ max(len(value) for value in col) for col in cols ]
    format = ' '.join(['%%-%ds' % width for width in col_widths ])
    for row in lstoflst:
        print format % tuple(row)
#    titlestr = "m^-3\t".join(['{:.2g}'.format(i) for i in ne]) +'m^-3'
#    
#    print "SNR for the follwing electron densities"
#    print "Range\t" +titlestr
#    
#    for irng,vrng in enumerate(rng):
#        rngstr = "{0:.2f}km\t".format(vrng)
#        datastr = "dB \t".join(['{:.2f}'.format(i) for i in SNRdb[irng]])+'dB'
#        print rngstr+datastr
    
def snrcalc(sysdict,rng = np.arange(100.0,600.0,100.0),Kpulse = 1,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
    
    assert type(te) is np.ndarray, "te is not an numpy array."
    assert type(ne) is np.ndarray, "ne is not an numpy array."
    assert type(ti) is np.ndarray,  "ti is not an numpy array."
    powdata = powcalc(sysdict,rng,ne,te,ti)
    noisedata = noisepow(sysdict['Tsys'],sysdict['BandWidth'])
    
    return powdata/noisedata
    
    
def varcalc(sysdict,rng=np.arange(100.0,600.0,100.0),Kpulse = 1,ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
    assert type(te) is np.ndarray, "te is not an numpy array."
    assert type(ne) is np.ndarray, "ne is not an numpy array."
    assert type(ti) is np.ndarray,  "ti is not an numpy array."
    powdata = powcalc(sysdict,rng,ne,te,ti)
    noisedata = noisepow(sysdict['Tsys'],sysdict['BandWidth'])
    return (powdata+noisedata)**2/Kpulse**2
    
def powcalc(sysdict,rng = np.arange(100.0,600.0,100.0),ne = np.array([1e11]),te=np.array([1e3]),ti=np.array([1e3])):
    assert type(te) is np.ndarray, "te is not an numpy array."
    assert type(ne) is np.ndarray, "ne is not an numpy array."
    assert type(ti) is np.ndarray,  "ti is not an numpy array."
    assert ti.shape==te.shape and te.shape == ne.shape, "te, ti and ne must be the same shape."

    
    Pt =sysdict['Pt']
    G = sysdict['G']
    taup = sysdict['tau']
    k = sysdict['k']
    antconst = 0.4;
    
    (NE,RNG) = np.meshgrid(ne,rng*1e3)
    TE = np.meshgrid(te,rng)[0]
    TI = np.meshgrid(ti,rng)[0]
    Tr = TE/TI
    debyel = np.sqrt(v_epsilon0*v_Boltz*TE/(NE*v_elemcharge))
    
    pt2 = Pt*taup/RNG**2
    
    if sysdict['Ksys'] == None:
        pt1 = antconst*v_C_0*G/(8*k**2)
        rcs =v_electron_rcs* NE/((1.0+k**2*debyel**2)*(1.0+k**2*debyel**2+Tr))
    else:
        pt1 = sysdict['Ksys'][0]
        rcs =NE/((1.0+k**2*debyel**2)*(1.0+k**2*debyel**2+Tr))
    Pr = pt1*pt2*rcs
    return Pr
    
def noisepow(Tsys,BW):
    return Tsys*BW*v_Boltz
def pow2db(x):
    return 10.0*np.log10(x)
def mag2db(x):
    return 20.0*np.log10(x)
    
if __name__== '__main__':
    
    print "Without angle"
    sensdict = sensconst.getConst('risr')
    sensdict['tau'] = 320e-6
    printsnr(sensdict,ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    angles = [(90,85)]
    ang_data = np.array([[iout[0],iout[1]] for iout in angles])
    sensdict = sensconst.getConst('risr',ang_data)
    sensdict['tau'] = 320e-6

    print "With Angle"
    printsnr(sensdict,ne = np.array([1e11,2e11]),te=np.array([1e3,2e11]),ti=np.array([1e3,2e11]))
    