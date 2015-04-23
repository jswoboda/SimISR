#!/usr/bin/env python
"""
Created on Tue Apr 14 11:10:40 2015

@author: John Swoboda
"""
from const.physConstants import v_C_0, v_Boltz
import scipy as sp
def makemat(Sphere_Coords,sensdict,simparams):
    #
    range_gates = simparams['Rangegates']
    angles = simparams['angles']
    Nbeams = len(angles)
    rho = Sphere_Coords[:,0]
    Az = Sphere_Coords[:,1]
    El = Sphere_Coords[:,2]
    rng_len=sensdict['t_s']*v_C_0/1000.0
    weights = {ibn:sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}

    for ibn in range(Nbeams):
        print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
        weight = weights[ibn]
        for isamp in sp.arange(len(range_gates)):
            range_g = range_gates[isamp]
            range_m = range_g*1e3
            rnglims = [range_g-rng_len/2.0,range_g+rng_len/2.0]
            rangelog = sp.argwhere((rho>=rnglims[0])&(rho<rnglims[1]))
            cur_pnts = samp_num+isamp
            if rangelog.size==0:
                pdb.set_trace()
            #create the weights and weight location based on the beams pattern.
            weight_cur =weight[rangelog[:,0]]
            weight_cur = weight_cur/weight_cur.sum()

