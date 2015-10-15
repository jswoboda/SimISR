#!/usr/bin/env python
"""
Created on Tue Apr 14 11:10:40 2015

@author: John Swoboda
"""
from const.physConstants import v_C_0, v_Boltz
import scipy as sp

def makematPA(Sphere_Coords,timein,timeout,sensdict,simparams):
    """Make a Ntimeout*Nbeam*Nrng x Ntime*Nloc matrix. The output space will have range repeated first,
    then beams then time. The coordinates will be [t0,b0,r0],[t0,b0,r1],[t0,b0,r2],...
    [t0,b1,r0],[t0,b1,r1], ... [t1,b0,r0],[t1,b0,r1],...[t1,b1,r0]..."""
    #
    fullmat = True
    range_gates = simparams['Rangegates']
    rng_bin=sensdict['t_s']*v_C_0/1000.0
    sumrule = simparams['SUMRULE']
    #
    minrgbin = -sumrule[0].min()
    maxrgbin = len(rng_vec)-sumrule[1].max()
    minrg = minrgbin*rng_bin
    maxrg = maxrgbin*rng_bin
    angles = simparams['angles']
    Nbeams = len(angles)
    rho = Sphere_Coords[:,0]
    Az = Sphere_Coords[:,1]
    El = Sphere_Coords[:,2]

    rng_vec2 = simparams['Rangegatesfinal']
    nrgout = len(rng_vec2)

    Nlocbeg = len(rho)
    Ntbeg = len(timein)
    Ntout = len(timeout)
    if fullmat:
        outmat= sp.matrix(sp.zeros((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg)))
    else:
        sp.sparse((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg),dype =sp.float64)

    weights = {ibn:sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}

    # usually the matrix size is nbeamsxnrange
    for ibn in range(Nbeams):
        print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
        weight = weights[ibn]
        for isamp in sp.arange(len(rng_vec2)):
            range_g = rng_vec2[isamp]
            range_m = range_g*1e3
            rnglims = [range_g-minrg,range_g+maxrg]
            rangelog = sp.argwhere((rho>=rnglims[0])&(rho<rnglims[1]))
            cur_pnts = samp_num+isamp
            if rangelog.size==0:
                pdb.set_trace()
            # This is a nearest neighbors interpolation for the spectrums in the range domain
            if sp.sum(rangelog)==0:
                minrng = sp.argmin(sp.absolute(range_g-rho))
                rangelog[minrng] = True
            #create the weights and weight location based on the beams pattern.
            weight_cur =weight[rangelog[:,0]]
            weight_cur = weight_cur/weight_cur.sum()


