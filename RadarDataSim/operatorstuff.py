#!/usr/bin/env python
"""
Created on Tue Apr 14 11:10:40 2015

@author: John Swoboda
"""
from const.physConstants import v_C_0, v_Boltz
import scipy as sp

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
def makematPA(Sphere_Coords,timein,timeout,sensdict,simparams):
    """Make a Ntimeout*Nbeam*Nrng x Ntime*Nloc matrix. The output space will have range repeated first,
    then beams then time. The coordinates will be [t0,b0,r0],[t0,b0,r1],[t0,b0,r2],...
    [t0,b1,r0],[t0,b1,r1], ... [t1,b0,r0],[t1,b0,r1],...[t1,b1,r0]..."""
    #
    fullmat = True
    rng_vec = simparams['Rangegates']
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
        outmat = sp.matrix(sp.zeros((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg)))
    else:
        outmat = sp.sparse((Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg),dype =sp.float64)

    weights = {ibn:sensdict['ArrayFunc'](Az,El,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}

    for iton,ito in enumerate(timeout):
        overlaps = sp.array([getOverlap(ito,x) for x in timein])
        weights_time = overlaps/overlaps.sum()
        itpnts = sp.where(weights_time>0)[0]

        # usually the matrix size is nbeamsxnrange
        for ibn in range(Nbeams):
            print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
            weight = weights[ibn]
            for isamp in range(nrgout):
                # make the row
                irow = isamp+ibn*nrgout+iton*nrgout*Nbeams

                range_g = rng_vec2[isamp]
                rnglims = [range_g-minrg,range_g+maxrg]
                rangelog = sp.argwhere((rho>=rnglims[0])&(rho<rnglims[1]))

                # This is a nearest neighbors interpolation for the spectrums in the range domain
                if sp.sum(rangelog)==0:
                    minrng = sp.argmin(sp.absolute(range_g-rho))
                    rangelog[minrng] = True
                #create the weights and weight location based on the beams pattern.
                weight_cur =weight[rangelog[:,0]]
                weight_cur = weight_cur/weight_cur.sum()
                weight_loc = sp.where(rangelog[:,0])[0]

                w_loc_rep = sp.tile(weight_loc,len(itpnts))
                t_loc_rep = sp.repeat(itpnts)
                icols = t_loc_rep*Nlocbeg+w_loc_rep

                weights_final = weights_time[t_loc_rep]*weight_cur[w_loc_rep]*range_g**2/rho[w_loc_rep]**2
                outmat[irow,icols] = weights_final


    return(weights_final)



