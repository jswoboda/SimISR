#!/usr/bin/env python
"""
Created on Mon Mar 16 19:36:27 2015

@author: John Swoboda
"""
import scipy as sp
from ISRSpectrum.ISRSpectrum import ISRSpectrum

def ISRSspecmake(ionocont,sensdict,npts):
    Vi = ionocont.getDoppler()
    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])

    paramshape = ionocont.Param_List.shape
    if ionocont.Time_Vector is None:
        outspecs = np.zeros((paramshape[0],1,npts))
        full_grid = False
    else:
        outspecs = np.zeros((paramshape[0],paramshape[1],npts))
        full_grid = True

    (N_x,N_t) = outspecs.shape[:2]
    #pdb.set_trace()
    for i_x in np.arange(N_x):
        for i_t in np.arange(N_t):
            if full_grid:
                cur_params = ionocont.Param_List[i_x,i_t]
            else:
                cur_params = ionocont.Param_List[i_x]
            (omeg,cur_spec,rcs) = specobj.getspec(cur_params,rcsflag=True)
            cur_spec_weighted = len(cur_spec)**2*cur_spec*rcs/cur_spec.sum()
            outspecs[i_x,i_t] = cur_spec_weighted

    return (omeg,outspecs,npts)