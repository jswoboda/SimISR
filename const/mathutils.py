#!/usr/bin/env python
"""
Created on Thu Jun 12 17:43:43 2014

@author: Bodangles
"""

import numpy as np
def diric(x,M):
    """ This calculates the dirichete sinc function """
    if M % 1 != 0 or M <= 0:
        raise RuntimeError('n must be a strictly positive integer')
        
    y=np.sin(0.5*x)
    ilog = np.abs(y) < 1e-12
    nilog = np.logical_not(ilog)
    y[nilog]=np.sin((M/2)*x[nilog])/(M*y[nilog])
    y[ilog]=np.sign(np.cos(x[ilog]*((M+1.0)/2.0)))
    return y
    
def phys2array(az,el):
    """ This takes the physical angles of azimuth and elevation in degrees
    and brings them to the array space."""
    
    azt = (az)*np.pi/180.0
    elt = 90-el
    xout = elt*np.sin(azt)
    yout = elt*np.cos(azt)
    return (xout,yout)

