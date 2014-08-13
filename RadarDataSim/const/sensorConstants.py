#!/usr/bin/env python
"""
Created on Tue Nov 26 12:42:11 2013

@author: John Swoboda
These are system constants for various sensors
"""
import pdb
import os

import tables
import numpy as np
from scipy.interpolate import griddata
import scipy as sp

from physConstants import v_C_0
## Parameters for Sensor
#AMISR = {'Name':'AMISR','Pt':2e6,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':449e6,'fs':50e3,\
#    'taurg':14,'Tsys':120,'BeamWidth':(2,2)}
#AMISR['t_s'] = 1/AMISR['fs'] 


def getConst(typestr,angles = None):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    if typestr.lower() =='risr':
        h5filename = os.path.join(dirname,'RISR_PARAMS.h5')
    elif typestr.lower() =='pfisr':
        h5filename = os.path.join(dirname,'PFISR_PARAMS.h5')
    h5file = tables.open_file(h5filename)
    kmat = h5file.root.Params.Kmat.read()
    freq = float(h5file.root.Params.Frequency.read())
    P_r = float(h5file.root.Params.Power.read())
    h5file.close()
    
    az = kmat[:,1]
    el = kmat[:,2]
    ksys = kmat[:,3]
    
    (xin,yin) = angles2xy(az,el)
    points = sp.vstack((xin,yin)).transpose()
    if angles !=None:
        (xvec,yvec) = angles2xy(angles[:,0],angles[:,1])
        ksysout = griddata(points, ksys, (xvec, yvec), method='nearest')
    else:
        ksysout = None

    sensdict = {'Name':typestr,'Pt':P_r,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':freq,'fs':50e3,\
    'taurg':14,'Tsys':120,'BeamWidth':(2,2),'Ksys':ksysout,'BandWidth':22970}
    sensdict['t_s'] = 1.0/sensdict['fs']
    return sensdict
    
def angles2xy(az,el):
    """ """
    
    azt = (az)*np.pi/180.0
    elt = 90-el
    xout = elt*np.sin(azt)
    yout = elt*np.cos(azt)
    return (xout,yout)
    
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

def AMISR_Pattern(AZ,EL,Az0,El0):
    """     
    AMISR_Pattern 
    by John Swoboda
    This function will create an idealized antenna pattern for the AMISR
    array. The pattern is not normalized. 
    The antenna is assumed to made of a grid of ideal cross dipole 
    elements. In the array every other column is shifted by 1/2 dy. The
    parameters are taken from the AMISR spec and the method for calculating
    the field is derived from a report by Adam R. Wichman.
    The inputs for the az and el coordinates can be either an array or
    scalar. If both are arrays they must be the same shape.
    ###########################################################################
    Inputs
    Az - An array or scalar holding the azimuth coordinates in radians.
    EL - An array or scalar holding the elevation coordinates in radians.
       Also vertical is at zero radians.
    Az0 - A scalar that determines the azimuth pointing angle of the antenna.
    El0 - A scalar that determines the elevation pointing angle of the
    antenna.
    ###########################################################################
    Outputs
    Patout - The normalized radiation density.
    ###########################################################################"""
    f0=440e6 # frequency of AMISR in Hz
    lam0=v_C_0/f0 # wavelength in m
    k0=2*np.pi/lam0 # wavenumber in rad/m
    
    dx=0.4343 # x spacing[m]
    dy=0.4958 # y spacing[m]
    # element pattern from an ideal cross dipole array.
    elementpower=(1/2)*(1+(np.cos(EL))**2)
    
    m=8.0;# number of pannels in the x direction
    mtot = 8.0*m;# number of elements times panels in x direction
    
    n = 16.0;# number of pannels in the y direction
    ntot = n*4.0;# number of elements times panels in y direction
    # relative phase between the x elements
    phix = k0*dx*(np.sin(EL)*np.cos(AZ)-np.sin(El0)*np.cos(Az0))
    # relative phase between the y elements
    phiy = k0*dy*(np.sin(EL)*np.sin(AZ)-np.sin(El0)*np.sin(Az0))
    
    AF = (1/2)*(1.0+np.exp(1j*((1/2)*phiy+phix)))*diric(2.0*phix,mtot/2.0)*diric(phiy,ntot);
    arrayfac = abs(AF)**2;
    Patout = elementpower*arrayfac
    return Patout