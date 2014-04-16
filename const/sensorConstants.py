#!/usr/bin/env python
"""
Created on Tue Nov 26 12:42:11 2013

@author: John Swoboda
These are system constants for various sensors
"""
import tables
import os
import numpy as np
import pdb
from scipy.interpolate import griddata
import scipy as sp

## Parameters for Sensor
#AMISR = {'Name':'AMISR','Pt':2e6,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':449e6,'fs':50e3,\
#    'taurg':14,'Tsys':120,'BeamWidth':(2,2)}
#AMISR['t_s'] = 1/AMISR['fs'] 


def getConst(typestr,angles):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    if typestr.lower() =='risr':
        h5filename = os.path.join(dirname,'RISR_PARAMS.h5')
    elif typestr.lower() =='pfisr':
        h5filename = os.path.join(dirname,'PFISR_PARAMS.h5')
    h5file = tables.open_file(h5filename)
    kmat = h5file.root.Params.Kmat.read()
    freq = float(h5file.root.Params.Frequency.read())
    pow = float(h5file.root.Params.Power.read())
    h5file.close()
    
    
    az = kmat[:,1]
    el = kmat[:,2]
    ksys = kmat[:,3]
    
    (xin,yin) = angles2xy(az,el)
    (xvec,yvec) = angles2xy(angles[:,0],angles[:,1])
    
    
    points = sp.vstack((xin,yin)).transpose()
    ksysout = griddata(points, ksys, (xvec, yvec), method='linear')

    sensdict = {'Name':typestr,'Pt':pow,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':freq,'fs':50e3,\
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