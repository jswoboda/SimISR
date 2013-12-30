#!/usr/bin/env python
"""
Created on Tue Nov 26 12:42:11 2013

@author: John Swoboda
These are system constants for various sensors
"""
from physConstants import *

## Parameters for Sensor
AMISR = {'Name':'AMISR','Pt':2e6,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':449e6,'fs':50e3,\
    'taurg':14,'Tsys':120,'BeamWidth':(2,2)}
AMISR['t_s'] = 1/AMISR['fs'] 
