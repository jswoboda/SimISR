#!/usr/bin/env python
"""
Created on Tue Dec 31 11:21:38 2013

@author: Bodangles
"""

import numpy as np
import scipy as sp

from radarData import RadarData
from IonoContainer import IonoContainer, MakeTestIonoclass
import time
import sys
from physConstants import *
from sensorConstants import *
sys.path.append('/Users/Bodangles/Documents/Python/RadarDataSim')
from beamtools.bcotools import getangles





t1 = time.time()
IPP = .0087
angles = getangles('spcorbco.txt')
t_int = 8.7*len(angles)
pulse = np.ones(14)
rng_lims = [250,500]
ioncont = MakeTestIonoclass()
time_lim = t_int

radardata = RadarData(ioncont,AMISR,angles,IPP,t_int,time_lim,pulse,rng_lims)
radardata.fitalldata()
t2 = time.time()
print(t2-t1)