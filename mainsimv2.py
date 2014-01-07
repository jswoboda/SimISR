#!/usr/bin/env python
"""
Created on Sat Jan  4 17:54:00 2014

@author: Bodangles
"""

import numpy as np
import scipy as sp
import scipy.io as sio
from radarData import RadarData
from IonoContainer import IonoContainer, MakeTestIonoclass
import time
import sys
from physConstants import *
from sensorConstants import *
sys.path.append('/Users/Bodangles/Documents/Python/RadarDataSim')
from beamtools.bcotools import getangles
import os
outpath = '/Users/Bodangles/Documents/MATLAB/ursi2014'



t1 = time.time()
IPP = .0087
angles = getangles('spcorbco.txt')
t_int = 8.7*len(angles)
pulse = np.ones(14)
rng_lims = [250,500]
ioncont = MakeTestIonoclass()
ioncont.savemat(os.path.join(outpath,'spcorrinput.mat'))
time_lim = t_int

radardata = RadarData(ioncont,AMISR,angles,IPP,t_int,time_lim,pulse,rng_lims)
radardata.fitalldata()
out_cont = radardata.makeionocontainer()
out_cont.savemat(os.path.join(outpath,'spcorresult.mat'))
t2 = time.time()
print(t2-t1)