#!/usr/bin/env python
"""
basicplot2.py
This script will create figures 7 and 13 from the paper Space-Time Ambiguity Functions for Electronically Scanned ISR Applications.
This file relies on the RadarDataSim module to be installed on the python path
@author: John Swoboda
"""
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from RadarDataSim.IonoContainer import Chapmanfunc, TempProfile

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

H_0 = 40 #km
z_0 = 230 #km
N_0 = 1e11

zvec = sp.arange(100.0,600.0,2.0)

Ne_profile = Chapmanfunc(zvec,H_0,z_0,N_0)
(Te,Ti)= TempProfile(zvec)

fig = plt.figure()

plt.plot(Ne_profile,zvec,'o')
plt.xscale('log')
plt.xlabel(r'Log $(N_e)$ m$^{-3}$',fontsize=16)
plt.ylabel(r'Alt km',fontsize=16)
plt.grid(True)

plt.title(r'$N_e$ vs. Altitude',fontsize=20)
figname = 'paramsvsalt.png'

fig2 = plt.figure()
plt.plot(Ti,zvec,label=r'$T_i$',linewidth=3)
plt.hold(True)
plt.plot(Te,zvec,label=r'$T_e$',linewidth=3)
plt.hold(False)
plt.xlabel(r'Temp in $^\circ$K',fontsize=16)
plt.ylabel(r'Alt km',fontsize=16)
plt.axis([500,2500,100,600])
plt.grid(True)
plt.title(r'$T_e$ and $T_i$ vs. Altitude',fontsize=20)
