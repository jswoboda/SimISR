#!/usr/bin/env python
"""
Created on Fri Jan  3 09:24:32 2014

@author: Bodangles
"""
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from IonoContainer import Chapmanfunc, TempProfile

figsdir = '/Users/Bodangles/Documents/Research/Confernces/URSI2014/Presentation/'

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

H_0 = 40 #km
z_0 = 300 #km
N_0 = 10**11

zvec =  sp.arange(100.0,600.0,10.0)

Ne_profile = Chapmanfunc(zvec,H_0,z_0,N_0)
(Te,Ti)= TempProfile(zvec)

fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(Ne_profile,zvec,'o')
plt.xscale('log')
plt.xlabel(r'Log $(N_e)$ m$^{-3}$',fontsize=16)
plt.ylabel(r'Alt km',fontsize=16)
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(Te,zvec,label=r'$T_e$')
plt.hold(True)
plt.plot(Ti,zvec,label=r'$T_i$')
plt.grid(True)
plt.xlabel(r'Temperature K',fontsize=16)
plt.legend(loc='upper right')
plt.axis([800,2200,200,500])
plt.suptitle(r'Parameters vs. Altitude',fontsize=20)
figname = 'paramsvsalt.png'
plt.savefig(os.path.join(figsdir,figname))
