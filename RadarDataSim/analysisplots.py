#!/usr/bin/env python
"""
Created on Wed May  6 13:55:26 2015

@author: John Swoboda
"""
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import numpy as np
import seaborn as sns

from RadarDataSim.IonoContainer import IonoContainer


def plotbeamparameters(param='Ne',ffit=None,fin=None,acfname=None):
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    rc('text', usetex=True)

    fitdisp= ffit is not None
    indisp = ffit is not None
    acfdisp = acfname is not None

    if not param.lower()=='ne':
        acfdisp = False

    if acfdisp:
        Ionoacf = IonoContainer.readh5(acfname)
        dataloc = Ionoacf.Sphere_Coords
    if indisp:
        Ionoin = IonoContainer.readh5(fin)
    if fitdisp:
        Ionofit = IonoContainer.readh5(ffit)
        dataloc = Ionoacf.Sphere_Coords

    angles = dataloc[:,1:]
    b = np.ascontiguousarray(angles).view(np.dtype((np.void, angles.dtype.itemsize * angles.shape[1])))
    _, idx, invidx = np.unique(b, return_index=True,return_inverse=True)

    Neind = sp.argwhere(param==Ionofit.Param_Names)[0,0]
    beamnums = [0,1]
    beamlist = angles[idx]
    for ibeam in beamnums:
        curbeam = beamlist[ibeam]
        indxkep = np.argwhere(invidx==ibeam)[:,0]
        Ne_data = np.abs(Ionoacf.Param_List[indxkep,0,0])*2.0
        Ne_fit = Ionofit.Param_List[indxkep,0,Neind]
        rng= dataloc[indxkep,0]
        curlocs = dataloc[indxkep]
        origNe = np.zeros_like(Ne_data)
        rngin = np.zeros_like(rng)
        for ilocn,iloc in enumerate(curlocs):
            temparam,_,tmpsph = Ionoin.getclosestsphere(iloc)[:3]
            origNe[ilocn]=temparam[0,-1,0]
            rngin[ilocn] = tmpsph[0]
        print sp.nanmean(Ne_data/origNe)
        fig = plt.figure()
        plt.plot(Ne_data,rng,'bo',label='Data')
        plt.gca().set_xscale('log')
        plt.hold(True)
        plt.plot(origNe,rngin,'g.',label='Input')
        plt.plot(Ne_fit,rngin,'r*',label='Fit')
        plt.xlabel('$N_e$')
        plt.ylabel('Range km')
        plt.title('Ne vs Range for beam {0} {1}'.format(*curbeam))
        plt.legend(loc=1)

        plt.savefig('comp{0}'.format(ibeam))
        plt.close(fig)

def plotaltparameters(param='Ne',ffit=None,fin=None,acfname=None):
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    rc('text', usetex=True)

    fitdisp= ffit is not None
    indisp = ffit is not None
    acfdisp = acfname is not None

    if not param.lower()=='ne':
        acfdisp = False

    if acfdisp:
        Ionoacf = IonoContainer.readh5(acfname)
        dataloc = Ionoacf.Sphere_Coords
    if indisp:
        Ionoin = IonoContainer.readh5(fin)
    if fitdisp:
        Ionofit = IonoContainer.readh5(ffit)
        dataloc = Ionoacf.Sphere_Coords

    angles = dataloc[:,1:]
    b = np.ascontiguousarray(angles).view(np.dtype((np.void, angles.dtype.itemsize * angles.shape[1])))
    _, idx, invidx = np.unique(b, return_index=True,return_inverse=True)

    Neind = sp.argwhere(param==Ionofit.Param_Names)[0,0]
    beamnums = [0,1]
    beamlist = angles[idx]
    for ibeam in beamnums:
        curbeam = beamlist[ibeam]
        indxkep = np.argwhere(invidx==ibeam)[:,0]
        Ne_data = np.abs(Ionoacf.Param_List[indxkep,0,0])*2.0
        Ne_fit = Ionofit.Param_List[indxkep,0,Neind]
        rng= dataloc[indxkep,0]
        curlocs = dataloc[indxkep]
        origNe = np.zeros_like(Ne_data)
        rngin = np.zeros_like(rng)
        for ilocn,iloc in enumerate(curlocs):
            temparam,_,tmpsph = Ionoin.getclosestsphere(iloc)[:3]
            origNe[ilocn]=temparam[0,-1,0]
            rngin[ilocn] = tmpsph[0]
        print sp.nanmean(Ne_data/origNe)
        fig = plt.figure()
        plt.plot(Ne_data,rng,'bo',label='Data')
        plt.gca().set_xscale('log')
        plt.hold(True)
        plt.plot(origNe,rngin,'g.',label='Input')
        plt.plot(Ne_fit,rngin,'r*',label='Fit')
        plt.xlabel('$N_e$')
        plt.ylabel('Range km')
        plt.title('Ne vs Range for beam {0} {1}'.format(*curbeam))
        plt.legend(loc=1)

        plt.savefig('comp{0}'.format(ibeam))
        plt.close(fig)
if __name__== '__main__':
