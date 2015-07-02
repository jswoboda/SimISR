#!/usr/bin/env python
"""
Created on Wed May  6 13:55:26 2015
analysisplots.py
This module is used to plot the output from various stages of the simulator to debug
problems.
@author: John Swoboda
"""
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import scipy.fftpack as scfft
import numpy as np
import seaborn as sns
import pdb

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

def plotspecs(coords,times,cartcoordsys = True, specsfilename=None,acfname=None,outdir='',npts = 128):
    indisp = specsfilename is not None
    acfdisp = acfname is not None

    if sp.ndim(coords)==1:
        coords = coords[sp.newaxis,:]
    Nt = len(times)
    Nloc = coords.shape[0]
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    if indisp:
        Ionoin = IonoContainer.readh5(specsfilename)
        omeg = Ionoin.Param_Names
        npts = Ionoin.Param_List.shape[-1]
        specin = sp.zeros((Nloc,Nt,Ionoin.Param_List.shape[-1])).astype(Ionoin.Param_List.dtype)
        for icn, ic in enumerate(coords):
            if cartcoordsys:
                tempin = Ionoin.getclosest(ic,times)[0]
            else:
                tempin = Ionoin.getclosestsphere(ic,times)[0]
            if sp.ndim(tempin)==1:
                tempin = tempin[sp.newaxis,:]
            specin[icn] = tempin/npts/npts

    if acfdisp:
        Ionoacf = IonoContainer.readh5(acfname)
        ACFin = sp.zeros((Nloc,Nt,Ionoacf.Param_List.shape[-1])).astype(Ionoacf.Param_List.dtype)
        ts = Ionoacf.Param_Names[1]-Ionoacf.Param_Names[0]
        omeg = sp.arange(-sp.ceil((npts+1)/2),sp.floor((npts+1)/2))/ts/npts
        for icn, ic in enumerate(coords):
            if cartcoordsys:
                tempin = Ionoacf.getclosest(ic,times)[0]
            else:
                tempin = Ionoacf.getclosestsphere(ic,times)[0]
            if sp.ndim(tempin)==1:
                tempin = tempin[sp.newaxis,:]
            ACFin[icn] = tempin
        specout = scfft.fftshift(scfft.fft(ACFin,n=npts,axis=-1),axes=-1)


    nfig = sp.ceil(Nt*Nloc/6.0)
    imcount = 0
    for i_fig in sp.arange(nfig):
        (figmplf, axmat) = plt.subplots(2, 3,figsize=(16, 12), facecolor='w')
        axvec = axmat.flatten()
        for iax,ax in enumerate(axvec):
            if imcount>=Nt*Nloc:
                break
            iloc = int(sp.floor(imcount/Nt))
            itime = int(imcount-(iloc*Nt))
            if indisp:
                ax.plot(omeg*1e-3,specin[iloc,itime].real,label='Input',linewidth=5)
            if indisp:
                ax.plot(omeg*1e-3,specout[iloc,itime].real,label='Output',linewidth=5)

            ax.set_xlabel('f in kHz')
            ax.set_ylabel('Amp')
            ax.set_title('Location {0}, Time {1}'.format(coords[iloc],times[itime]))
            imcount=imcount+1

        fname= os.path.join(outdir,'Specs_{0:0>3}.png'.format(i_fig))
        plt.savefig(fname)
        plt.close(figmplf)

