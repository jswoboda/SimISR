#!/usr/bin/env python
"""
Created on Wed May  6 13:55:26 2015
analysisplots.py
This module is used to plot the output from various stages of the simulator to debug
problems.
@author: John Swoboda
"""
import os, glob
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import scipy.fftpack as scfft
import scipy.interpolate as spinterp

import numpy as np
import seaborn as sns
import pdb

from RadarDataSim.IonoContainer import IonoContainer
from RadarDataSim.utilFunctions import readconfigfile,spect2acf


def maketi(Ionoin):
    (Nloc,Nt,Nion,Nppi) = Ionoin.Param_List.shape
    Paramlist = Ionoin.Param_List[:,:,:-1,:]
    Nisum = sp.sum(Paramlist[:,:,:,0],axis=2)
    Tisum = sp.sum(Paramlist[:,:,:,0]*Paramlist[:,:,:,1],axis=2)
    Tiave = Tisum/Nisum
    Newpl = sp.zeros((Nloc,Nt,Nion+1,Nppi))
    Newpl[:,:,:-1,:] = Ionoin.Param_List
    Newpl[:,:,-1,0] = Nisum
    Newpl[:,:,-1,1] = Tiave
    newrow = sp.array(['Ni','Ti'])
    newpn = sp.vstack((Ionoin.Param_Names,newrow))
    Ionoin.Param_List = Newpl
    Ionoin.Param_Names = newpn
    return Ionoin

def plotbeamparameters(times,configfile,maindir,params=['Ne'],indisp=True,fitdisp = True,filetemplate='params',suptitle = 'Parameter Comparison'):
    """ """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    rc('text', usetex=True)
    ffit = os.path.join(maindir,'Fitted','fitteddata.h5')
    inputfiledir = os.path.join(maindir,'Origparams')
    (sensdict,simparams) = readconfigfile(configfile)

    paramslower = [ip.lower() for ip in params]
    Nt = len(times)
    Np = len(params)

    if fitdisp:
        Ionofit = IonoContainer.readh5(ffit)
        dataloc = Ionofit.Sphere_Coords
        pnames = Ionofit.Param_Names
        pnameslower = sp.array([ip.lower() for ip in pnames.flatten()])
        p2fit = [sp.argwhere(ip==pnameslower)[0][0] if ip in pnameslower else None for ip in paramslower]
        time2fit = [None]*Nt
        for itn,itime in enumerate(times):
            filear = sp.argwhere(Ionofit.Time_Vector>=itime)
            if len(filear)==0:
                filenum = len(Ionofit.Time_Vector)-1
            else:
                filenum = filear[0][0]
            time2fit[itn] = filenum

    angles = dataloc[:,1:]
    rng = sp.unique(dataloc[:,0])
    b = np.ascontiguousarray(angles).view(np.dtype((np.void, angles.dtype.itemsize * angles.shape[1])))
    _, idx, invidx = np.unique(b, return_index=True,return_inverse=True)

    beamlist = angles[idx]

    Nb = beamlist.shape[0]

    if indisp:
        dirlist = glob.glob(os.path.join(inputfiledir,'*.h5'))
        filesonly= [os.path.splitext(os.path.split(ifile)[-1])[0] for ifile in dirlist]

        timelist = sp.array([int(i.split()[0]) for i in filesonly])
        time2file = [None]*Nt
        for itn,itime in enumerate(times):
            filear = sp.argwhere(timelist>=itime)
            if len(filear)==0:
                filenum = len(timelist)-1
            else:
                filenum = filear[0][0]
            time2file[itn] = filenum


    nfig = int(sp.ceil(Nt*Nb*Np/9.0))
    imcount = 0
    curfilenum = -1
    for i_fig in range(nfig):
        lines = [None]*2
        labels = [None]*2
        (figmplf, axmat) = plt.subplots(3, 3,figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for iax,ax in enumerate(axvec):
            if imcount>=Nt*Nb*Np:
                break
            itime = int(sp.floor(imcount/Nb/Np))
            iparam = int(imcount/Nb-Np*itime)
            ibeam = int(imcount-(itime*Np*Nb+iparam*Nb))

            curbeam = beamlist[ibeam]

            altlist = sp.sin(curbeam[1]*sp.pi/180.)*rng

            if fitdisp:

                indxkep = np.argwhere(invidx==ibeam)[:,0]

                curfit = Ionofit.Param_List[indxkep,time2fit[itime],p2fit[iparam]]
                rng_fit= dataloc[indxkep,0]
                alt_fit = rng_fit*sp.sin(curbeam[1]*sp.pi/180.)

                lines[1]= ax.plot(curfit,alt_fit,marker='.',c='g')[0]
                labels[1] = 'Fitted Parameters'
            if indisp:
                filenum = time2file[itime]
                if curfilenum!=filenum:
                    curfilenum=filenum
                    datafilename = os.path.join(inputfiledir,dirlist[filenum])
                    Ionoin = IonoContainer.readh5(datafilename)
                    if 'ti' in paramslower:
                        Ionoin = maketi(Ionoin)
                    pnames = Ionoin.Param_Names
                    pnameslower = sp.array([ip.lower() for ip in pnames.flatten()])


                prmloc = sp.argwhere(paramslower[iparam]==pnameslower)

                if prmloc.size !=0:
                    curprm = prmloc[0][0]

                curcoord = sp.zeros(3)
                curcoord[1:] = curbeam
                curdata = sp.zeros(len(rng))
                for irngn, irng in enumerate(rng):
                    curcoord[0] = irng
                    tempin = Ionoin.getclosestsphere(curcoord,times)[0]
                    Ntloc = tempin.shape[0]
                    tempin = sp.reshape(tempin,(Ntloc,len(pnameslower)))
                    curdata[irngn] = tempin[0,curprm]
                lines[0]= ax.plot(curdata,altlist,marker='o',c='b')[0]
                labels[0] = 'Input Parameters'
                if paramslower[iparam]!='ne':
                    ax.set(xlim=[0.25*sp.amin(curdata),2.5*sp.amax(curdata)])
            if paramslower[iparam]=='ne':
                ax.set_xscale('log')

            ax.set_xlabel(params[iparam])
            ax.set_ylabel('Alt km')
            ax.set_title('{0} vs Altitude, Time: {1}s Az: {2}$^o$ El: {3}$^o$'.format(params[iparam],times[itime],*curbeam))
            imcount=imcount+1

        figmplf.suptitle(suptitle, fontsize=20)
        if None in labels:
            labels.remove(None)
            lines.remove(None)
        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
        fname= filetemplate+'_{0:0>3}.png'.format(i_fig)
        plt.savefig(fname)
        plt.close(figmplf)

def plotspecs(coords,times,configfile,maindir,cartcoordsys = True, indisp=True,acfdisp= True,filetemplate='spec',suptitle = 'Spectrum Comparison'):
    """ This will create a set of images that compare the input ISR spectrum to the
    output ISR spectrum from the simulator.
    Inputs
    coords - An Nx3 numpy array that holds the coordinates of the desired points.
    times - A numpy list of times in seconds.
    configfile - The name of the configuration file used.
    cartcoordsys - (default True)A bool, if true then the coordinates are given in cartisian if
    false then it is assumed that the coords are given in sphereical coordinates.
    specsfilename - (default None) The name of the file holding the input spectrum.
    acfname - (default None) The name of the file holding the estimated ACFs.
    filetemplate (default 'spec') This is the beginning string used to save the images."""
#    indisp = specsfilename is not None
#    acfdisp = acfname is not None

    acfname = os.path.join(maindir,'ACF','00lags.h5')
    specsfiledir = os.path.join(maindir,'Spectrums')
    (sensdict,simparams) = readconfigfile(configfile)
    simdtype = simparams['dtype']
    npts = simparams['numpoints']*3.0
    amb_dict = simparams['amb_dict']
    if sp.ndim(coords)==1:
        coords = coords[sp.newaxis,:]
    Nt = len(times)
    Nloc = coords.shape[0]
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    if indisp:
        dirlist = os.listdir(specsfiledir)
        timelist = sp.array([int(i.split()[0]) for i in dirlist])
        for itn,itime in enumerate(times):
            filear = sp.argwhere(timelist>=itime)
            if len(filear)==0:
                filenum = len(timelist)-1
            else:
                filenum = filear[0][0]
            specsfilename = os.path.join(specsfiledir,dirlist[filenum])
            Ionoin = IonoContainer.readh5(specsfilename)
            if itn==0:
                specin = sp.zeros((Nloc,Nt,Ionoin.Param_List.shape[-1])).astype(Ionoin.Param_List.dtype)
            omeg = Ionoin.Param_Names
            npts = Ionoin.Param_List.shape[-1]

            for icn, ic in enumerate(coords):
                if cartcoordsys:
                    tempin = Ionoin.getclosest(ic,times)[0]
                else:
                    tempin = Ionoin.getclosestsphere(ic,times)[0]
#                if sp.ndim(tempin)==1:
#                    tempin = tempin[sp.newaxis,:]
                specin[icn,itn] = tempin[0,:]/npts/npts

    if acfdisp:
        Ionoacf = IonoContainer.readh5(acfname)
        ACFin = sp.zeros((Nloc,Nt,Ionoacf.Param_List.shape[-1])).astype(Ionoacf.Param_List.dtype)
        ts = sensdict['t_s']
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


    nfig = int(sp.ceil(Nt*Nloc/6.0))
    imcount = 0

    for i_fig in range(nfig):
        lines = [None]*3
        labels = [None]*3
        (figmplf, axmat) = plt.subplots(2, 3,figsize=(16, 12), facecolor='w')
        axvec = axmat.flatten()
        for iax,ax in enumerate(axvec):
            if imcount>=Nt*Nloc:
                break
            iloc = int(sp.floor(imcount/Nt))
            itime = int(imcount-(iloc*Nt))

            maxvec = []

            if indisp:
                # apply ambiguity funciton to spectrum
                curin = specin[iloc,itime]
                rcs = curin.real.sum()
                (tau,acf) = spect2acf(omeg,curin)

                # apply ambiguity function
                tauint = amb_dict['Delay']
                acfinterp = sp.zeros(len(tauint),dtype=simdtype)

                acfinterp.real =spinterp.interp1d(tau,acf.real,bounds_error=0)(tauint)
                acfinterp.imag =spinterp.interp1d(tau,acf.imag,bounds_error=0)(tauint)
                # Apply the lag ambiguity function to the data
                guess_acf = sp.zeros(amb_dict['Wlag'].shape[0],dtype=sp.complex128)
                for i in range(amb_dict['Wlag'].shape[0]):
                    guess_acf[i] = sp.sum(acfinterp*amb_dict['Wlag'][i])

            #    pdb.set_trace()
                guess_acf = guess_acf*rcs/guess_acf[0].real

                # fit to spectrums
                spec_interm = scfft.fftshift(scfft.fft(guess_acf,n=npts))
                maxvec.append(spec_interm.real.max())
                lines[0]= ax.plot(omeg*1e-3,spec_interm.real,label='Input',linewidth=5)[0]
                labels[0] = 'Input Spectrum With Ambiguity Applied'
                normset = spec_interm.real.max()/curin.real.max()
                lines[1]= ax.plot(omeg*1e-3,curin.real*normset,label='Input',linewidth=5)[0]
                labels[1] = 'Input Spectrum'
            if acfdisp:
                lines[2]=ax.plot(omeg*1e-3,specout[iloc,itime].real,label='Output',linewidth=5)[0]
                labels[2] = 'Estimated Spectrum'
                maxvec.append(specout[iloc,itime].real.max())
            ax.set_xlabel('f in kHz')
            ax.set_ylabel('Amp')
            ax.set_title('Location {0}, Time {1}'.format(coords[iloc],times[itime]))
            ax.set_ylim(0.0,max(maxvec)*1.1)

            imcount=imcount+1
        figmplf.suptitle(suptitle, fontsize=20)
        if None in labels:
            labels.remove(None)
            lines.remove(None)
        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
        fname= filetemplate+'_{0:0>3}.png'.format(i_fig)
        plt.savefig(fname)
        plt.close(figmplf)

def analysisdump(maindir,configfile,suptitle=None):
    """ """
    plotdir = os.path.join(maindir,'AnalysisPlots')
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    #plot spectrums
    filetemplate1 = os.path.join(maindir,'AnalysisPlots','Spec')
    (sensdict,simparams) = readconfigfile(configfile)
    angles = simparams['angles']
    ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
    zenang = ang_data[sp.argmax(ang_data[:,1])]
    rnggates = simparams['Rangegatesfinal']
    rngchoices = sp.linspace(sp.amin(rnggates),sp.amax(rnggates),4)
    angtile = sp.tile(zenang,(len(rngchoices),1))
    coords = sp.column_stack((sp.transpose(rngchoices),angtile))
    times = simparams['Timevec']


    filetemplate2= os.path.join(maindir,'AnalysisPlots','Params')
    if suptitle is None:
        plotspecs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate1)


        plotbeamparameters(times,configfile,maindir,params=['Ne','Te','Ti'],filetemplate=filetemplate2)
    else:
        plotspecs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate1,suptitle=suptitle)
        plotbeamparameters(times,configfile,maindir,params=['Ne','Te','Ti'],filetemplate=filetemplate2,suptitle=suptitle)