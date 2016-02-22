#!/usr/bin/env python
"""
Created on Wed May  6 13:55:26 2015
analysisplots.py
This module is used to plot the output from various stages of the simulator to debug
problems.
@author: John Swoboda
"""
import os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy as sp
import scipy.fftpack as scfft
import scipy.interpolate as spinterp

import numpy as np
import seaborn as sns
import pdb

from RadarDataSim.IonoContainer import IonoContainer
from RadarDataSim.utilFunctions import readconfigfile,spect2acf,acf2spect
from RadarDataSim.specfunctions import ISRspecmakeout,ISRSfitfunction,makefitsurf

def beamvstime(times,configfile,maindir,params=['Ne'],filetemplate='BeamParams',suptitle = 'Parameter Comparison',werrors=False):
    """ """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
#    rc('text', usetex=True)
    ffit = os.path.join(maindir,'Fitted','fitteddata.h5')
    (sensdict,simparams) = readconfigfile(configfile)

    paramslower = [ip.lower() for ip in params]
    Np = len(params)

    
    Ionofit = IonoContainer.readh5(ffit)
    times = Ionofit.Time_Vector
    Nt = len(times)
    dataloc = Ionofit.Sphere_Coords
    pnames = Ionofit.Param_Names
    pnameslower = sp.array([ip.lower() for ip in pnames.flatten()])
    p2fit = [sp.argwhere(ip==pnameslower)[0][0] if ip in pnameslower else None for ip in paramslower]
        
    angles = dataloc[:,1:]
    b = np.ascontiguousarray(angles).view(np.dtype((np.void, angles.dtype.itemsize * angles.shape[1])))
    _, idx, invidx = np.unique(b, return_index=True,return_inverse=True)

    beamlist = angles[idx]

    Nb = beamlist.shape[0]

    
    nfig = int(sp.ceil(Nt*Nb*Np/9.0))
    imcount = 0
    for i_fig in range(nfig):
        lines = [None]*2
        labels = [None]*2
        (figmplf, axmat) = plt.subplots(3, 3,figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for iax,ax in enumerate(axvec):
            if imcount>=Nb*Np:
                break
            iparam = int(sp.floor(imcount/Nb))
            ibeam = int(imcount-iparam*Nb)

            curbeam = beamlist[ibeam]
            curparm = paramslower[iparam]
            if curparm == 'nepow':
                curparm = 'ne'
            prmloc = sp.argwhere(curparm==pnameslowerin)
            
            indxkep = np.argwhere(invidx==ibeam)[:,0]
            rng_fit= dataloc[indxkep,0]
            rngargs = np.argsort(rng_fit)
            rng_fit = rng_fit[rngargs]
            alt_fit = rng_fit*sp.sin(curbeam[1]*sp.pi/180.)
            curfit = Ionofit.Param_List[indxkep,:,p2fit[iparam]]
            curfit=curfit[rngargs]
            Tmat, Amat =np.meshgrid(times[:,0],alt_fit)
            image = ax.pcolor(Tmat,Amat,curfit,cmat='plasma')
            if curparm=='ne':
                image.set_norm(colors.LogNorm(vmin=1e9,vmax=1e11))
                cbarstr = params[iparam] + ' m-3'
            else:
                image.set_norm(colors.PowerNorm(gamma=1.,vmin=500,vmax=3e3))
                cbarstr = params[iparam] + ' K'
            cbar = plt.colorbar()
            cbar.set_label(cbarstr)
            ax.set_xlabel("Time in s")
            ax.set_ylabel('Alt km')
            ax.set_title('{0} vs Altitude, Az: {1}$^o$ El: {2}$^o$'.format(params[iparam],*curbeam))
            imcount=imcount+1

        figmplf.suptitle(suptitle, fontsize=20)

        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
        fname= filetemplate+'_{0:0>3}.png'.format(i_fig)
        plt.savefig(fname)
        plt.close(figmplf)
    
def fitsurfaceplot(paramdict,plotvals,configfile,y_acf,yerr=None,filetemplate='fitsurfs',suptitle = 'Fit Surfaces'):

    (sensdict,simparams) = readconfigfile(configfile)
    specs = simparams['species']
    nspecs = len(specs)

    # make param lists
    paramlist = [[]]*(2*nspecs+1)
    paramlist[2*(nspecs-1)] =paramdict['Ne']
    paramlist[2*(nspecs-1)+1] =paramdict['Te']

    if 'frac' in paramdict.keys():
        frac = paramdict['frac']
    else:
        frac = [[1./(nspecs-1)]]*(nspecs-1)

    for ispec in range(nspecs-1):
        paramlist[2*ispec] =frac[ispec]
        paramlist[2*ispec+1] =  paramdict['Ti'][ispec]

    if 'Vi' in paramdict.keys():
        paramlist[-1] = paramdict['Vi']
    else:
        paramlist[-1] =[0.]

    pvals = {'Ne':2*(nspecs-1),'Te':2*(nspecs-1)+1,'Ti':1,'frac':0}

    fitsurfs= makefitsurf(paramlist,y_acf,sensdict,simparams,yerr)
    quad = (3,3)
    i_fig=0
    for iplt, idict in enumerate(plotvals):
        iaxn = sp.mod(iplt,sp.prod(quad))

        if iaxn==0:
            (figmplf, axmat) = plt.subplots(quad[0],quad[1],figsize=(20, 15), facecolor='w')
            axvec = axmat.flatten()

        setstr = idict['setparam']
        xstr = idict['xparam']
        ystr = idict['yparam']
        mloc = pvals[setstr]
        xdim = pvals[xstr]
        ydim = pvals[ystr]
        setval = paramlist[setstr][idict['indx']]
        transarr = sp.arange(2*nspecs+1).tolist()
        transarr.remove(mloc)
        transarr.remove(xdim)
        transarr.remove(ydim)
        transarr = [mloc,ydim,xdim] +transarr
        fitupdate = sp.transpose(fitsurfs,transarr)
        while fitupdate.ndim>3:
            fitupdate = sp.nanmean(fitupdate,dim=-1)
        Z1 = fitupdate[idict['indx']]
        iax = axvec[iaxn]
        xvec = paramdict[xstr]
        yvec = paramdict[ystr]
        [Xmat,Ymat]= sp.meshgrid(xvec,yvec)

        iax.pcolor(Xmat,Ymat,Z1,norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
        iax.xlabel=xstr
        iax.ylabel=ystr
        iax.title('{0} at {0}'.format(setstr,setval))
        if iaxn ==sp.prod(quad)-1:
            figmplf.suptitle(suptitle, fontsize=20)
            fname= filetemplate+'_{0:0>4}.png'.format(i_fig)
            plt.savefig(fname)
            plt.close(figmplf)
            i_fig+=1




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

def plotbeamparameters(times,configfile,maindir,params=['Ne'],indisp=True,fitdisp = True,filetemplate='params',suptitle = 'Parameter Comparison',werrors=False):
    """ """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
#    rc('text', usetex=True)
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
                errorexist = 'n'+paramslower[iparam] in pnameslower
                if errorexist and werrors:
                    eparam = sp.argwhere( 'n'+paramslower[iparam]==pnameslower)[0][0]
                    curerror = Ionofit.Param_List[indxkep,time2fit[itime],eparam]
                    lines[1]=ax.errorbar(curfit, alt_fit, xerr=curerror,fmt='-.',c='g')[0]
                else:
                    lines[1]= ax.plot(curfit,alt_fit,marker='.',c='g')[0]
                labels[1] = 'Fitted Parameters'

            if indisp:
                filenum = time2file[itime]
                if curfilenum!=filenum:
                    curfilenum=filenum
                    datafilename = dirlist[filenum]
                    Ionoin = IonoContainer.readh5(datafilename)
                    if 'ti' in paramslower:
                        Ionoin = maketi(Ionoin)
                    pnames = Ionoin.Param_Names
                    pnameslowerin = sp.array([ip.lower() for ip in pnames.flatten()])

                curparm = paramslower[iparam]
                if curparm == 'nepow':
                    curparm = 'ne'
                prmloc = sp.argwhere(curparm==pnameslowerin)

                if prmloc.size !=0:
                    curprm = prmloc[0][0]

                curcoord = sp.zeros(3)
                curcoord[1:] = curbeam
                curdata = sp.zeros(len(rng))
                for irngn, irng in enumerate(rng):
                    curcoord[0] = irng
                    tempin = Ionoin.getclosestsphere(curcoord,times)[0]
                    Ntloc = tempin.shape[0]
                    tempin = sp.reshape(tempin,(Ntloc,len(pnameslowerin)))
                    curdata[irngn] = tempin[0,curprm]
                lines[0]= ax.plot(curdata,altlist,marker='o',c='b')[0]
                labels[0] = 'Input Parameters'
                if curparm!='ne':
                    ax.set(xlim=[0.25*sp.amin(curdata),2.5*sp.amax(curdata)])
            if curparm=='ne':
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

def plotspecs(coords,times,configfile,maindir,cartcoordsys = True, indisp=True,acfdisp= True,
              fitdisp=True,filetemplate='spec',suptitle = 'Spectrum Comparison'):
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

    acfname = os.path.join(maindir,'ACF','00lags.h5')
    ffit = os.path.join(maindir,'Fitted','fitteddata.h5')
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

                specin[icn,itn] = tempin[0,:]/npts/npts
    fs = sensdict['fs']

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

    if fitdisp:
        Ionofit = IonoContainer.readh5(ffit)
        (omegfit,outspecsfit) =ISRspecmakeout(Ionofit.Param_List,sensdict['fc'],sensdict['fs'],simparams['species'],npts)
        Ionofit.Param_List= outspecsfit
        Ionofit.Param_Names = omegfit
        specfit = sp.zeros((Nloc,Nt,npts))
        for icn, ic in enumerate(coords):
            if cartcoordsys:
                tempin = Ionofit.getclosest(ic,times)[0]
            else:
                tempin = Ionofit.getclosestsphere(ic,times)[0]
            if sp.ndim(tempin)==1:
                tempin = tempin[sp.newaxis,:]
            specfit[icn] = tempin/npts/npts


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
            if fitdisp:
                curfitspec = specfit[iloc,itime]
                rcsfit = curfitspec.sum()
                (taufit,acffit) = spect2acf(omegfit,curfitspec)
                guess_acffit = sp.dot(amb_dict['WttMatrix'],acffit)
                guess_acffit = guess_acffit*rcsfit/guess_acffit[0].real
                spec_intermfit = scfft.fftshift(scfft.fft(guess_acffit,n=npts))
                lines[1]= ax.plot(omeg*1e-3,spec_intermfit.real,label='Fitted Spectrum',linewidth=5)[0]
                labels[1] = 'Fitted Spectrum'
            if indisp:
                # apply ambiguity function to spectrum
                curin = specin[iloc,itime]
                rcs = curin.real.sum()
                (tau,acf) = spect2acf(omeg,curin)
                guess_acf = sp.dot(amb_dict['WttMatrix'],acf)

                guess_acf = guess_acf*rcs/guess_acf[0].real

                # fit to spectrums
                spec_interm = scfft.fftshift(scfft.fft(guess_acf,n=npts))
                maxvec.append(spec_interm.real.max())
                lines[0]= ax.plot(omeg*1e-3,spec_interm.real,label='Input',linewidth=5)[0]
                labels[0] = 'Input Spectrum With Ambiguity Applied'

            if acfdisp:
                lines[2]=ax.plot(omeg*1e-3,specout[iloc,itime].real,label='Output',linewidth=5)[0]
                labels[2] = 'Estimated Spectrum'
                maxvec.append(specout[iloc,itime].real.max())
            ax.set_xlabel('f in kHz')
            ax.set_ylabel('Amp')
            ax.set_title('Location {0}, Time {1}'.format(coords[iloc],times[itime]))
            ax.set_ylim(0.0,max(maxvec)*1)
            ax.set_xlim([-fs*5e-4,fs*5e-4])
            imcount=imcount+1
        figmplf.suptitle(suptitle, fontsize=20)
        if None in labels:
            labels.remove(None)
            lines.remove(None)
        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
        fname= filetemplate+'_{0:0>3}.png'.format(i_fig)

        plt.savefig(fname)
        plt.close(figmplf)

def plotacfs(coords,times,configfile,maindir,cartcoordsys = True, indisp=True,acfdisp= True,
             fitdisp=True, filetemplate='acf',suptitle = 'ACF Comparison'):

    """ This will create a set of images that compare the input ISR acf to the
    output ISR acfs from the simulator.
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
    ffit = os.path.join(maindir,'Fitted','fitteddata.h5')

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
    pulse = simparams['Pulse']
    ts = sensdict['t_s']
    tau1 = sp.arange(pulse.shape[-1])*ts
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
                specin[icn,itn] = tempin[0,:]/npts
    if acfdisp:
        Ionoacf = IonoContainer.readh5(acfname)
        ACFin = sp.zeros((Nloc,Nt,Ionoacf.Param_List.shape[-1])).astype(Ionoacf.Param_List.dtype)

        omeg = sp.arange(-sp.ceil((npts+1)/2),sp.floor((npts+1)/2))/ts/npts
        for icn, ic in enumerate(coords):
            if cartcoordsys:
                tempin = Ionoacf.getclosest(ic,times)[0]
            else:
                tempin = Ionoacf.getclosestsphere(ic,times)[0]
            if sp.ndim(tempin)==1:
                tempin = tempin[sp.newaxis,:]
            ACFin[icn] = tempin


    if fitdisp:
        Ionofit = IonoContainer.readh5(ffit)
        (omegfit,outspecsfit) =ISRspecmakeout(Ionofit.Param_List,sensdict['fc'],sensdict['fs'],simparams['species'],npts)
        Ionofit.Param_List= outspecsfit
        Ionofit.Param_Names = omegfit
        specfit = sp.zeros((Nloc,Nt,npts))
        for icn, ic in enumerate(coords):
            if cartcoordsys:
                tempin = Ionofit.getclosest(ic,times)[0]
            else:
                tempin = Ionofit.getclosestsphere(ic,times)[0]
            if sp.ndim(tempin)==1:
                tempin = tempin[sp.newaxis,:]
            specfit[icn] = tempin/npts/npts

    nfig = int(sp.ceil(Nt*Nloc/6.0))
    imcount = 0

    for i_fig in range(nfig):
        lines = [None]*6
        labels = [None]*6
        (figmplf, axmat) = plt.subplots(2, 3,figsize=(24, 18), facecolor='w')
        axvec = axmat.flatten()
        for iax,ax in enumerate(axvec):
            if imcount>=Nt*Nloc:
                break
            iloc = int(sp.floor(imcount/Nt))
            itime = int(imcount-(iloc*Nt))

            maxvec = []
            minvec = []

            if indisp:
                # apply ambiguity funciton to spectrum
                curin = specin[iloc,itime]
                (tau,acf) = spect2acf(omeg,curin)
                acf1 = scfft.ifftshift(acf)[:len(pulse)]
                rcs=acf1[0].real
                guess_acf = sp.dot(amb_dict['WttMatrix'],acf)
                guess_acf = guess_acf*rcs/guess_acf[0].real

                # fit to spectrums
                maxvec.append(guess_acf.real.max())
                maxvec.append(guess_acf.imag.max())
                minvec.append(acf1.real.min())
                minvec.append(acf1.imag.min())
                lines[0]= ax.plot(tau1*1e6,guess_acf.real,label='Input',linewidth=5)[0]
                labels[0] = 'Input ACF With Ambiguity Applied Real'
                lines[1]= ax.plot(tau1*1e6,guess_acf.imag,label='Input',linewidth=5)[0]
                labels[1] = 'Input ACF With Ambiguity Applied Imag'

            if fitdisp:
                curinfit = specfit[iloc,itime]
                (taufit,acffit) = spect2acf(omegfit,curinfit)
                rcsfit=acffit[0].real
                guess_acffit = sp.dot(amb_dict['WttMatrix'],acffit)
                guess_acffit = guess_acffit*rcsfit/guess_acffit[0].real

                lines[2]= ax.plot(tau1*1e6,guess_acffit.real,label='Input',linewidth=5)[0]
                labels[2] = 'Fitted ACF real'
                lines[3]= ax.plot(tau1*1e6,guess_acffit.imag,label='Input',linewidth=5)[0]
                labels[3] = 'Fitted ACF Imag'
            if acfdisp:
                lines[4]=ax.plot(tau1*1e6,ACFin[iloc,itime].real,label='Output',linewidth=5)[0]
                labels[4] = 'Estimated ACF Real'
                lines[5]=ax.plot(tau1*1e6,ACFin[iloc,itime].imag,label='Output',linewidth=5)[0]
                labels[5] = 'Estimated ACF Imag'

                maxvec.append(ACFin[iloc,itime].real.max())
                maxvec.append(ACFin[iloc,itime].imag.max())
                minvec.append(ACFin[iloc,itime].real.min())
                minvec.append(ACFin[iloc,itime].imag.min())
            ax.set_xlabel('t in us')
            ax.set_ylabel('Amp')
            ax.set_title('Location {0}, Time {1}'.format(coords[iloc],times[itime]))
            ax.set_ylim(min(minvec),max(maxvec)*1)
            ax.set_xlim([tau1.min()*1e6,tau1.max()*1e6])
            imcount=imcount+1
        figmplf.suptitle(suptitle, fontsize=20)
        if None in labels:
            labels.remove(None)
            lines.remove(None)
        plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
        fname= filetemplate+'_{0:0>3}.png'.format(i_fig)
        plt.savefig(fname)
        plt.close(figmplf)

def plotspecsgen(timeomeg,speclist,needtrans,specnames=None,filename='specs.png',n=None):
    fig1 = plt.figure()
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    lines = []
    if specnames is None:
        specnames = ['Spec {0}'.format(i) for i in range(len(speclist))]
    labels = specnames
    xlims = [sp.Inf,-sp.Inf]
    ylims = [sp.Inf,-sp.Inf]
    for ispecn,ispec in enumerate(speclist):
        if type(timeomeg)==list:
            curbasis = timeomeg[ispecn]
        else:
            curbasis=timeomeg

        if needtrans[ispecn]:
           curbasis,ispec= acf2spect(curbasis,ispec,n=n)

        lines.append(plt.plot(curbasis*1e-3,ispec.real,linewidth=5)[0])
        xlims = [min(xlims[0],min(curbasis)*1e-3),max(xlims[1],max(curbasis)*1e-3)]
        ylims = [min(ylims[0],min(ispec.real)),max(ylims[1],max(ispec.real))]
    plt.xlabel('f in kHz')
    plt.ylabel('Amp')
    plt.title('Output Spectrums')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.legend(lines,labels)
    plt.savefig(filename)
    plt.close(fig1)

def analysisdump(maindir,configfile,suptitle=None):
    """ """
    plotdir = os.path.join(maindir,'AnalysisPlots')
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    #plot spectrums
    filetemplate1 = os.path.join(maindir,'AnalysisPlots','Spec')
    filetemplate3 = os.path.join(maindir,'AnalysisPlots','ACF')

    (sensdict,simparams) = readconfigfile(configfile)
    angles = simparams['angles']
    ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
    if sensdict['Name'].lower() in ['risr','pfisr']:
        ang_data_temp = ang_data.copy()
        beamlistlist = sp.array(simparams['outangles']).astype(int)
        ang_data = sp.array([ang_data_temp[i].mean(axis=0)  for i in beamlistlist ])
    
    zenang = ang_data[sp.argmax(ang_data[:,1])]
    rnggates = simparams['Rangegatesfinal']
    rngchoices = sp.linspace(sp.amin(rnggates),sp.amax(rnggates),4)
    angtile = sp.tile(zenang,(len(rngchoices),1))
    coords = sp.column_stack((sp.transpose(rngchoices),angtile))
    times = simparams['Timevec']


    filetemplate2= os.path.join(maindir,'AnalysisPlots','Params')
    if simparams['Pulsetype'].lower()=='barker':
        if suptitle is None:
            plotbeamparameters(times,configfile,maindir,params=['Ne'],filetemplate=filetemplate2,werrors=True)
        else:
            plotbeamparameters(times,configfile,maindir,params=['Ne'],filetemplate=filetemplate2,suptitle=suptitle,werrors=True)
    else:
        if suptitle is None:
            plotspecs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate1)
            plotacfs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate3)
            plotbeamparameters(times,configfile,maindir,params=['Ne','Nepow','Te','Ti'],filetemplate=filetemplate2,werrors=True)
        else:
            plotspecs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate1,suptitle=suptitle)
            plotacfs(coords,times,configfile,maindir,cartcoordsys = False, filetemplate=filetemplate3,suptitle=suptitle)
            plotbeamparameters(times,configfile,maindir,params=['Ne','Nepow','Te','Ti'],filetemplate=filetemplate2,suptitle=suptitle,werrors=True)