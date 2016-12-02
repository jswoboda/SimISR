#!/usr/bin/env python
"""
singleacf.py
This module is used to plot and do other things
@author: John Swoboda
"""

import scipy as sp
import scipy.optimize
import scipy.fftpack as scfft
from ISRSpectrum.ISRSpectrum import ISRSpectrum
from RadarDataSim.specfunctions import ISRSfitfunction
from RadarDataSim.utilFunctions import MakePulseDataRepLPC, CenteredLagProduct, readconfigfile
from RadarDataSim.IonoContainer import IonoContainer
import matplotlib.pyplot as plt
import seaborn as sns

def fitcheck(repall=[100]):
    x_0=sp.array([[[  1.00000000e+11,   2.00000000e+03],
                [  1.00000000e+11,   2.00000000e+03]],

               [[  5.00000000e+11,   2.00000000e+03],
                [  5.00000000e+11,   2.00000000e+03]],

               [[  1.00000000e+11,   3.00000000e+03],
                [  1.00000000e+11,   2.00000000e+03]],

               [[  1.00000000e+11,   2.00000000e+03],
                [  1.00000000e+11,   3.00000000e+03]]])
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    x_0_red = x_0[0].flatten()
    x_0_red[-2]=x_0_red[-1]
    x_0_red[-1]=0.
    configfile = 'statsbase.ini'
    (sensdict,simparams) = readconfigfile(configfile)
    ambdict = simparams['amb_dict']
    pulse = simparams['Pulse']
    l_p = len(pulse)
    Nlags = l_p
    lagv = sp.arange(l_p)
    ntypes = x_0.shape[0]
    nspec = 128
    nrg = 64
    des_pnt = 16
    ISpec = ISRSpectrum(nspec=nspec,sampfreq=50e3)
    species = ['O+','e-']
    spvtime=sp.zeros((ntypes,nspec))
    lablist =['Normal','E-Ne','E-Ti','E-Te']
    v_i = 0
    fitfunc=ISRSfitfunction

    sumrule = simparams['SUMRULE']
    Nrg1 = nrg+1-l_p
    minrg = -sumrule[0].min()
    maxrg = Nrg1-sumrule[1].max()
    Nrg2 = maxrg-minrg





    
    for i in range(ntypes):
        f,curspec,rcs = ISpec.getspecsep(x_0[i],species,v_i,rcsflag=True)
        specsum = sp.absolute(curspec).sum()
        spvtime[i] = rcs*curspec*nspec**2/specsum
        
    acforig = scfft.ifft(scfft.ifftshift(spvtime,axes=1),axis=1)/nspec
    acfamb = sp.dot(ambdict['WttMatrix'],scfft.fftshift(acforig,axes=1).transpose()).transpose()
    specamb = scfft.fftshift(scfft.fft(acfamb,n=nspec,axis=1),axes=1)
    
    fig, axmat = plt.subplots(2,2)
    axvec=axmat.flatten()

    figs,axmats = plt.subplots(2,2)
    axvecs = axmats.flatten()
    
    for i in range(ntypes):
        ax=axvec[i]                                  
        ax.plot(lagv,acfamb[i].real,label='Input')
        ax.set_title(lablist[i])
        axs=axvecs[i]                                  
        axs.plot(f*1e-3,specamb[i].real,label='Input',linewidth=4)
        axs.set_title(lablist[i])
        

    for irep, rep1 in enumerate(repall):
        rawdata = sp.zeros((ntypes,rep1,nrg),dtype=sp.complex128)
        acfest = sp.zeros((ntypes,Nrg1,l_p),dtype=rawdata.dtype)
        acfestsr = sp.zeros((ntypes,Nrg2,l_p),dtype=rawdata.dtype)
        specest = sp.zeros((ntypes,nspec),dtype=rawdata.dtype)
        for i in range(ntypes):
            for j in range(nrg-(l_p-1)):
                rawdata[i,:,j:j+l_p] = MakePulseDataRepLPC(pulse,spvtime[i],20,rep1)+rawdata[i,:,j:j+l_p]
            acfest[i]=CenteredLagProduct(rawdata[i],pulse=pulse)/(rep1)
            for irngnew,irng in enumerate(sp.arange(minrg,maxrg)):
                for ilag in range(Nlags):
                    acfestsr[i][irngnew,ilag] = acfest[i][irng+sumrule[0,ilag]:irng+sumrule[1,ilag]+1,ilag].mean(axis=0)
            ax=axvec[i]
            ax.plot(lagv,acfestsr[i,des_pnt].real/l_p,label='Np = {0}'.format(rep1))
            if irep==len(repall)-1:
                ax.legend()
            specest[i] = scfft.fftshift(scfft.fft(acfestsr[i,des_pnt],n=nspec))
            axs=axvecs[i]
            axs.plot(f*1e-3,specest[i].real/l_p,label='Np = {0}'.format(rep1),linewidth=4)
            if irep==len(repall)-1:
                axs.legend()
        print('Parameters fitted after {0} pulses'.format(rep1))
        print('Ni                Ti             Te                 Vi')
        for i in range(ntypes):
            d_func = (acfestsr[i,des_pnt]/l_p,sensdict,simparams)
            (x,cov_x,infodict,mesg,ier) = scipy.optimize.leastsq(func=fitfunc,
                                                         x0=x_0_red,args=d_func,full_output=True)    
            print(x)
        print(' ')
    fig.suptitle('ACF with Sum Rule')
    fig.savefig('pulsetestacf.png',dpi=400)
    plt.close(fig)
    figs.suptitle('Spectrum Full Array')
    figs.savefig('pulsetestspec.png',dpi=400)
    plt.close(figs)

