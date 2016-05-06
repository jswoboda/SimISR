#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a set of data 
@author: John Swoboda
"""
import os,inspect,glob
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from RadarDataSim.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf,makeconfigfile
from RadarDataSim.IonoContainer import IonoContainer
from  RadarDataSim.runsim import main as runsim 
from RadarDataSim.analysisplots import analysisdump,maketi
from radarsystools.radarsystools import RadarSys

def makehist(testpath,npulses):
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne','Nepow','Te','Ti','Vi'] 
    errdict = makehistdata(params,testpath)
    ernames = ['Data','Error','Error Percent']
    filetemplate= os.path.join(testpath,'AnalysisPlots')
    
    for ierr, iername in enumerate(ernames):
        (figmplf, axmat) = plt.subplots(3, 2,figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for ipn, iparam in enumerate(params):
            plt.scs(axvec[ipn])
            histhand = sns.distplot(errdict[iparam], bins=50, kde=False, rug=False)
            axvec[ipn].set_title(iparam)
        figmplf.suptitle(iername, fontsize=20)
        fname= filetemplate+ierr+'_{0:0>5}Pulses.png'.format(npulses)
        plt.savefig(fname)
        plt.close(figmplf)
def makehistdata(params,maindir):
    
    
    ffit = os.path.join(maindir,'Fitted','fitteddata.h5')
    inputfiledir = os.path.join(maindir,'Origparams')

    paramslower = [ip.lower() for ip in params]
    
    # set up data dictionary
    
    errordict = {ip:[] for ip in params}
    errordictrel = {ip:[] for ip in params}
    #Read in fitted data
    
    Ionofit = IonoContainer.readh5(ffit)
    times=Ionofit.Time_Vector
    
    dataloc = Ionofit.Sphere_Coords
    pnames = Ionofit.Param_Names
    pnameslower = sp.array([ip.lower() for ip in pnames.flatten()])
    p2fit = [sp.argwhere(ip==pnameslower)[0][0] if ip in pnameslower else None for ip in paramslower]
    
    datadict = {ip:Ionofit.Param_List[:,:,p2fit[ipn]].flatten() for ipn, ip in enumerate(params)}
    
    
    # Determine which imput files are to be used.
    
    dirlist = glob.glob(os.path.join(inputfiledir,'*.h5'))
    sortlist,outime,filelisting,timebeg,timelist_s = IonoContainer.gettimes(dirlist)
    time2files = []
    for itn,itime in enumerate(times):
        log1 = (outime[:,0]>=itime[0]) & (outime[:,0]<itime[1])
        log2 = (outime[:,1]>itime[0]) & (outime[:,1]<=itime[1])
        tempindx = sp.where(log1|log2)[0]
        time2files.append(filelisting[tempindx])
    
    
    curfilenum=-1
    for iparam,pname in enumerate(params):            
        curparm = paramslower[iparam]
        # Use Ne from input to compare the ne derived from the power.
        if curparm == 'nepow':
            curparm = 'ne'
        datalist = []
        for itn,itime in enumerate(times):
            for iplot,filenum in enumerate(time2files[itime]):
                        
                if curfilenum!=filenum:
                    curfilenum=filenum
                    datafilename = dirlist[filenum]
                    Ionoin = IonoContainer.readh5(datafilename)
                    if ('ti' in paramslower) or ('vi' in paramslower):
                        Ionoin = maketi(Ionoin)
                    pnames = Ionoin.Param_Names
                    pnameslowerin = sp.array([ip.lower() for ip in pnames.flatten()])
                prmloc = sp.argwhere(curparm==pnameslowerin)
                if prmloc.size !=0:
                    curprm = prmloc[0][0]
                # build up parameter vector bs the range values by finding the closest point in space in the input
                curdata = sp.zeros(len(dataloc))
                
                for irngn, curcoord in enumerate(dataloc):
                    
                    tempin = Ionoin.getclosestsphere(curcoord,[times[itime]])[0]
                    Ntloc = tempin.shape[0]
                    tempin = sp.reshape(tempin,(Ntloc,len(pnameslowerin)))
                    curdata[irngn] = tempin[0,curprm]
                datalist.append(curdata)
        errordict[pname] = datadict[pname]-sp.vstack(datalist)  
        errordictrel[pname] = 100.*errordict[pname]/sp.absolute(sp.vstack(datalist))
    return datadict,errordict,errordictrel

def configfilesetup(testpath,npulses):
    """ """
    
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    defcon = os.path.join(curloc,'statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = ratio1 * simparams['TimeLim']
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(os.path.join(testpath,'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
def makedata(testpath):
    """ """
    finalpath = os.path.join(testpath,'Origparams')
    if not os.path.isdir(finalpath):
        os.mkdir(finalpath)
    data = sp.array([1e12,2500.])
    z = sp.linspace(50.,1e3,50)
    nz = len(z)
    params = sp.tile(data[sp.newaxis,sp.newaxis,sp.newaxis,:],(nz,1,2,1))
    coords = sp.column_stack((sp.ones(nz),sp.ones(nz),z))
    species=['O+','e-']
    times = sp.array([[0,1e3]])
    vel = sp.zeros((nz,1,3))
    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
        
    finalfile = os.path.join(finalpath,'0 stats.h5')
    Icont1.saveh5(finalfile)
    Icont1.saveh5(os.path.join(testpath,'startfile.h5'))
    
def getinfo(curdir):
    """ """
    origdataname = os.path.join(curdir,'Origparams','0 stats.h5')
    fittedname=glob.glob(os.path.join(curdir,'Fitted','*.h5'))[0]
    
    # Get Parameters
    Origparam = IonoContainer.readh5(origdataname)
    fittedparam = IonoContainer.readh5(fittedname)
    
    Ne_real = Origparam.Param_List[0,0,1,0]
    Te_real = Origparam.Param_List[0,0,1,1]
    Ti_real = Origparam.Param_List[0,0,0,1]
    
    realdict = {'Ne':Ne_real,'Te':Te_real,'Ti':Ti_real,'Nepow':Ne_real}
    paramnames = fittedparam.Param_Names
    params = ['Ne','Te','Ti','Nepow']
    params_loc = [sp.argwhere(i==sp.array(paramnames))[0][0] for i in params]
    
    datadict = {i:fittedparam.Param_List[:,:,j] for i,j in zip(params,params_loc)}
    datavars = {i:sp.nanvar(datadict[i],axis=1) for i in params}
    dataerror = {i:sp.nanmean((datadict[i]-realdict[i])**2,axis=1) for i in params}
    
    return (datadict,datavars,dataerror, realdict)
def main(plist = None,functlist = ['spectrums','radardata','fitting']):

    if plist is None:
        plist = sp.array([50,100,200,500,1000,2000,5000])
        
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curloc)[0],'Testdata','StatsTest')
    
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()
    allfolds = []
    rsystools = []
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold =os.path.join(testpath,foldname)
        allfolds.append(curfold)
        if not os.path.isdir(curfold):
            os.mkdir(curfold)
            makedata(curfold)
            configfilesetup(curfold,ip)
        config = os.path.join(curfold,'stats.ini')
        (sensdict,simparams) = readconfigfile(config)
        rtemp = RadarSys(sensdict,simparams['Rangegatesfinal'],ip)
        rsystools.append(rtemp.rms(sp.array([1e12]),sp.array([2.5e3]),sp.array([2.5e3])))
        if check_run :
            runsim(functlist_red,curfold,os.path.join(curfold,'stats.ini'),True)
        if 'analysis' in functlist:
            analysisdump(curfold,)
        if 'stats' in functlist:
            datadict,datavars,dataerror,realdict = getinfo(curfold)
            
        
    