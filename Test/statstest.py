#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a number of data sets for statistical analysis. It'll then make
statistics and histograms of the output parameters.
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
#from radarsystools.radarsystools import RadarSys

def makehist(testpath,npulses):
    """ This functions are will create histogram from data made in the testpath.
        Inputs
            testpath - The path that the data is located.
            npulses - The number of pulses in the sim.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne','Te','Ti','Vi'] 
    pvals = [1e11,1e11,2.5e3,2.5e3,0.]
    errdict = makehistdata(params,testpath)
    ernames = ['Data','Error','Error Percent']
    sig1 = sp.sqrt(1./npulses)
    
    
    for ierr, iername in enumerate(ernames):
        filetemplate= os.path.join(testpath,'AnalysisPlots',iername)
        (figmplf, axmat) = plt.subplots(2, 2,figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for ipn, iparam in enumerate(params):
            plt.sca(axvec[ipn])
            if sp.any(sp.isinf(errdict[ierr][iparam])):
                continue
            histhand = sns.distplot(errdict[ierr][iparam], bins=50, kde=True, rug=False)
            xlim = histhand.get_xlim()
            if ierr==0:
                x0=pvals[ipn]
            else:
                x0=0
            if ierr==2:
                sig=sig1*100.
            else:
                sig=sig1*pvals[ipn]
            x = sp.linspace(xlim[0],xlim[1],100)
            den1 = sp.stats.norm(x0,sig).pdf(x)
            #plt.plot(x,den1,'g--')
            
            axvec[ipn].set_title(iparam)
        figmplf.suptitle(iername +' Pulses: {0}'.format(npulses), fontsize=20)
        fname= filetemplate+'_{0:0>5}Pulses.png'.format(npulses)
        plt.savefig(fname)
        plt.close(figmplf)
        
def makehistdata(params,maindir):
    """ This will make the histogram data for the statistics.
        Inputs
            params -  A list of parameters that will have statistics created
            maindir - The directory that the simulation data is held.
        Outputs
            datadict - A dictionary with the data values in numpy arrays. The keys are param names.
            errordict - A dictionary with the data values in numpy arrays. The keys are param names.
            errdictrel -  A dictionary with the error values in numpy arrays, normalized by the correct value. The keys are param names.
    """
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
        log3 = (outime[:,0]<=itime[0]) & (outime[:,1]>itime[1])
        tempindx = sp.where(log1|log2|log3)[0]
        time2files.append(filelisting[tempindx])
    
    
    curfilenum=-1
    for iparam,pname in enumerate(params):            
        curparm = paramslower[iparam]
        # Use Ne from input to compare the ne derived from the power.
        if curparm == 'nepow':
            curparm = 'ne'
        datalist = []
        for itn,itime in enumerate(times):
            for iplot,filenum in enumerate(time2files[itn]):
                filenum = int(filenum)
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
                    
                    tempin = Ionoin.getclosestsphere(curcoord,[itime])[0]
                    Ntloc = tempin.shape[0]
                    tempin = sp.reshape(tempin,(Ntloc,len(pnameslowerin)))
                    curdata[irngn] = tempin[0,curprm]
                datalist.append(curdata)
        errordict[pname] = datadict[pname]-sp.hstack(datalist)  
        errordictrel[pname] = 100.*errordict[pname]/sp.absolute(sp.hstack(datalist))
    return datadict,errordict,errordictrel

def configfilesetup(testpath,npulses):
    """ This will create the configureation file given the number of pulses for 
        the test. This will make it so that there will be 12 integration periods 
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses. 
    """
    
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
    """ This will make the input data for the test case. The data will have the 
        default set of parameters Ne=Ne=1e11 and Te=Ti=2000.
        Inputs
            testpath - Directory that will hold the data.
            
    """
    finalpath = os.path.join(testpath,'Origparams')
    if not os.path.isdir(finalpath):
        os.mkdir(finalpath)
    data = sp.array([1e11,2500.])
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
    

def main(plist = None,functlist = ['spectrums','radardata','fitting','analysis','stats']):
    """ This function will call other functions to create the input data, config
        file and run the radar data sim. The path for the simulation will be 
        created in the Testdata directory in the RadarDataSim module. The new
        folder will be called BasicTest. The simulation is a long pulse simulation
        will the desired number of pulses from the user.
        Inputs
            npulse - Number of pulses for the integration period, default==100.
            functlist - The list of functions for the RadarDataSim to do.
    """
    if plist is None:
        plist = sp.array([50,100,200,500,1000,2000,5000])
    if isinstance(plist,list):
        plist=sp.array(plist)
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    testpath = os.path.join(os.path.split(curloc)[0],'Testdata','StatsTest')
    
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()
    allfolds = []
#    rsystools = []
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold =os.path.join(testpath,foldname)
        allfolds.append(curfold)
        if not os.path.isdir(curfold):
            os.mkdir(curfold)
            
            configfilesetup(curfold,ip)
        makedata(curfold)
        config = os.path.join(curfold,'stats.ini')
        (sensdict,simparams) = readconfigfile(config)
#        rtemp = RadarSys(sensdict,simparams['Rangegatesfinal'],ip)
#        rsystools.append(rtemp.rms(sp.array([1e12]),sp.array([2.5e3]),sp.array([2.5e3])))
        if check_run :
            runsim(functlist_red,curfold,os.path.join(curfold,'stats.ini'),True)
        if 'analysis' in functlist:
            analysisdump(curfold,config)
        if 'stats' in functlist:
            makehist(curfold,ip)
            
if __name__== '__main__':
    from argparse import ArgumentParser
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    p = ArgumentParser(description=descr)
    
    p.add_argument("-p", "--npulses",help='Number of pulses.',nargs='+',type=int,default=[50,100,200,500,1000,2000,5000])
    p.add_argument('-f','--funclist',help='Functions to be uses',nargs='+',default=['spectrums','radardata','fitting','analysis','stats'])#action='append',dest='collection',default=['spectrums','radardata','fitting','analysis'])
    
    p = p.parse_args()
    
    main(p.npulses,p.funclist)     
    
