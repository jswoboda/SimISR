#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a number of data sets for statistical analysis. It'll then make
statistics and histograms of the output parameters.
@author: John Swoboda
"""
from RadarDataSim import Path
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from RadarDataSim.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf,makeconfigfile
from RadarDataSim.IonoContainer import IonoContainer
from  RadarDataSim.runsim import main as runsim 
from RadarDataSim.analysisplots import analysisdump,maketi
#from radarsystools.radarsystools import RadarSys
PVALS = [1e11,2.1e3,1.1e3,0.]
SIMVALUES = sp.array([[PVALS[0],PVALS[2]],[PVALS[0],PVALS[1]]])
def makehistmult(testpathlist,npulseslist):
    
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne','Te','Ti','Vi']
    paramsLT = ['N_e','T_e','T_i','V_i']
    errdictlist =[ makehistdata(params,itest)[0] for itest in testpathlist] 
    (figmplf, axmat) = plt.subplots(2, 2,figsize=(12,8), facecolor='w')
    axvec = axmat.flatten()
    histlims = [[4e10,2e11],[1200.,3000.],[300.,1900.],[-250.,250.]]
    histvecs = [sp.linspace(ipm[0],ipm[1],100) for ipm in histlims]
    linehand = []
    lablist= ['J = {:d}'.format(i) for i in npulseslist]
    
    for iax,iparam in enumerate(params):
        for idict,inpulse in zip(errdictlist,npulseslist):
            curvals = idict[iparam]
            curhist,binout = sp.histogram(curvals,bins=histvecs[iax])
            dx=binout[1]-binout[0]
            curhist_norm = curhist.astype(float)/(curvals.size*dx)
            plthand = axvec[iax].plot(binout[:-1],curhist_norm,label='J = {:d}'.format(inpulse))[0]
            linehand.append(plthand)
            
        axvec[iax].set_xlabel(r'$'+paramsLT[iax]+'$')
        axvec[iax].set_title(r'Histogram for $'+paramsLT[iax]+'$')
    leg = figmplf.legend(linehand[:len(npulseslist)],lablist)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    spti = figmplf.suptitle('Parameter Distributions',fontsize=18)
    return (figmplf,axvec,linehand)

def makehistsingle(testpath,npulses):
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne','Te','Ti','Vi']
    paramsLT = ['N_e','T_e','T_i','V_i']
    datadict,er1,er2,edatadict = makehistdata(params,testpath)
    (figmplf, axmat) = plt.subplots(2, 2,figsize=(12,8), facecolor='w')
    axvec = axmat.flatten()
    histlims = [[4e10,2e11],[1200.,3000.],[300.,1900.],[-250.,250.]]
    histvecs = [sp.linspace(ipm[0],ipm[1],100) for ipm in histlims]
    linehand = []
    
    lablist=['Histogram','Variance','Error']
    for iax,iparam in enumerate(params):
        
        mu= PVALS[iax]
        curvals = datadict[iparam]
        mu = sp.nanmean(curvals.real)
        RMSE = sp.sqrt(sp.nanvar(curvals))
        Error_mean = sp.sqrt(sp.nanmean(sp.power(edatadict[iparam],2)))
        curhist,x = sp.histogram(curvals,bins=histvecs[iax])
        dx=x[1]-x[0]
        curhist_norm = curhist.astype(float)/(curvals.size*dx)
        plthand = axvec[iax].plot(x[:-1],curhist_norm,'r-',label='Histogram'.format(npulses))[0]
        linehand.append(plthand)
        rmsedist = sp.stats.norm.pdf((x-mu)/RMSE)/RMSE
        plthand = axvec[iax].plot(x,rmsedist,label='Var'.format(npulses))[0]
        linehand.append(plthand)
        emeandist = sp.stats.norm.pdf((x-mu)/Error_mean)/Error_mean
        plthand = axvec[iax].plot(x,emeandist,label='Error'.format(npulses))[0]
        linehand.append(plthand)
        axvec[iax].set_xlabel(r'$'+paramsLT[iax]+'$')
        axvec[iax].set_title(r'Distributions for $'+paramsLT[iax]+'$')
    leg = figmplf.legend(linehand[:len(lablist)],lablist)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    spti = figmplf.suptitle('Pulses J = {:d}'.format(npulses),fontsize=18)
    return (figmplf,axvec,linehand)
def makehist(testpath,npulses):
    """ This functions are will create histogram from data made in the testpath.
        Inputs
            testpath - The path that the data is located.
            npulses - The number of pulses in the sim.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne','Te','Ti','Vi'] 
    pvals = [1e11,1e11,2.1e3,1.1e3,0.]
    errdict = makehistdata(params,testpath)[:4]
    ernames = ['Data','Error','Error Percent']
    sig1 = sp.sqrt(1./npulses)
    
    
    for ierr, iername in enumerate(ernames):
        filetemplate= str(Path(testpath).join('AnalysisPlots',iername))
        (figmplf, axmat) = plt.subplots(2, 2,figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for ipn, iparam in enumerate(params):
            plt.sca(axvec[ipn])
            if sp.any(sp.isinf(errdict[ierr][iparam])):
                continue
            histhand = sns.distplot(errdict[ierr][iparam], bins=100, kde=True, rug=False)
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
    maindir=Path(maindir)
    ffit = maindir.joinpath('Fitted','fitteddata.h5')
    inputfiledir = maindir.joinpath('Origparams')

    paramslower = [ip.lower() for ip in params]
    eparamslower = ['n'+ip.lower() for ip in params]
    
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
    
    ep2fit = [sp.argwhere(ip==pnameslower)[0][0] if ip in pnameslower else None for ip in eparamslower]
    
    edatadict = {ip:Ionofit.Param_List[:,:,ep2fit[ipn]].flatten() for ipn, ip in enumerate(params)}
    # Determine which input files are to be used.
    
    dirlist = [str(i) for i in inputfiledir.glob('*.h5')]
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
    return datadict,errordict,errordictrel,edatadict

def configfilesetup(testpath,npulses):
    """ This will create the configureation file given the number of pulses for 
        the test. This will make it so that there will be 12 integration periods 
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses. 
    """
    
    curloc = Path(__file__).parent
    defcon = curloc.joinpath('statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = ratio1 * simparams['TimeLim']
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(testpath.joinpath('stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
def makedata(testpath):
    """ This will make the input data for the test case. The data will have the 
        default set of parameters Ne=Ne=1e11 and Te=Ti=2000.
        Inputs
            testpath - Directory that will hold the data.
            
    """
    finalpath = testpath.joinpath('Origparams')
    if not finalpath.exists():
        finalpath.mkdir()
    data=SIMVALUES
    z = sp.linspace(50.,1e3,50)
    nz = len(z)
    params = sp.tile(data[sp.newaxis,sp.newaxis,:,:],(nz,1,1,1))
    coords = sp.column_stack((sp.ones(nz),sp.ones(nz),z))
    species=['O+','e-']
    times = sp.array([[0,1e3]])
    vel = sp.zeros((nz,1,3))
    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
        
    finalfile = finalpath.joinpath('0 stats.h5')
    Icont1.saveh5(str(finalfile))
    # set start temp to 1000 K.
    Icont1.Param_List[:,:,:,1]=1e3
    Icont1.saveh5(str(testpath.joinpath('startfile.h5')))
    

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
    curloc = Path(__file__).parent
    testpath=curloc.parent.joinpath('Testdata','StatsTest')
    
    testpath.mkdir(exist_ok=True,parents=True)
    
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()
    allfolds = []
#    rsystools = []
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold = testpath.joinpath(foldname)
        allfolds.append(curfold)
        
        curfold.mkdir(exist_ok=True,parents=True)
            
        configfilesetup(curfold,ip)
        makedata(curfold)
        config = curfold/'stats.ini'
        (sensdict,simparams) = readconfigfile(config)
#        rtemp = RadarSys(sensdict,simparams['Rangegatesfinal'],ip)
#        rsystools.append(rtemp.rms(sp.array([1e12]),sp.array([2.5e3]),sp.array([2.5e3])))
        if check_run :
            runsim(functlist_red,curfold,str(curfold.joinpath('stats.ini')),True)
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
    
