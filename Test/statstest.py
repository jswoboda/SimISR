#!/usr/bin/env python
"""
Created on Wed Mar 30 13:01:31 2016
This will create a number of data sets for statistical analysis. It'll then make
statistics and histograms of the output parameters.
@author: John Swoboda
"""
import itertools
import math
from SimISR import Path
import scipy as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from SimISR.utilFunctions import MakePulseDataRep,CenteredLagProduct,readconfigfile,spect2acf,makeconfigfile
from SimISR.IonoContainer import IonoContainer
from  SimISR.runsim import main as runsim
from SimISR.analysisplots import analysisdump,maketi
#from radarsystools.radarsystools import RadarSys
PVALS = [1e11,2.1e3,1.1e3,0.]
SIMVALUES = sp.array([[PVALS[0],PVALS[2]],[PVALS[0],PVALS[1]]])
#
TE = {'param':'Te','paramLT':'T_e','lims':[1200.,3000.],'val':PVALS[1]}
TI = {'param':'Ti','paramLT':'T_i','lims':[300.,1900.],'val':PVALS[2]}
NE = {'param':'Ne','paramLT':'N_e','lims':[4e10,2e11],'val':PVALS[0]}
VI = {'param':'Vi','paramLT':'V_i','lims':[-250.,250.],'val':PVALS[3]}

PARAMDICT = {'Te':TE, 'Ti':TI, 'Ne':NE, 'Vi':VI}
def makehistmult(testpathlist,npulseslist):
    """
        Plot a set of histograms over each other.
    """
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
            curhist, binout = sp.histogram(curvals,bins=histvecs[iax])
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
    """
        Make a histogram from a single collection of data.
    """
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

        mu = PVALS[iax]
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

def make2dhist(testpath, xaxis=TE, yaxis=TI, figmplf=None, curax=None):
    """
        This will plot a 2-D histogram of two variables.

        Args:
            testpath (obj:`str`): The path where the SimISR data is stored.
            npulses (obj:`int`): The number of pulses.
            xaxis (obj: `dict`): default TE, Dictionary that holds the parameter info along the x axis of the distribution.
            yaxis (obj: `dict`): default TE, Dictionary that holds the parameter info along the y axis of the distribution.
            figmplf (obj: `matplotb figure`): default None, Figure that the plot will be placed on.
            curax (obj: `matplotlib axis`): default None, Axis that the plot will be made on.

        Returns:
            figmplf (obj: `matplotb figure`), curax (obj: `matplotlib axis`):,hist_h (obj: `matplotlib axis`)):
            The figure handle the plot is made on, the axis handle the plot is on, the plot handle itself.
    """

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = [xaxis['param'], yaxis['param']]
    datadict, _, _, _ = makehistdata(params, testpath)
    if (figmplf is None) and (curax is None):
        (figmplf, curax) = plt.subplots(1, 1, figsize=(6, 6), facecolor='w')

    b1 = sp.linspace(*xaxis['lims'])
    b2 = sp.linspace(*yaxis['lims'])
    bins = [b1, b2]
    d1 = sp.column_stack((datadict[params[0]],datadict[params[1]]))
    H, xe, ye = sp.histogram2d(d1[:,0].real, d1[:,1].real, bins=bins, normed=True)

    hist_h = curax.pcolor(xe[:-1], ye[:-1], H, cmap='viridis', vmin=0)
    curax.set_xlabel(r'$'+xaxis['paramLT']+'$')
    curax.set_ylabel(r'$'+yaxis['paramLT']+'$')
    curax.set_title(r'Joint distributions for $'+ xaxis['paramLT']+'$'+' and $'+
                    yaxis['paramLT']+'$')
    plt.colorbar(hist_h, ax=curax, label='Probability', format='%1.1e')
    return (figmplf, curax, hist_h)

def makehist(testpath,npulses):
    """
        This functions are will create histogram from data made in the testpath.
        Inputs
            testpath - The path that the data is located.
            npulses - The number of pulses in the sim.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    params = ['Ne', 'Te', 'Ti', 'Vi']
    pvals = [1e11, 2.1e3, 1.1e3, 0.]
    histlims = [[1e10, 3e11], [1000., 3000.], [100., 2500.], [-400., 400.]]
    erlims = [[-2e11, 2e11], [-1000., 1000.], [-800., 800], [-400., 400.]]
    erperlims = [[-100., 100.]]*4
    lims_list = [histlims, erlims, erperlims]
    errdict = makehistdata(params, testpath)[:4]
    ernames = ['Data', 'Error', 'Error Percent']
    sig1 = sp.sqrt(1./npulses)


    # Two dimensiontal histograms
    pcombos = [i for i in itertools.combinations(params, 2)]
    c_rows = int(math.ceil(float(len(pcombos))/2.))
    (figmplf, axmat) = plt.subplots(c_rows, 2, figsize=(12, c_rows*6), facecolor='w')
    axvec = axmat.flatten()
    for icomn, icom in enumerate(pcombos):
        curax = axvec[icomn]
        str1, str2 = icom
        _, _, _ = make2dhist(testpath, PARAMDICT[str1], PARAMDICT[str2], figmplf, curax)
    filetemplate = str(Path(testpath).joinpath('AnalysisPlots', 'TwoDDist'))
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    figmplf.suptitle('Pulses: {0}'.format(npulses), fontsize=20)
    fname = filetemplate+'_{0:0>5}Pulses.png'.format(npulses)
    plt.savefig(fname)
    plt.close(figmplf)
    # One dimensiontal histograms
    for ierr, iername in enumerate(ernames):
        filetemplate= str(Path(testpath).joinpath('AnalysisPlots', iername))
        (figmplf, axmat) = plt.subplots(2, 2, figsize=(20, 15), facecolor='w')
        axvec = axmat.flatten()
        for ipn, iparam in enumerate(params):
            plt.sca(axvec[ipn])
            if sp.any(sp.isinf(errdict[ierr][iparam])):
                continue
            binlims = lims_list[ierr][ipn]
            bins = sp.linspace(binlims[0],binlims[1],100)
            xdata = errdict[ierr][iparam]
            xlog = sp.logical_and(xdata >= binlims[0], xdata < binlims[1])

            histhand = sns.distplot(xdata[xlog], bins=bins, kde=True, rug=False)

            axvec[ipn].set_title(iparam)
        figmplf.suptitle(iername +' Pulses: {0}'.format(npulses), fontsize=20)
        fname = filetemplate+'_{0:0>5}Pulses.png'.format(npulses)
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
    maindir = Path(maindir)
    ffit = maindir.joinpath('Fitted', 'fitteddata.h5')
    inputfiledir = maindir.joinpath('Origparams')

    paramslower = [ip.lower() for ip in params]
    eparamslower = ['n'+ip.lower() for ip in params]

    # set up data dictionary

    errordict = {ip:[] for ip in params}
    errordictrel = {ip:[] for ip in params}
    #Read in fitted data

    Ionofit = IonoContainer.readh5(str(ffit))
    times = Ionofit.Time_Vector

    dataloc = Ionofit.Sphere_Coords
    rng = dataloc[:, 0]
    rng_log = sp.logical_and(rng > 200., rng < 400)
    dataloc_out = dataloc[rng_log]
    pnames = Ionofit.Param_Names
    pnameslower = sp.array([ip.lower() for ip in pnames.flatten()])
    p2fit = [sp.argwhere(ip == pnameslower)[0][0] if ip in pnameslower else None for ip in paramslower]

    datadict = {ip:Ionofit.Param_List[rng_log, :, p2fit[ipn]].flatten() for ipn, ip in enumerate(params)}

    ep2fit = [sp.argwhere(ip==pnameslower)[0][0] if ip in pnameslower else None for ip in eparamslower]

    edatadict = {ip:Ionofit.Param_List[rng_log, :, ep2fit[ipn]].flatten() for ipn, ip in enumerate(params)}
    # Determine which input files are to be used.

    dirlist = [str(i) for i in inputfiledir.glob('*.h5')]
    _, outime, filelisting, _, _ = IonoContainer.gettimes(dirlist)
    time2files = []
    for itn, itime in enumerate(times):
        log1 = (outime[:, 0] >= itime[0]) & (outime[:, 0] < itime[1])
        log2 = (outime[:, 1] > itime[0]) & (outime[:, 1] <= itime[1])
        log3 = (outime[:, 0] <= itime[0]) & (outime[:, 1] > itime[1])
        tempindx = sp.where(log1|log2|log3)[0]
        time2files.append(filelisting[tempindx])


    curfilenum = -1
    for iparam, pname in enumerate(params):
        curparm = paramslower[iparam]
        # Use Ne from input to compare the ne derived from the power.
        if curparm == 'nepow':
            curparm = 'ne'
        datalist = []
        for itn, itime in enumerate(times):
            for  filenum in time2files[itn]:
                filenum = int(filenum)
                if curfilenum != filenum:
                    curfilenum = filenum
                    datafilename = dirlist[filenum]
                    Ionoin = IonoContainer.readh5(datafilename)
                    if ('ti' in paramslower) or ('vi' in paramslower):
                        Ionoin = maketi(Ionoin)
                    pnames = Ionoin.Param_Names
                    pnameslowerin = sp.array([ip.lower() for ip in pnames.flatten()])
                prmloc = sp.argwhere(curparm == pnameslowerin)
                if prmloc.size != 0:
                    curprm = prmloc[0][0]
                # build up parameter vector bs the range values by finding the closest point in space in the input
                curdata = sp.zeros(len(dataloc_out))

                for irngn, curcoord in enumerate(dataloc_out):

                    tempin = Ionoin.getclosestsphere(curcoord, [itime])[0]
                    Ntloc = tempin.shape[0]
                    tempin = sp.reshape(tempin, (Ntloc, len(pnameslowerin)))
                    curdata[irngn] = tempin[0, curprm]
                datalist.append(curdata)
        errordict[pname] = datadict[pname]-sp.hstack(datalist)
        errordictrel[pname] = 100.*errordict[pname]/sp.absolute(sp.hstack(datalist))
    return datadict, errordict, errordictrel, edatadict

def configfilesetup(testpath,npulses):
    """ This will create the configureation file given the number of pulses for
        the test. This will make it so that there will be 12 integration periods
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses.
    """

    curloc = Path(__file__).resolve().parent
    defcon = curloc.joinpath('statsbase.ini')

    (sensdict, simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint'] = ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = ratio1 * simparams['TimeLim']

    simparams['startfile'] = 'startfile.h5'
    makeconfigfile(testpath.joinpath('stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)

def makedata(testpath):
    """
        This will make the input data for the test case. The data will have the
        default set of parameters Ne=Ne=1e11 and Te=Ti=2000.
        Inputs
            testpath - Directory that will hold the data.

    """
    finalpath = testpath.joinpath('Origparams')
    if not finalpath.exists():
        finalpath.mkdir()
    data = SIMVALUES
    z = sp.linspace(50., 1e3, 50)
    nz = len(z)
    params = sp.tile(data[sp.newaxis, sp.newaxis, :, :], (nz, 1, 1, 1))
    coords = sp.column_stack((sp.ones(nz), sp.ones(nz), z))
    species = ['O+', 'e-']
    times = sp.array([[0, 1e9]])
    vel = sp.zeros((nz, 1, 3))
    Icont1 = IonoContainer(coordlist=coords, paramlist=params, times=times,
                           sensor_loc=sp.zeros(3), ver=0, coordvecs=['x', 'y', 'z'],
                           paramnames=None, species=species, velocity=vel)

    finalfile = finalpath.joinpath('0 stats.h5')
    Icont1.saveh5(str(finalfile))
    # set start temp to 1000 K.
    Icont1.Param_List[:, :, :, 1] = 1e3
    Icont1.saveh5(str(testpath.joinpath('startfile.h5')))


def main(plist = None, functlist = ['spectrums','radardata','fitting','analysis','stats'], datadir=None):
    """ This function will call other functions to create the input data, config
        file and run the radar data sim. The path for the simulation will be
        created in the Testdata directory in the SimISR module. The new
        folder will be called BasicTest. The simulation is a long pulse simulation
        will the desired number of pulses from the user.
        Inputs
            npulse - Number of pulses for the integration period, default==100.
            functlist - The list of functions for the SimISR to do.
    """
    if plist is None:
        plist = sp.array([50, 100, 200, 500, 1000, 2000, 5000])
    if isinstance(plist, list):
        plist = sp.array(plist)

    if datadir is None:
        curloc = Path(__file__).resolve().parent
        testpath = curloc.parent.joinpath('Testdata', 'StatsTest')
    else:
        datadir = Path(datadir)
        testpath = datadir

    testpath.mkdir(exist_ok=True, parents=True)

    functlist_default = ['spectrums', 'radardata', 'fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run = sp.any(check_list)
    functlist_red = sp.array(functlist_default)[check_list].tolist()
    allfolds = []
#    rsystools = []
    for ip in plist:
        foldname = 'Pulses_{:04d}'.format(ip)
        curfold = testpath.joinpath(foldname)
        allfolds.append(curfold)

        curfold.mkdir(exist_ok=True, parents=True)

        configfilesetup(curfold, ip)
        makedata(curfold)
        config = curfold/'stats.ini'
#        rtemp = RadarSys(sensdict,simparams['Rangegatesfinal'],ip)
#        rsystools.append(rtemp.rms(sp.array([1e12]),sp.array([2.5e3]),sp.array([2.5e3])))
        if check_run:
            runsim(functlist_red, curfold, str(curfold.joinpath('stats.ini')), True)
        if 'analysis' in functlist:
            analysisdump(curfold, config, params = ['Ne', 'Te', 'Ti', 'Vi'])
        if 'stats' in functlist:
            makehist(curfold, ip)

if __name__== '__main__':
    from argparse import ArgumentParser
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    parser = ArgumentParser(description=descr)

    parser.add_argument("-p", "--npulses", help='Number of pulses.', nargs='+',
                        type=int, default=[50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument('-f','--funclist', help='Functions to be uses', nargs='+',
                        default=['spectrums', 'radardata', 'fitting', 'analysis', 'stats'])#action='append',dest='collection',default=['spectrums','radardata','fitting','analysis'])
    parser.add_argument('dir', help='original directory', default='../Testdata/StatsTest/',nargs='?')
    args = parser.parse_args()

    main(args.npulses, args.funclist, args.dir)
