#!/usr/bin/env python
"""
Created on Mon Mar 16 12:14:42 2015

@author: John Swoboda
"""
from __future__ import print_function

#imported basic modules
import os, time, sys, getopt, glob
from datetime import datetime
import traceback
import pdb
# Imported scipy and matplotlib modules
import scipy as sp
import tables
# My modules
from RadarDataSim.IonoContainer import IonoContainer
from RadarDataSim.radarData import RadarDataFile
import RadarDataSim.specfunctions as specfuncs
from RadarDataSim.specfunctions import ISRSfitfunction
from RadarDataSim.fitterMethodGen import Fitterionoconainer
from RadarDataSim.utilFunctions import readconfigfile

#%% Make spectrums
def makespectrums(basedir,configfile,remakealldata):

    dirio=('Origparams','Spectrums')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])
    dirlist = glob.glob(os.path.join(inputdir,'*.h5'))
    numlist = [os.path.splitext(os.path.split(x)[-1])[0] for x in dirlist]
    numdict = {numlist[i]:dirlist[i] for i in range(len(dirlist))}
    slist = sorted(numlist,key=ke)

    (sensdict,simparams) = readconfigfile(configfile)

    for inum in slist:

        outfile = os.path.join(outputdir,inum+' spectrum.h5')
        curfile = numdict[inum]
        print('Processing file {} starting at {}\n'.format(os.path.split(curfile)[1]
            ,datetime.now()))
        curiono = IonoContainer.readh5(curfile)
        if curiono.Time_Vector[0]==1e-6:
            curiono.Time_Vector[0] = 0.0
#        curiono.coordreduce(coordlims)
#        curiono.saveh5(os.path.join(inputdir,inum+' red.h5'))
        curiono.makespectruminstanceopen(specfuncs.ISRSspecmake,sensdict,
                                     simparams['numpoints']).saveh5(outfile)
        print('Finished file {} starting at {}\n'.format(os.path.split(curfile)[1],datetime.now()))

#%% Make Radar Data
def makeradardata(basedir,configfile,remakealldata):
    dirio = ('Spectrums','Radardata','ACF')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])
    outputdir2 = os.path.join(basedir,dirio[2])

    dirlist = glob.glob(os.path.join(inputdir,'*.h5'))
    filelist = [os.path.split(item)[1] for item in dirlist]
    timelist = [int(item.partition(' ')[0]) for item in filelist]
    Ionodict = {timelist[it]:dirlist[it] for it in range(len(dirlist))}

    radardatalist = glob.glob(os.path.join(outputdir,'*RawData.h5'))
    if radardatalist and (not remakealldata):
        numlist2 = [os.path.splitext(os.path.split(x)[-1])[0] for x in radardatalist]
        numdict2 = {numlist2[i]:radardatalist[i] for i in range(len(radardatalist))}
        slist2 = sorted(numlist2,key=ke)
        outlist2 = [numdict2[ikey] for ikey in slist2]
    else:
        outlist2 = None

    rdata = RadarDataFile(Ionodict,configfile,outputdir,outfilelist=outlist2)
    ionoout = rdata.processdataiono()
    ionoout.saveh5(os.path.join(outputdir2,'00lags.h5'))
    return ()
#%% Fitt data
def fitdata(basedir,configfile,optintputs):
    dirio = ('ACF','Fitted')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])

    dirlist = glob.glob(os.path.join(inputdir,'*lags.h5'))

    Ionoin=IonoContainer.readh5(dirlist[0])
    fitterone = Fitterionoconainer(Ionoin,configfile)
    (fitteddata,fittederror) = fitterone.fitdata(ISRSfitfunction,startvalfunc,exinputs=[fitterone.simparams['startfile']])
    (Nloc,Ntimes,nparams)=fitteddata.shape
    fittederronly = fittederror[:,:,range(nparams),range(nparams)]
    paramlist = sp.concatenate((fitteddata,fittederronly),axis=2)

    paramnames = []
    species = readconfigfile(configfile)[1]['species']
    for isp in species[:-1]:
        paramnames.append('Ni_'+isp)
        paramnames.append('Ti_'+isp)
    paramnames = paramnames+['Ne','Te','Vi']
    paramnamese = ['n'+ip for ip in paramnames]
    paranamsf = sp.array(paramnames+paramnamese)


    Ionoout=IonoContainer(Ionoin.Sphere_Coords,paramlist,Ionoin.Time_Vector,ver =1,coordvecs = Ionoin.Coord_Vecs, paramnames=paranamsf,species=species)
    outfile = os.path.join(outputdir,'fitteddata.h5')
    Ionoout.saveh5(outfile)

#%% start values for the fit function
def startvalfunc(Ne_init, loc,time,exinputs):
    """ """

    Ionoin = IonoContainer.readh5(exinputs[0])

    numel =sp.prod(Ionoin.Param_List.shape[-2:]) +1

    xarray = sp.zeros((loc.shape[0],len(time),numel))
    for ilocn, iloc in enumerate(loc):
        (datast,vel)=Ionoin.getclosest(iloc,time)[:2]
        datast[:,-1,0] = Ne_init[ilocn,:]
        ionoden =datast[:,:-1,0]
        ionodensum = sp.repeat(sp.sum(ionoden,axis=-1)[:,sp.newaxis],ionoden.shape[-1],axis=-1)
        ionoden = sp.repeat(Ne_init[ilocn,:,sp.newaxis],ionoden.shape[-1],axis=-1)*ionoden/ionodensum
        datast[:,:-1,0] = ionoden
        xarray[ilocn,:,:-1]=sp.reshape(datast,(len(time),numel-1))
        locmag = sp.sqrt(sp.sum(iloc*iloc))
        ilocmat = sp.repeat(iloc[sp.newaxis,:],len(time),axis=0)
        xarray[ilocn,:,-1] = sp.sum(vel*ilocmat)/locmag


    return xarray

#%% For sorting
def ke(item):
    if item[0].isdigit():
        return int(item.partition(' ')[0])
    else:
        return float('inf')
#%% Main function
def main(funcnamelist,basedir,configfile,remakealldata):


    inputsep = '***************************************************************\n'

    funcdict = {'spectrums':makespectrums, 'radardata':makeradardata, 'fitting':fitdata}
    #inout = {'spectrums':('Origparams','Spectrums'),'radardata':('Spectrums','Radardata'),'fitting':('ACF','Fitted')}
    #pdb.set_trace()

    # check for the directories
    dirnames = ['Origparams','Spectrums','Radardata','ACF','Fitted']
    for idir in dirnames:
        curdir = os.path.join(basedir,idir)
        if not os.path.exists(curdir):
            os.makedirs(curdir)

    if len(funcnamelist)==3:
        funcname='all'
    else:
        funcname=''.join(funcnamelist)

    dfilename = 'diary'+funcname+'.txt'
    dfullfilestr = os.path.join(basedir,dfilename)
    f= open(dfullfilestr,'a')


    for curfuncn in funcnamelist:
        curfunc = funcdict[curfuncn]
        f.write(inputsep)
        f.write(curfunc.__name__+'\n')
        f.write(time.asctime()+'\n')
        try:
            stime = datetime.now()
            curfunc(basedir,configfile,curfuncn)
            ftime = datetime.now()
            ptime = ftime-stime
            f.write('Success!\n')
            f.write('Duration: {}\n'.format(ptime))
            f.write('Base directory: {}\n'.format(basedir))

        except Exception, e:
            f.write('Failed!\n')
            ftime = datetime.now()
            ptime = ftime-stime
            f.write('Duration: {}\n'.format(ptime))
            f.write('Base directory: {}\n'.format(basedir))
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc(file = f)
        #pdb.set_trace()
    f.write(inputsep)
    f.close()

    return()
if __name__ == "__main__":

    argv = sys.argv[1:]

    outstr = 'runsim.py -f <function: spectrums, radardata, fitting or all> -i <basedir> -c <config> -r <type y to remake data>'

    try:
        opts, args = getopt.getopt(argv,"hf:i:c:r:")
    except getopt.GetoptError:
        print(outstr)
        sys.exit(2)

    remakealldata = False
    for opt, arg in opts:
        if opt == '-h':
            print(outstr)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            basedir = arg
        elif opt in ("-c", "--cfile"):
            outdirexist = True
            configfile = arg
        elif opt in ("-f", "--func"):
            funcname = arg

        elif opt in ('-r', "--re"):
            if arg.lower() == 'y':
                remakealldata = True
    if funcname.lower() == 'all':
        funcnamelist=['spectrums','radardata','fitting']
    else:
        funcnamelist= funcname.split()

    main(funcnamelist,basedir,configfile,remakealldata)
