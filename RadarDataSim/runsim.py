#!/usr/bin/env python
"""

runsim.py by John Swoboda 3/16/2015
This script will run the RadarDataSim code. The user can run a number of different 
aspects including making data, fitting applying matrix formulation of the 
space-time ambuiguty operator and calling inversion methods. 


Inputs 
             -f These will be the possible functions that can be used
                 spectrums :makespectrums, 
                 radardata :makeradardata, 
                 fitting :fitdata,'
                 fittingmat :fitdata,
                 fittinginv :fitdata,
                 applymat applymat
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
from RadarDataSim.specfunctions import ISRSfitfunction, ISRSfitfunction_lmfit
from RadarDataSim.fitterMethodGen import Fitterionoconainer
from RadarDataSim.utilFunctions import readconfigfile
from operators import RadarSpaceTimeOperator
#%% Make spectrums
def makespectrums(basedir,configfile,remakealldata):
    """ This will make all of the spectra for a set of data and save it in a 
    folder in basedir called Spectrums. It is assumed that the data in the Origparams
    is time tagged in the title with a string seperated by a white space and the 
    rest of the file name. For example ~/DATA/Basedir/0 origdata.h5. 
    Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
         """

    dirio=('Origparams','Spectrums')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])
    # determine the list of h5 files in Origparams directory
    dirlist = glob.glob(os.path.join(inputdir,'*.h5'))
    # Make the lists of numbers and file names for the dictionary
    (listorder,timevector,timebeg) = IonoContainer.gettimes(dirlist)
    slist = [dirlist[ikey] for ikey in listorder]
    (sensdict,simparams) = readconfigfile(configfile)
    for inum, curfile in zip(timebeg,slist):

        outfile = os.path.join(outputdir,str(inum)+' spectrum.h5')
        print('Processing file {0} starting at {1}\n'.format(os.path.split(curfile)[1]
            ,datetime.now()))
        curiono = IonoContainer.readh5(curfile)

        curiono.makespectruminstanceopen(specfuncs.ISRSspecmake,sensdict,
                                     simparams['numpoints']).saveh5(outfile)
        print('Finished file {0} starting at {1}\n'.format(os.path.split(curfile)[1],datetime.now()))

#%% Make Radar Data
def makeradardata(basedir,configfile,remakealldata):
    """ This function will make the radar data and create the acf estimates.
    Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
        remakealldata: A bool that determines if the radar data is remade. If false
            only the acfs will be estimated using the radar that is already made."""
            
    dirio = ('Spectrums','Radardata','ACF')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])
    outputdir2 = os.path.join(basedir,dirio[2])

    # determine the list of h5 files in Origparams directory
    dirlist = glob.glob(os.path.join(inputdir,'*.h5'))
    # Make the lists of numbers and file names for the dictionary
    (listorder,timevector,timebeg) = IonoContainer.gettimes(dirlist)

    Ionodict = {timebeg[itn]:dirlist[it] for itn, it in enumerate(listorder)}
    
    # Find all of the raw data files
    radardatalist = glob.glob(os.path.join(outputdir,'*RawData.h5'))
    if radardatalist and (not remakealldata):
        (listorderr,timevectorr,timebegr) = IonoContainer.gettimes(radardatalist)
        outlist2 = [radardatalist[ikey] for ikey in listorderr]
    else:
        outlist2 = None
        
    # create the radar data file class
    rdata = RadarDataFile(Ionodict,configfile,outputdir,outfilelist=outlist2)
    # From the ACFs and uncertainties
    (ionoout,ionosig) = rdata.processdataiono()
    # save the acfs and uncertianties in ionocontainer h5 files.
    ionoout.saveh5(os.path.join(outputdir2,'00lags.h5'))
    ionosig.saveh5(os.path.join(outputdir2,'00sigs.h5'))
    return ()
#%% Fit data
def fitdata(basedir,configfile,optinputs):
    """ This function will run the fitter on the estimated ACFs saved in h5 files.
        Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
        optinputs:A string that helps determine the what type of acfs will be fitted.
         """
    # determine the input folders which can be ACFs from the full simulation
    dirdict = {'fitting':('ACF','Fitted'),'fittingmat':('ACFMat','FittedMat'),'fittinginv':('ACFInv','FittedInv')}
    dirio = dirdict[optinputs]
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])

    dirlist = glob.glob(os.path.join(inputdir,'*lags.h5'))
    dirlistsig = glob.glob(os.path.join(inputdir,'*sigs.h5'))

    Ionoin=IonoContainer.readh5(dirlist[0])
    if len(dirlistsig)==0:
        Ionoinsig=None
    else:
        Ionoinsig=IonoContainer.readh5(dirlistsig[0])
    fitterone = Fitterionoconainer(Ionoin,Ionoinsig,configfile)
    (fitteddata,fittederror) = fitterone.fitdata(ISRSfitfunction_lmfit,startvalfunc,exinputs=[fitterone.simparams['startfile']])


    if fitterone.simparams['Pulsetype'].lower() == 'barker':
        paramlist=fitteddata
        species = fitterone.simparams['species']
        paranamsf=['Ne']
    else:
        (Nloc,Ntimes,nparams)=fitteddata.shape
        fittederronly = sp.sqrt(fittederror[:,:,range(nparams),range(nparams)])

    

        paramnames = []
        species = fitterone.simparams['species']
        Nions = len(species)-1
        Nis = fitteddata[:,:,0:Nions*2:2]
        Tis = fitteddata[:,:,1:Nions*2:2]
        Nisum = sp.nansum(Nis,axis=2)[:,:,sp.newaxis]
        Tisum = sp.nansum(Nis*Tis,axis=2)[:,:,sp.newaxis]
        Ti = Tisum/Nisum

        nNis = fittederronly[:,:,0:Nions*2:2]
        nTis = fittederronly[:,:,1:Nions*2:2]
        nNisum = sp.nansum(Nis*nNis**2,axis=2)[:,:,sp.newaxis]
        nNi = sp.sqrt(nNisum/Nisum)
        nTisum = sp.nansum(Nis*nTis**2,axis=2)[:,:,sp.newaxis]
        nTi = sp.sqrt(nTisum/Nisum)
        paramlist = sp.concatenate((fitteddata,Nisum,Ti,fittederronly,nNi,nTi),axis=2)
        for isp in species[:-1]:
            paramnames.append('Ni_'+isp)
            paramnames.append('Ti_'+isp)
        paramnames = paramnames+['Ne','Te','Vi','Nepow','Ni','Ti']
        paramnamese = ['n'+ip for ip in paramnames]
        paranamsf = sp.array(paramnames+paramnamese)


    Ionoout=IonoContainer(Ionoin.Sphere_Coords,paramlist,Ionoin.Time_Vector,ver =1,coordvecs = Ionoin.Coord_Vecs, paramnames=paranamsf,species=species)

    outfile = os.path.join(outputdir,'fitteddata.h5')
    Ionoout.saveh5(outfile)
#%% apply the matrix for the data
def applymat(basedir,configfile,optinputs):
    """ This function apply the matrix version of the space time ambiugty function
        to the ACFs and save the outcome in h5 files within the directory ACFMat.
        Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
         """
    dirio = ('Spectrums','Mat','ACFMat')
    inputdir = os.path.join(basedir,dirio[0])
    outputdir = os.path.join(basedir,dirio[1])
    outputdir2 = os.path.join(basedir,dirio[2])
    
    dirlist = glob.glob(os.path.join(inputdir,'*.h5'))
    (listorder,timevector,timebeg) = IonoContainer.gettimes(dirlist)
    Ionolist = [dirlist[ikey] for ikey in listorder]
    RSTO = RadarSpaceTimeOperator(Ionolist,configfile,timevector)
    Ionoout = RSTO.mult_iono(Ionolist)
    outfile=os.path.join(outputdir2,'00lags.h5')
    Ionoout.saveh5(outfile)
#%% start values for the fit function
def startvalfunc(Ne_init, loc,time,exinputs):
    """ This is a method to determine the start values for the fitter.
    Inputs 
        Ne_init - A nloc x nt numpy array of the initial estimate of electron density. Basically
        the zeroth lag of the ACF.
        loc - A nloc x 3 numpy array of cartisian coordinates.
        time - A nt x 2 numpy array of times in seconds
        exinputs - A list of extra inputs allowed for by the fitter class. It only
            has one element and its the name of the ionocontainer file holding the 
            rest of the start parameters.
    Outputs
        xarrya - This is a numpy arrya of starting values for the fitter parmaeters."""

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
    """ Main function for this module. The function will set up the directory 
    structure, create or update a diary file and run the simulation depending
    on the user input. 
    Inputs 
        funcnamelist: A list of strings that coorespond to specific functions. 
                The stirng and function they correspond to, that will be shown.
                 
                 spectrums: makespectrums, This will create the ISR spectrums
                 
                 radardata :makeradardata, This will make the radar data and
                     form the ACF estimates. If the raw radar data exists then 
                     the user must use the -r option on the command line and set 
                     it to y.
                     
                 fitting :fitdata, This will apply the fitter to the data in 
                 the ACF folder of the base directory.
                 
                 fittingmat :fitdata,This will apply the fitter to the data in 
                 the ACFMat folder of the base directory.
                 
                 fittinginv :fitdata,This will apply the fitter to the data in 
                 the ACFInv folder of the base directory.
                 
                 applymat :applymat, This wil create and apply a matrix  
                 formulation of thespace-time ambiguity function to ISR spectrums.
        basedir: The base directory that will contain all of the data. This directory 
                must contain a directory called Origparams with h5 files to run 
                the full simulation. The user can also start with a directory 
                from a later stage of the simulation instead though.
        
        configfile: The configuration used for the simulation. Can be an ini file or
                a pickle file.
                
        remakealldata: A bool to determine if the raw radar data will be remade. If
                this is False the radar data will only be made if it does
                not exist in the file first.
        
    """

    inputsep = '***************************************************************\n'

    funcdict = {'spectrums':makespectrums, 'radardata':makeradardata, 'fitting':fitdata,'fittingmat':fitdata,
                'fittinginv':fitdata,'applymat':applymat}
    #inout = {'spectrums':('Origparams','Spectrums'),'radardata':('Spectrums','Radardata'),'fitting':('ACF','Fitted')}
    #pdb.set_trace()

    # check for the directories
    dirnames = ['Origparams','Spectrums','Radardata','ACF','Fitted','ACFOrig','ACFMat','ACFInv','FittedMat','FittedInv']
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

    failure=False
    for curfuncn in funcnamelist:
        curfunc = funcdict[curfuncn]
        f.write(inputsep)
        f.write(curfunc.__name__+'\n')
        f.write(time.asctime()+'\n')
        if curfunc.__name__=='fitdata':
            ex_inputs=curfuncn
        else:
            ex_inputs = remakealldata
        try:
            stime = datetime.now()
            curfunc(basedir,configfile,ex_inputs)
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
            failure=True
            break
        #pdb.set_trace()
    f.write(inputsep)
    f.close()

    return failure
if __name__ == "__main__":

    argv = sys.argv[1:]

    outstr = ''' 
             Usage: python runsim.py -f <function: spectrums, radardata, fitting or all> -i <basedir> -c <config> -r <type y to remake data>

             or 
             
             python runsim.py -h
             
             This script will run the RadarDataSim code. The user 
             can run a number of different aspects including making data, fitting
             applying matrix formulation sof the space-time operator and calling 
             inversion methods. 
             
                          
             Manditory Arguments to run code, for help just use the -h option. 
             
             -f These will be the possible strings for the argument and the  
                function they correspond to, that will be used ish shown.
                 
                 spectrums: makespectrums, This will create the ISR spectrums
                 
                 radardata :makeradardata, This will make the radar data and
                     form the ACF estimates. If the raw radar data exists then 
                     the user must use the -r option on the command line and set 
                     it to y.
                     
                 fitting :fitdata, This will apply the fitter to the data in 
                 the ACF folder of the base directory.
                 
                 fittingmat :fitdata,This will apply the fitter to the data in 
                 the ACFMat folder of the base directory.
                 
                 fittinginv :fitdata,This will apply the fitter to the data in 
                 the ACFInv folder of the base directory.
                 
                 applymat :applymat, This wil create and apply a matrix  
                 formulation of thespace-time ambiguity function to ISR spectrums.
                 
                 
                 all - This will run the commands from using the spectrums, radardata, 
                     and fitting
                     
            -i The base directory that will contain all of the data. This directory 
                must contain a directory called Origparams with h5 files to run 
                the full simulation. The user can also start with a directory 
                from a later stage of the simulation instead though.
                
            -c The configuration used for the simulation. Can be an ini file or
                a pickle file.
                
            Optional arguments
            
            -r If a y follows this then the raw radar data will be remade. If
                this is not used the radar data will only be made if it does
                not exist in the file first.
            
             Example:
             python runsim.py -f radardata -f fitting -i ~/DATA/ExampleLongPulse -c ~/DATA/Example -r y'''

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
            basedir = os.path.expanduser(arg)
        elif opt in ("-c", "--cfile"):
            outdirexist = True
            configfile = os.path.expanduser(arg)
        elif opt in ("-f", "--func"):
            funcname = arg

        elif opt in ('-r', "--re"):
            if arg.lower() == 'y':
                remakealldata = True
    if funcname.lower() == 'all':
        funcnamelist=['spectrums','radardata','fitting']
    else:
        funcnamelist= funcname.split()

    failflag = main(funcnamelist,basedir,configfile,remakealldata)
