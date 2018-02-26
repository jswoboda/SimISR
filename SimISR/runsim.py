#!/usr/bin/env python
"""

runsim.py by John Swoboda 3/16/2015
This script will run the SimISR code. The user can run a number of different
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
import time
import sys
from datetime import datetime
import traceback
import argparse

# Imported scipy and matplotlib modules
import scipy as sp
# My modules
from SimISR import Path
from SimISR.IonoContainer import IonoContainer
from SimISR.radarData import RadarDataFile
import SimISR.specfunctions as specfuncs
from SimISR.fitterMethodGen import Fitterionoconainer
from SimISR.utilFunctions import readconfigfile, update_progress
from SimISR.operators import RadarSpaceTimeOperator

#%% Make spectrums
def makespectrums(basedir, configfile, printlines=True):
    """ This will make all of the spectra for a set of data and save it in a
    folder in basedir called Spectrums. It is assumed that the data in the Origparams
    is time tagged in the title with a string seperated by a white space and the
    rest of the file name. For example ~/DATA/Basedir/0 origdata.h5.
    Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
         """
    basedir = Path(basedir).expanduser()
    dirio = ('Origparams', 'Spectrums')
    inputdir = basedir/dirio[0]
    outputdir = basedir/dirio[1]
    # determine the list of h5 files in Origparams directory
    dirlist = sorted(inputdir.glob('*.h5'))
    # Make the lists of numbers and file names for the dictionary
    (listorder, _, _, timebeg, _) = IonoContainer.gettimes(dirlist)
    slist = [dirlist[ikey] for ikey in listorder]
    (sensdict, simparams) = readconfigfile(configfile)
    # Delete data
    outfiles = outputdir.glob('*.h5')
    for ifile in outfiles:
        ifile.unlink()

    for inum, curfile in zip(timebeg, slist):

        outfile = outputdir / (str(inum)+' spectrum.h5')
        update_progress(float(inum)/float(len(slist)),
                        'Processing file {} starting at {}'.format(curfile.name, datetime.now()))
        curiono = IonoContainer.readh5(str(curfile))

        curiono.makespectruminstanceopen(specfuncs.ISRSspecmake, sensdict,
                                         simparams, float(inum), float(len(slist)),
                                         printlines).saveh5(str(outfile))
        update_progress(float(inum+1)/float(len(slist)),
                        'Finished file {} starting at {}'.format(curfile.name, datetime.now()))

#%% Make Radar Data
def makeradardata(basedir,configfile,remakealldata):
    """ This function will make the radar data and create the acf estimates.
    Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
        remakealldata: A bool that determines if the radar data is remade. If false
            only the acfs will be estimated using the radar that is already made."""

    dirio = ('Spectrums', 'Radardata', 'ACF')
    inputdir = basedir/dirio[0]
    outputdir = basedir/dirio[1]
    outputdir2 = basedir/dirio[2]

    # determine the list of h5 files in Origparams directory
    dirlist = [str(i) for i in inputdir.glob('*.h5')]
    # Make the lists of numbers and file names for the dictionary
    if len(dirlist) > 0:
        (listorder, _, _, timebeg, _) = IonoContainer.gettimes(dirlist)

        Ionodict = {timebeg[itn]:dirlist[it] for itn, it in enumerate(listorder)}
    else:
        Ionodict = {0.:str(inputdir.joinpath('00.h5'))}
    # Find all of the raw data files
    radardatalist = outputdir.glob('*RawData.h5')
    if radardatalist and (not remakealldata):
        # XXX need to work on time stuff
        outlist2 = radardatalist
    else:
        outlist2 = None

    # create the radar data file class
    rdata = RadarDataFile(Ionodict, configfile, outputdir, outfilelist=outlist2)
    # From the ACFs and uncertainties
    (ionoout, ionosig) = rdata.processdataiono()
    # save the acfs and uncertianties in ionocontainer h5 files.
    ionoout.saveh5(str(outputdir2.joinpath('00lags.h5')))
    ionosig.saveh5(str(outputdir2.joinpath('00sigs.h5')))
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
    dirdict = {'fitting':('ACF', 'Fitted'), 'fittingmat':('ACFMat', 'FittedMat'),
               'fittinginv':('ACFInv', 'FittedInv'), 'fittingmatinv':('ACFMatInv', 'FittedMatInv')}
    dirio = dirdict[optinputs[0]]
    inputdir = basedir/dirio[0]
    outputdir = basedir/dirio[1]
    fitlist = optinputs[1]
    if len(optinputs) > 2:
        exstr = optinputs[2]
        printlines = optinputs[3]
    else:
        exstr = ''
    dirlist = [str(i) for i in inputdir.glob('*lags{0}.h5'.format(exstr))]
    dirlistsig = [str(i) for i in inputdir.glob('*sigs{0}.h5'.format(exstr))]

    Ionoin = IonoContainer.readh5(dirlist[0])
    if len(dirlistsig) == 0:
        Ionoinsig = None
    else:
        Ionoinsig = IonoContainer.readh5(dirlistsig[0])
    fitterone = Fitterionoconainer(Ionoin, Ionoinsig, configfile)

    fitoutput = fitterone.fitdata(specfuncs.ISRSfitfunction,
                                  fitterone.simparams['startfile'], fittimes=fitlist,
                                  printlines=printlines)


    (fitteddata, fittederror, funcevals, fittedcov) = fitoutput
    if fitterone.simparams['Pulsetype'].lower() == 'barker':
        paramlist = fitteddata
        species = fitterone.simparams['species']
        paranamsf = ['Ne']
    else:
        fittederronly = sp.sqrt(fittederror)
        paramnames = []
        species = fitterone.simparams['species']
        # Seperate Ti and put it in as an element of the ionocontainer.
        Ti = fitteddata[:, :, 1]

        nTi = fittederronly[:, :, 1]

        nTiTe = fittedcov[:, :, 0, 1]
        nTiNe = fittedcov[:, :, 0, 2]
        nTiVi = fittedcov[:, :, 0, 3]
        nTeNe = fittedcov[:, :, 1, 2]
        nTeVi = fittedcov[:, :, 1, 3]
        nNeVi = fittedcov[:, :, 2, 3]
        cov_list = [nTiTe[:, :, sp.newaxis], nTiNe[:, :, sp.newaxis],
                    nTiVi[:, :, sp.newaxis], nTeNe[:, :, sp.newaxis],
                    nTeVi[:, :, sp.newaxis], nNeVi[:, :, sp.newaxis]]
        cov_list_names = ['nTiTe', 'nTiNe', 'nTiVi', 'nTeNe', 'nTeVi','nNeVi']
        paramlist = sp.concatenate([fitteddata, Ti[:, :, sp.newaxis], fittederronly,
                                    nTi[:, :, sp.newaxis], funcevals[:, :, sp.newaxis]]
                                   + cov_list, axis=2)
        for isp in species[:-1]:
            paramnames.append('Ni_'+isp)
            paramnames.append('Ti_'+isp)
        paramnames = paramnames+['Ne', 'Te', 'Vi', 'Nepow', 'Ti']
        paramnamese = ['n'+ip for ip in paramnames]
        paranamsf = sp.array(paramnames+paramnamese+['FuncEvals']+cov_list_names)

    if fitlist is None:
        timevec = Ionoin.Time_Vector
    else:
        if len(fitlist) == 0:
            timevec = Ionoin.Time_Vector
        else:
            timevec = Ionoin.Time_Vector[fitlist]
    # This requires
    if set(Ionoin.Coord_Vecs) == {'x', 'y', 'z'}:
        newver = 0
        ionoout = IonoContainer(Ionoin.Cart_Coords, paramlist.real, timevec, ver=newver,
                                coordvecs=Ionoin.Coord_Vecs, paramnames=paranamsf,
                                species=species)
    elif set(Ionoin.Coord_Vecs) == {'r', 'theta', 'phi'}:
        newver = 1
        ionoout = IonoContainer(Ionoin.Sphere_Coords, paramlist.real, timevec, ver=newver,
                                coordvecs=Ionoin.Coord_Vecs, paramnames=paranamsf,
                                species=species)


    outfile = outputdir.joinpath('fitteddata{0}.h5'.format(exstr))
    ionoout.saveh5(str(outfile))
#%% apply the matrix for the data
def applymat(basedir, configfile, optinputs):
    """
        This function apply the matrix version of the space time ambiugty function
        to the ACFs and save the outcome in h5 files within the directory ACFMat.
        Inputs:
        basedir: A string for the directory that will hold all of the data for the simulation.
        configfile: The configuration file for the simulation.
    """
    dirio = ('Spectrums', 'Mat', 'ACFMat')
    basedir = Path(basedir)
    inputdir = basedir.joinpath(dirio[0])
    outputdir2 = basedir.joinpath(dirio[2])

    dirlist = [str(i) for i in inputdir.glob('*.h5')]
    (listorder, timevector, _, _, _) = IonoContainer.gettimes(dirlist)
    ionolist = [dirlist[ikey] for ikey in listorder]
    rsto = RadarSpaceTimeOperator(ionolist, configfile, timevector, mattype='matrix')
    ionoout = rsto.mult_iono(ionolist)
    outfile = outputdir2.joinpath('00lags.h5')
    ionoout.saveh5(str(outfile))

#%% For sorting
def ke(item):
    """
        Used for sorting names of files.
    """
    if item[0].isdigit():
        return int(item.partition(' ')[0])
    else:
        return float('inf')
#%% Main function
def main(funcnamelist,basedir,configfile,remakealldata,fitlist=None,invtype='',printlines=True):
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
        fitlist:  A list of time entries that will be fit.

        invtype

    """

    inputsep = '***************************************************************\n'

    funcdict = {'spectrums':makespectrums, 'radardata':makeradardata, 'fitting':fitdata,'fittingmat':fitdata,
                'fittinginv':fitdata,'applymat':applymat,'fittingmatinv':fitdata}
    #inout = {'spectrums':('Origparams','Spectrums'),'radardata':('Spectrums','Radardata'),'fitting':('ACF','Fitted')}
    #pdb.set_trace()

    # check for the directories
    dirnames = ['Origparams','Spectrums','Radardata','ACF','Fitted','ACFOrig','ACFMat','ACFInv','FittedMat','FittedInv','ACFMatInv','FittedMatInv']
    basedir=Path(basedir).expanduser()
    for idir in dirnames:
        curdir = basedir/idir
        curdir.mkdir(exist_ok=True,parents=True)

    if len(funcnamelist)==3:
        funcname='all'
    else:
        funcname=''.join(funcnamelist)

    dfilename = 'diary'+funcname+'.txt'
    dfullfilestr = basedir/dfilename

    with open(str(dfullfilestr),'a') as f:
        failure=False
        for curfuncn in funcnamelist:
            curfunc = funcdict[curfuncn]
            f.write(inputsep)
            f.write(curfunc.__name__+'\n')
            f.write(time.asctime()+'\n')
            if curfunc.__name__=='fitdata':
                ex_inputs=[curfuncn,fitlist,invtype,printlines]
            elif curfunc.__name__=='makeradardata':
                ex_inputs = remakealldata
            else:
                ex_inputs=printlines
            try:
                stime = datetime.now()
                curfunc(basedir,configfile,ex_inputs)
                ftime = datetime.now()
                ptime = ftime-stime
                f.write('Success!\n')
                f.write('Duration: {}\n'.format(ptime))
                f.write('Base directory: {}\n'.format(basedir))

            except Exception as e:
                f.write('Failed!\n')
                ftime = datetime.now()
                ptime = ftime-stime
                f.write('Duration: {}\n'.format(ptime))
                f.write('Base directory: {}\n'.format(basedir))
                traceback.print_exc(file=sys.stdout)
                traceback.print_exc(file=f)
                failure = True
                break

        f.write(inputsep)


    return failure

def parse_command_line(str_input=None):
    """
        This will parse through the command line arguments
    """
    # if str_input is None:
    parser = argparse.ArgumentParser()
    # else:
    #     parser = argparse.ArgumentParser(str_input)
    fstr = '''      These will be the possible strings for the argument and the
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
                    '''
    parser.add_argument('-f', '--funclist', dest='funclist', default='all', help=fstr)

    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="prints debug output and additional detail.")
    parser.add_argument("-c", "--config", dest="config", default='default.ini',
                        help=" Config file used for the simulation, .ini or yaml file.")
    parser.add_argument('-p', "--path", dest='path', default=None,
                        help='Number of incoherent integrations in calculations.')
    parser.add_argument('-r', "--remake", action="store_true", dest='remake', default=False,
                        help='Remake data flag.')

    if str_input is None:
        return parser.parse_args()
    else:
        return parser.parse_args(str_input)

if __name__ == "__main__":

    outstr = '''
             Usage: python runsim.py -f <function: spectrums, radardata, fitting or all> -i <basedir> -c <config> -r <type y to remake data>

             or

             python runsim.py -h

             This script will run the SimISR code. The user
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

    args_commd = parse_command_line()
    if args_commd.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)
    basedir =  str(Path(args_commd.path).expanduser())
    configfile = str(Path(args_commd.config).expanduser())
    funcname = args_commd.funclist
    remakealldata = args_commd.remake

    if funcname.lower() == 'all':
        funcnamelist = ['spectrums', 'radardata', 'fitting']
    else:
        funcnamelist = funcname.split()

    failflag = main(funcnamelist, basedir, configfile, remakealldata)
