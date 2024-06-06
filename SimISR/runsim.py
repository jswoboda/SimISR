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
#imported basic modules
import time
import sys
from datetime import datetime
import traceback
import argparse
import ipdb

# Imported scipy and matplotlib modules
import scipy as sp
# My modules
from pathlib import Path
from SimISR.IonoContainer import IonoContainer
from SimISR.radarData import RadarDataFile
import SimISR.specfunctions as specfuncs
from SimISR.utilFunctions import readconfigfile, update_progress

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
    """
        This function will make the radar data and create the acf estimates.
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
    rdata = RadarDataFile(configfile, outputdir)
    rdata.makerfdata(Ionodict)
    # From the ACFs and uncertainties
    #
    # outdict = rdata.processdataiono()
    # # save the acfs and uncertianties in ionocontainer h5 files.
    # for i_ptype in outdict:
    #     ionoout, ionosig = outdict[i_ptype]
    #
    #     ionoout.saveh5(str(outputdir2.joinpath('00'+i_ptype+'lags.h5')))
    #     ionosig.saveh5(str(outputdir2.joinpath('00'+i_ptype+'sigs.h5')))
    return ()
def processdata(basedir, configfile, optinputs):
    """
        Creates the ACFs from the digital_rf data
    """
    dirio = ('Radardata', 'ACF')
    outputdir = basedir/dirio[0]
    outputdir2 = basedir/dirio[1]
    # create the radar data file class
    rdata = RadarDataFile(configfile, outputdir)
    # From the ACFs and uncertainties

    outdict = rdata.processdataiono()
    # save the acfs and uncertianties in ionocontainer h5 files.
    for i_ptype in outdict:
        ionoout, ionosig = outdict[i_ptype]

        ionoout.saveh5(str(outputdir2.joinpath('00'+i_ptype+'lags.h5')))
        ionosig.saveh5(str(outputdir2.joinpath('00'+i_ptype+'sigs.h5')))
    return ()

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
def runsimisr(funcnamelist,basedir,configfile,remakealldata,printlines=True):
    """ Main function for this module. 
    
    The function will set up the directory
    structure, create or update a diary file and run the simulation depending
    on the user input.
   
    Parameters
    ----------
    funcnamelist : list
        A list of strings that coorespond to specific functions
            spectrums: makespectrums, This will create the ISR spectrums

            radardata :makeradardata, This will make the radar data and
                form the ACF estimates. If the raw radar data exists then
                the user must use the -r option on the command line and set
                it to y.

        basedir: The base directory that will contain all of the data. This directory
                must contain a directory called Origparams with h5 files to run
                the full simulation. The user can also start with a directory
                from a later stage of the simulation instead though.

        configfile: The configuration yaml file used for the simulation. 

        remakealldata: A bool to determine if the raw radar data will be remade. If
                this is False the radar data will only be made if it does
                not exist in the file first.


    """

    inputsep = '***************************************************************\n'

    funcdict = {'spectrums':makespectrums, 'radardata':makeradardata, 'process':processdata}
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
            if curfunc.__name__=='makeradardata':
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
