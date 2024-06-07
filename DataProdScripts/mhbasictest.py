#!python

"""
"""
import sys
import argparse
import shutil
from pathlib import Path
import numpy as np

from SimISR import make_test_ex, readconfigfile, makeconfigfile, runsimisr



def parse_command_line(str_input=None):
    """This will parse through the command line arguments

    Function to go through the command line and if given a list of strings all
    also output a namespace object.

    Parameters
    ----------
    str_input : list
        A list of strings or the input from the command line.

    Returns
    -------
    input_args : Namespace
        An object holding the input arguments wrt the variables.
    """
    scriptpath = Path(sys.argv[0])
    scriptname = scriptpath.name

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width
    title = "Run VS Ionosonde analysis"
    shortdesc = "Runs the basic processing and detection of the ionospheric echos for the vs ionosonde."
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )
    # desc = "This is the run script for SimVSR."
    # if str_input is None:
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # else:
    #     parser = argparse.ArgumentParser(str_input)

    parser.add_argument(
        "-d",
        "--datadir",
        dest="datadir",
        help="Original directory the data was recorded to.",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="Config file.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-n",
        "--nminutes",
        dest="nminutes",
        help="Number of minutes",
        required=True,
        type=int,
    )
 
    if str_input is None:
        return parser.parse_args()
    return parser.parse_args(str_input)


def configfilesetup(testpath, config, simtime_mins=4):
    """ This will create the configureation file given the number of pulses for
        the test. This will make it so that there will be 12 integration periods
        for a given number of pulses.

    Parameters
    ----------
    testpath : str 
        The location of the data.
    config : str
        Default config file.
    """
    curloc = Path(__file__).resolve().parent
    defcon = curloc/config
    (sensdict, simparams) = readconfigfile(str(defcon))
    simparams['TimeLim'] = simtime_mins*60

    simparams['fitmode'] = 1
    simparams['startfile'] = 'startfile.h5'
    makeconfigfile(str(testpath/config), simparams['Beamlist'],
                   sensdict['Name'], simparams)


def run_example(datadir="~/DATA/SimISR/MHchapmantest",config="MHsimple.yml",nminutes=4):

    curloc = Path(__file__).resolve().parent
    testpath = Path(datadir)
    if not testpath.is_dir():
        testpath.mkdir(parents=True)

    z = np.arange(90, 700,5,dtype = float)
    coords = np.column_stack([np.zeros_like(z),np.zeros_like(z),z])
    iono_t = make_test_ex(testv=False,testtemp=True,coords=coords)
    inputpath = testpath.joinpath('Origparams')

    if not inputpath.is_dir():
        inputpath.mkdir()
    inputfile = inputpath.joinpath('0 stats.h5')

    iono_t.saveh5(str(inputfile))
    iono_t.saveh5(str(testpath.joinpath('startfile.h5')))

    functlist = ['spectrums', 'radardata']
    configfilesetup(testpath, config, nminutes)
    config = str(testpath.joinpath(config))

    drfdirone = drfdata = testpath/'drfdata'
    if drfdirone.exists() and 'radardata' in functlist:
        shutil.rmtree(str(drfdirone))
    if not drfdirone.exists():
        drfdata = testpath/'drfdata'/'rf_data'/'zenith-l'
        drfdata.mkdir(parents=True, exist_ok=True)
        drfdatatx = testpath/'drfdata'/'rf_data'/'tx-h'
        drfdatatx.mkdir(parents=True, exist_ok=True)
        dmddir = testpath/'drfdata'/'metadata'
        dmddir.mkdir(parents=True, exist_ok=True)

        acmdata = dmddir.joinpath('antenna_control_metadata')
        acmdata.mkdir(parents=True, exist_ok=True)

        iddir = dmddir.joinpath('id_metadata')
        iddir.mkdir(parents=True, exist_ok=True)

        pmdir = dmddir.joinpath('powermeter')
        pmdir.mkdir(parents=True, exist_ok=True)
    runsimisr(functlist, str(testpath), config, True)


if __name__ == '__main__':
    
    args_commd = parse_command_line()
    arg_dict = {k: v for k, v in args_commd._get_kwargs() if v is not None}
    run_example(**arg_dict)