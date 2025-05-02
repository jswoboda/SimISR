from pickle import INT

#!/usr/bin/env python
"""
:platform: Unix, Windows, Mac
:synopsis: Gets a ISR sensor constants and calculates theoretical beam patterns.

from webbrowser import Elinks
.. moduleauthor:: John Swoboda <swoboj@bu.edu>
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
import scipy.constants as sconst

#
# from test.test_buffer import ndarray_print
from .mathutils import diric, angles2xy, jinc, rotcoords
from .h5fileIO import load_dict_from_hdf5

## Parameters for Sensor
# AMISR = {'Name':'AMISR','Pt':2e6,'k':9.4,'G':10**4.3,'lamb':0.6677,'fc':449e6,'fs':50e3,\
#    'taurg':14,'Tsys':120,'BeamWidth':(2,2)}
# AMISR['t_s'] = 1/AMISR['fs']


def getConst(typestr, angles=None):
    """Get the constants associated with a specific radar system. This will fill
    out a dictionary with all of the parameters.

    Args:
        type (str): Name of the radar system.
        angles (:obj:`numpy array`): Nx2 array where each row is an az, el pair in degrees.

    Returns:
        sensdict (dict[str, obj]): Holds the different sensor constants.::

            {
                    'Name': radar name,
                    'Pt': Transmit power in W,
                    'k': wave number in rad/m,
                    'lamb': Wave length in m,
                    'fc': Carrier Frequency in Hz,
                    'fs': Sampling frequency in Hz,
                    'taurg': Pulse length number of samples,
                    'Tsys': System Temperature in K,
                    'BeamWidth': Tuple of beamwidths in degrees,
                    'Ksys': ,
                    'BandWidth': Filter bandwidth in Hz,
                    'Angleoffset': Tuple of angle offset,
                    'ArrayFunc': Function to calculate antenna pattern,
                    't_s': Sampling time in s
            }
    """
    dirname = Path(__file__).expanduser().parent.parent
    dirname = dirname.joinpath("sensorinfo")
    if typestr.lower() == "risr" or typestr.lower() == "risr-n":
        arrayfunc = AMISR_Patternadj
        h5filename = dirname / "RISR_PARAMS.h5"
    elif typestr.lower() == "pfisr":
        arrayfunc = AMISR_Patternadj
        h5filename = dirname / "PFISR_PARAMS.h5"
    elif typestr.lower() == "millstone":
        arrayfunc = Millstone_Pattern_M
        h5filename = dirname / "Millstone_PARAMS.h5"
    elif typestr.lower() == "millstonez":
        arrayfunc = Millstone_Pattern_Z
        h5filename = dirname / "Millstonez_PARAMS.h5"
    elif typestr.lower() == "sondrestrom":
        arrayfunc = Sond_Pattern
        h5filename = dirname / "Sondrestrom_PARAMS.h5"

    am = load_dict_from_hdf5(str(h5filename))
    kmat = am["Params"]["Kmat"]
    freq = am["Params"]["Frequency"]
    P_r = am["Params"]["Power"]
    bandwidth = am["Params"]["Bandwidth"]
    ts = am["Params"]["Sampletime"]
    systemp = am["Params"]["Systemp"]
    Ang_off = am["Params"]["Angleoffset"]

    Ksens = freq * 2 * np.pi / sconst.c
    lamb = Ksens / 2.0 / np.pi
    az = kmat[:, 1]
    el = kmat[:, 2]
    ksys = kmat[:, 3]

    (xin, yin) = angles2xy(az, el)
    points = np.column_stack((xin, yin))
    if angles is not None:
        (xvec, yvec) = angles2xy(angles[:, 0], angles[:, 1])
        ksysout = griddata(points, ksys, (xvec, yvec), method="nearest")
    else:
        ksysout = None

    #'G':10**4.3, This doesn't get used anymore it seems
    sensdict = {
        "Name": typestr,
        "Pt": P_r,
        "k": Ksens,
        "lamb": lamb,
        "fc": freq,
        "fs": 1 / ts,
        "taurg": 14,
        "Tsys": systemp,
        "BeamWidth": (2, 2),
        "Ksys": ksysout,
        "BandWidth": bandwidth,
        "Angleoffset": Ang_off,
        "ArrayFunc": arrayfunc,
    }
    sensdict["t_s"] = ts
    return sensdict


def getangles(bcodes, radar="risr"):
    """getangles: This function creates take a set of beam codes and determines
    the angles that are associated with them.
    Inputs
    bcodes - A list of beam codes.
    radar - A string that holds the radar name.
    Outputs
    angles - A list of tuples of the angles.
    """
    if radar.lower() == "risr" or radar.lower() == "risr-n":
        reffile = get_files("RISR_PARAMS.h5")
    elif radar.lower() == "pfisr":
        reffile = get_files("PFISR_PARAMS.h5")
    elif radar.lower() == "millstone" or radar.lower() == "millstonez":
        reffile = get_files("Millstone_PARAMS.h5")
    elif radar.lower() == "sondrestrom":
        reffile = get_files("Sondrestrom_PARAMS.h5")

    am = load_dict_from_hdf5(str(reffile))
    kmat = am["Params"]["Kmat"]
    # make a beamcode to angle dictionary
    bco_dict = dict()
    for slin in kmat:
        bco_num = slin[0].astype(int)
        bco_dict[bco_num] = (float(slin[1]), float(slin[2]))

    # Read in file
    # file_name = 'SelectedBeamCodes.txt'
    if type(bcodes) is str:
        file_name = bcodes
        f = open(file_name)
        bcolines = f.readlines()
        f.close()

        bcolist = [int(float(x.rstrip())) for x in bcolines]
    else:
        bcolist = bcodes
    angles = [bco_dict[x] for x in bcolist]
    return angles


def Sond_Pattern(Az, El, Az0, El0, Angleoffset):
    """Gives the ideal antenna pattern for the Sondestrom radar.

    This function will call circular antenna beam patern function after it
    rotates the coordinates given the pointing direction.

    Args:
        Az (:obj:`numpy array`): Azimuth angles in degrees.
        El (:obj:`numpy array`): Elevation angles in degrees.
        Az_0 (float): The azimuth pointing angle in degrees.
        El_0 (float): The elevation pointing angle in degrees.
        Angleoffset (list): A 2 element list holding the offset of the face of the array
            from north.

    Returns:
        Beam_Pattern (:obj:`numpy array`): The relative beam pattern from the azimuth points.
    """

    d2r = np.pi / 180.0
    radius = 30.0
    lamb = sconst.c / 1.2e9

    __, Eladj = rotcoords(Az, El, -Az0, El0 - 90.0)
    Elr = (90.0 - Eladj) * d2r
    return Circ_Ant_Pattern(Elr, 1.2e9, radius)


def Millstone_Pattern_Z(Az, El, Az0, El0, Angleoffset):
    """Gives the ideal antenna pattern for the Zenith dish at Milstone hill.

    This function will call circular antenna beam patern function after it
    rotates the coordinates given the pointing direction.


    Args:
        Az (:obj:`numpy array`): Azimuth angles in degrees.
        El (:obj:`numpy array`): Elevation angles in degrees.
        Az_0 (float): The azimuth pointing angle in degrees.
        El_0 (float): The elevation pointing angle in degrees.
        Angleoffset (list): A 2 element list holding the offset of the face of the array
            from north.

    Returns:
        Beam_Pattern (:obj:`numpy array`): The relative beam pattern from the azimuth points.
    """
    d2r = np.pi / 180.0
    radius = 33.5
    lamb = sconst.c / 4.4e8
    __, Eladj = rotcoords(Az, El, 0.0, 0.0)
    Elr = (90.0 - Eladj) * d2r
    return Circ_Ant_Pattern(Elr, 4.4e8, radius)


def Millstone_Pattern_M(Az, El, Az0, El0, Angleoffset):
    """Gives the ideal antenna pattern for the MISA dish at Milstone hill.

    This function will call circular antenna beam patern function after it
    rotates the coordinates given the pointing direction.


    Args:
        Az (:obj:`numpy array`): Azimuth angles in degrees.
        El (:obj:`numpy array`): Elevation angles in degrees.
        Az_0 (float): The azimuth pointing angle in degrees.
        El_0 (float): The elevation pointing angle in degrees.
        Angleoffset (list): A 2 element list holding the offset of the face of the array
            from north.

    Returns:
        Beam_Pattern (:obj:`numpy array`): The relative beam pattern from the azimuth points.
    """
    d2r = np.pi / 180.0
    r = 23.0
    Azadj, Eladj = rotcoords(Az, El, -Az0, El0 - 90.0)
    Elr = (90.0 - Eladj) * d2r
    return Circ_Ant_Pattern(Elr, 4.4e8, r)


def Circ_Ant_Pattern(EL, freq=440e6, r=23):
    """This function calculates an idealized antenna pattern for a circular antenna array.The pattern is normalized to broad side =1.

    Parameters
    ----------
    EL : ndarray
        The elevation coordinates in radians. Vertical is at zero radians.
    freq : float
        Frequency of the radiation in Hz. Default = 440e6
    r : float
        Radius of the antenna in meters. Default = 23

    Returns
    -------
    Patout : ndarray
        The normalized antenna pattern.
    """

    lamb = sconst.c / freq
    d = r * 2
    t = (d / lamb) * np.sin(EL)
    Patout = (2) ** 2 * np.abs(jinc(t)) ** 2
    zel = np.logical_or(EL <= -np.pi / 2.0, EL > np.pi / 2.0)
    Patout[zel] = 0.0

    normfactor = 2**2 * jinc(0.0) ** 2
    return Patout / normfactor


def get_files(fname):
    """Gets the hdf5 files associated with the radar.

    Args:
        fname (str): Name for the radar.

    Returns:
        newpath (str): String holding the location for the file.
    """
    curpath = Path(__file__).parent.parent
    curpath = curpath.joinpath("sensorinfo")
    newpath = curpath.joinpath(fname)
    if not newpath.is_file():
        return False
    return str(newpath)


def AMISR_Patternadj(Az, El, Az0, El0, Angleoffset):
    """This function will call AMISR beam patern function after it rotates the coordinates given the offset of the phased array.

    Parameters
    ----------
    Az : ndarray
        Azimuth angles in degrees that the pattern will be evaluated over.
    El : ndarray
        Elevation angles in degrees that the pattern will be evaluated over.
    Az_0 : float
        The azimuth pointing angle in degrees.
    El_0 : float
        The elevation pointing angle in degrees.
    Angleoffset : list
        A 2 element list holding the offset of the face of the array from north.

    Returns
    -------
    Patout : ndarray
        The normalized radiation density evaluated over the AZ and and EL vectors.
    """
    d2r = np.pi / 180.0

    Azs, Els = rotcoords(Az, El, -Angleoffset[0], -Angleoffset[1])
    eps = np.finfo(Az.dtype).eps
    Azs[np.abs(Azs) < 15 * eps] = 0.0
    Azs = np.mod(Azs, 360.0)

    Az0s, El0s = rotcoords(Az0, El0, -Angleoffset[0], -Angleoffset[1])
    Elr = (90.0 - Els) * d2r
    El0r = (90.0 - El0s) * d2r
    Azr = Azs * d2r
    Az0r = Az0s * d2r
    Patout = AMISR_Pattern(Azr, Elr, Az0r, El0r)
    return Patout


def AMISR_Pattern(
    AZ, EL, Az0, El0, f0=440e6, dx=0.4343, dy=0.4958, mpp=8, mpan=8, npp=4, npan=16
):
    """Returns the AMISR pattern in the direction of the array face. Broadside is 0 degrees azimuth, 0 degrees elevation.

    This function will calculated an idealized antenna pattern for the AMISR array. The pattern is not normalized. The antenna is assumed to made of a grid of ideal cross dipole elements. In the array every other column is shifted by 1/2 dy. The parameters are taken from the AMISR spec and the method for calculating the field is derived from a report by Adam R. Wichman. The inputs for the az and el coordinates can be either an array or scalar. Both AZ and EL arrays must be the same shape.

    Parameters
    ----------:
    AZ : ndarray
        Azimuth angles in radians that the pattern will be evaluated over.
    EL : ndarray
        Elevation angles in radians that the pattern will be evaluated over.
    Az_0 : float
        The azimuth pointing angle in radians.
    El_0 : float
        The elevation pointing angle in radians.
    f0 : float
        Frequency of emission in Hz. Default is 440e6 Hz
    dx : float
        x spacing for elemets in meters, default = 0.4343
    dy : float
        y spacing for elemets in meters, default = 0.4958
    mpp : int
        Elements per panel in the x direction, default = 8
    mpan : int
        Number of panels in the x direction, default = 8
    npp : int
        Elements per panel in the y direction, default = 4
    npan : int
        Number of panels in the x direction, default = 16

    Returns
    -------
    Patout : ndarray
        The normalized radiation density evaluated over the AZ and and EL vectors.
    """
    # frequency of AMISR in Hz
    lam0 = sconst.c / f0  # wavelength in m
    k0 = 2 * np.pi / lam0  # wavenumber in rad/m

    # element pattern from an ideal cross dipole array.
    elementpower = (1.0 / 2.0) * (1.0 + (np.cos(EL) ** 2))
    # Use this to kill back lobes.
    zel = np.logical_or(EL <= -np.pi / 2.0, EL > np.pi / 2.0)
    elementpower[zel] = 0.0
    # m = 8.0  # number of pannels in the x direction
    # mtot = 8.0 * m  # number of elements times panels in x direction
    mtot = mpp * mpan
    ntot = npp * npan
    # n = 16.0  # number of pannels in the y direction
    # ntot = n * 4.0  # number of elements times panels in y direction
    # relative phase between the x elements
    phix = k0 * dx * (np.sin(EL) * np.cos(AZ) - np.sin(El0) * np.cos(Az0))
    # relative phase between the y elements
    phiy = k0 * dy * (np.sin(EL) * np.sin(AZ) - np.sin(El0) * np.sin(Az0))

    AF = (
        (1.0 + np.exp(1j * (phiy / 2.0 + phix)))
        * diric(2.0 * phix, mtot / 2.0)
        * diric(phiy, ntot)
    )
    # Normalize by the max array factor
    arrayfac = (abs(AF) ** 2) / 4.0
    Patout = elementpower * arrayfac
    return Patout
