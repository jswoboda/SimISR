import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from mathutils import diric,rotcoords
import scipy.constants as sconst
d2r = np.pi/180.


class AntPatPlug(object):
    """Antenna pluggin for AMISR arrays.
    Attributes
    ----------

    freq : float
        Frequency of emission in Hz. Default is 440e6 Hz
    az_rotation :
        Rotation off of North in degrees.
    el_tilt : float
        Rotation off of parallel to the ground.
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

    """
    def __init__(self,freq = 440e6,az_rotation=0,el_tilt=0, dx=0.4343, dy=0.4958, mpp=8, mpan=8, npp=4, npan=16):

        self.az_rotation = az_rotation
        self.el_tilt = el_tilt
        self.ant_params = dict(freq=freq,dx = dx, dy = dy,mpp = mpp,mpan = mpan,npp = npp,npan = npan)
    def calc_pattern(self,Az, El, Az0, El0):
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

        Returns
        -------
        Patout : ndarray
            The normalized radiation density evaluated over the AZ and and EL vectors.
        """

        d2r = np.pi / 180.0

        Azs, Els = rotcoords(Az, El, -self.az_rotation, -self.el_tilt)
        eps = np.finfo(Az.dtype).eps
        Azs[np.abs(Azs) < 15 * eps] = 0.0
        Azs = np.mod(Azs, 360.0)

        Az0s, El0s = rotcoords(Az0, El0, -self.az_rotation, -self.el_tilt)
        Elr = (90.0 - Els) * d2r
        El0r = (90.0 - El0s) * d2r
        Azr = Azs * d2r
        Az0r = Az0s * d2r
        Patout = AMISR_Pattern(Azr, Elr, Az0r, El0r,**self.ant_params)
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
