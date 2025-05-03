import numpy as np
from mathutils import jinc,rotcoords
import scipy.constants as sconst
d2r = np.pi/180.

class AntPatPlug(object):
    """

    Attributes
    ----------
    freq : float
        Frequency of the radiation in Hz. Default = 440e6
    r : float
        Radius of the antenna in meters. Default = 23
    """
    def __init__(self,freq,rad):
        """

        Parmeters
        ---------
        freq : float
            Frequency of the radiation in Hz. Default = 440e6
        r : float
            Radius of the antenna in meters. Default = 23
        """
        self.freq = freq
        self.rad = rad

    def calc_pattern(self,Az,El,Az0,El0):
        """

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
            The normalized antenna pattern.
        """
        Azadj, Eladj = rotcoords(Az, El, -Az0, El0 - 90.0)
        Elr = (90.0 - Eladj) * d2r
        Circ_Ant_Pattern(Elr, self.freq, self.rad)

def Circ_Ant_Pattern(EL, freq=440e6, r=23.):
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
