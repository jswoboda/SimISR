import numpy as np
import scipy.special


def jinc(t):
    """This will output a jinc function. Defined here as J_1(pi*t)/(pi*t)

    Parameters
    ----------
    t : ndarray
        Independent varianble input

    Returns
    -------
    outdata : ndarray
        Jinc(t)
    """
    t = np.asanyarray(t)

    outdata = np.ones_like(t)
    outdata[t == 0.0] = 0.5
    outdata[t != 0.0] = scipy.special.jn(1, np.pi * t[t != 0.0]) / (np.pi * t[t != 0.0])

    return outdata

def diric(x, n):
    """Based on octave-signal v. 1.10 diric. Calcaulates the Dirichlet function, periodic aka periodic sinc.

    :Authors: Michael Hirsch

    Parameters
    ----------
    x : ndarray
        Independent varianble input
    n : int
        Function degree

    Returns
    -------
    y : ndarray
        Function output
    """
    n = int(n)
    if n < 1:
        raise ValueError("n is a strictly positive integer")

    x = np.asanyarray(x)

    y = np.sin(n * x / 2) / (n * np.sin(x / 2))
    # edge case
    badx = np.isnan(y)
    y[badx] = (-1) ** ((n - 1) * x[badx] / (2 * np.pi))

    return y



def array2cart(Az, El):
    """Converts azimuth and elevation angles to X, Y and Z coordinates on a unit sphere.

    Parameters
    ----------
    Az : ndarray
        Azimuth angles in degrees.
    El : ndarray
        Elevation angles in degrees.

    Returns
    -------
    X : ndarray
        x coordinates.
    Y : ndarray
        y coordinates.
    Z : ndarray
            z coordinates.
    """
    Az = np.radians(Az)
    El = np.radians(El)

    X = np.cos(Az) * np.cos(El)
    Y = np.sin(Az) * np.cos(El)
    Z = np.sin(El)

    return X, Y, Z

def cart2array(X, Y, Z):
    """This function will turn the X, Y and Z coordinate to azimuth and elevation angles assuming a unit sphere.

    Parameters
    ----------
    X : ndarray
        x coordinates.
    Y : ndarray
        y coordinates.
    Z : ndarray
            z coordinates.

    Returns
    ----------
    Az : ndarray
        Azimuth angles in degrees.
    El : ndarray
        Elevation angles in degrees.
    """

    Az = np.degrees(np.arctan2(Y, X))
    El = np.degrees(np.arcsin(Z))

    return Az, El


def rotmatrix(Az_0, El_0):
    """Makes a rotation matrix.First rotate about the z axis and then rotate about the new y axis
    Reference:
    http://www.agi.com/resources/help/online/stk/11.0/index.htm#comm/CommRadar03-03.htm

    Parameters
    ----------
    Az_0 : ndarray
        Azimuth rotation angles in degrees.
    El_0 : ndarray
        Elevation roation angles in degrees.

    Returns
    -------
    rotmat : ndarray
        A 3x3 rotation matrix.
    """

    Az_0 = np.radians(Az_0)
    El_0 = np.radians(El_0)

    R_Az = np.array(
        [
            [np.cos(Az_0), -np.sin(Az_0), 0.0],
            [np.sin(Az_0), np.cos(Az_0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_El = np.array(
        [
            [np.cos(El_0), 0.0, np.sin(El_0)],
            [0.0, 1.0, 0.0],
            [-np.sin(El_0), 0.0, np.cos(El_0)],
        ]
    )

    return R_El.dot(R_Az)


def rotcoords(Az, El, Az_0, El_0):
    """This function will rotate the Az and Elevation cordinates given offset
    angles. This will use a rotation matrix after the angles have been changed to Cartisian coordinates assuming a unit sphere.

    Parameters
    ----------
    Az : ndarray
        Azimuth angles in degrees.
    El : ndarray
        Elevation angles in degrees.
    Az_0 : ndarray
        Azimuth rotation angles in degrees.
    El_0 : ndarray
        Elevation roation angles in degrees.

    Returns
    -------
    Az_out : ndarray
        Rotated azimuth angles in degrees
    El_out : ndarray
        Rotated elevation angles in degrees.
    """
    cartcords = array2cart(Az, El)
    cartmat = np.column_stack(cartcords).transpose()
    rotmat = rotmatrix(Az_0, El_0)

    rotcart = rotmat.dot(cartmat)

    return cart2array(rotcart[0], rotcart[1], rotcart[2])
