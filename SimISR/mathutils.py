#!/usr/bin/env python
"""

@author: John Swoboda

"""
import numpy as np
import scipy.special

def diric(x, n):
    """ Based on octave-signal v. 1.10 diric.

    :Authors: Michael Hirsch
    """
    n = int(n)
    if n < 1:
        raise ValueError('n is a strictly positive integer')

    x = np.asanyarray(x)

    y = np.sin(n*x/2) / (n*np.sin(x/2))
    # edge case
    badx = np.isnan(y)
    y[badx] = (-1)**((n-1)*x[badx]/(2*np.pi))

    return y


def jinc(t):
    """ This will output a jinc function.

    Args:
        t (:obj:`numpy array`): Time array in seconds.

    Returns:
        outdata (:obj:`numpy array`): Jinc(t)
    """
    t = np.asanyarray(t)

    outdata = np.ones_like(t)
    outdata[t == 0.] = 0.5
    outdata[t != 0.0] = scipy.special.jn(1,np.pi*t[t!=0.0])/(2*t[t!=0.0])

    return outdata


def angles2xy(az, el):
    """Creates x and y coordinates from az and el arrays.

    Elevation angle measured from the z=0 plane.
    Azimuth angle from x=0 and goes clockwise.

    Args:
        az (:obj:`numpy array`): Azimuth angles in degrees.
        el (:obj:`numpy array`): Elevation angles in degrees.

    Returns:
        (xout (:obj:`numpy array`), yout (:obj:`numpy array`)): x and y coordinates.
   """

    azt = np.radians(az)
    elt = np.radians(90-el)
    xout = elt*np.sin(azt)
    yout = elt*np.cos(azt)

    return xout, yout

def array2cart(Az, El):
    """ Converts azimuth and elevation angles to X, Y and Z coordinates
        on a unit sphere.

    Args:
        Az (:obj:`numpy array`): Azimuth angles in degrees.
        El (:obj:`numpy array`): Elevation angles in degrees.

    Returns:
        (X (:obj:`numpy array`),  Y (:obj:`numpy array`), Z (:obj:`numpy array`)): x, y and z coordinates.
    """
    Az = np.radians(Az)
    El = np.radians(El)

    X = np.cos(Az)*np.cos(El)
    Y = np.sin(Az)*np.cos(El)
    Z = np.sin(El)

    return X, Y, Z


def cart2array(X, Y, Z):
    """ This function will turn the X, Y and Z coordinate to azimuth and elevation angles
        assuming a unit sphere.

    Args:
        X (:obj:`numpy array`): x coordinates.
        Y (:obj:`numpy array`): y coordinates.
        Z (:obj:`numpy array`): z coordinates.

    Returns:
        (Az (:obj:`numpy array`), El (:obj:`numpy array`)): Azimuth and elevation angles in degrees.
    """

    Az = np.degrees(np.arctan2(Y, X))
    El = np.degrees(np.arcsin(Z))

    return Az, El


def rotmatrix(Az_0, El_0):
    """ Makes a rotation matrix.

    This creates a rotation matrix for the rotcoords function. First rotate
    about the z axis and then rotate about the new y axis
    http://www.agi.com/resources/help/online/stk/11.0/index.htm#comm/CommRadar03-03.htm

    Args:
        Az_0 (float): The azimuth rotation angle in degrees.
        El_0 (float): The elevation rotation angle in degrees.

    Return:
        rotmat (:obj:`numpy array`): A 3x3 rotation matrix.
    """

    Az_0 = np.radians(Az_0)
    El_0 = np.radians(El_0)

    R_Az = np.array([[np.cos(Az_0), -np.sin(Az_0), 0.],
                     [np.sin(Az_0), np.cos(Az_0), 0.],
                     [0., 0., 1.]])
    R_El = np.array([[np.cos(El_0), 0., np.sin(El_0)],
                     [0., 1., 0.],
                     [-np.sin(El_0), 0., np.cos(El_0)]])

    return R_El.dot(R_Az)

def rotcoords(Az, El, Az_0, El_0):
    """ Applies rotation matrix to Az and El arrays.

    This function will rotate the Az and Elevation cordinates given offset
    angles. This will use a rotation matrix after the angles have been
    changed to Cartisian coordinates assuming a unit sphere.

    Args:
        Az (:obj:`numpy array`): Azimuth angles in degrees.
        El (:obj:`numpy array`): Elevation angles in degrees.
        Az_0 (float): The azimuth rotation angle in degrees.
        El_0 (float): The elevation rotation angle in degrees.

    Returns:
        (Az_out (:obj:`numpy array`), El_out (:obj:`numpy array`)): Rotated azimuth and elevation angles in degrees.
   """
    cartcords = array2cart(Az, El)
    cartmat = np.column_stack(cartcords).transpose()
    rotmat = rotmatrix(Az_0, El_0)

    rotcart = rotmat.dot(cartmat)

    return cart2array(rotcart[0], rotcart[1], rotcart[2])
