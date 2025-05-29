#!python

import numpy as np

#%% Test functions
def chapman_func(z, H_0, Z_0, N_0):
    """This function will return the Chapman function for a given altitude vector z.  All of the height values are assumed km.

    Parameters
    ----------
    z : array_like
        An array of z values in km.
    H_0 : float
        A single float of the scale height in km.
    Z_0 : float
        The peak density location.
    N_0 : float
        The peak electron density.

    Returns
    -------
    Ne : array_like
        Electron density as a function of z in m^{-3}
    """
    z1 = (z-Z_0)/H_0
    Ne = N_0*np.exp(0.5*(1-z1-np.exp(-z1)))
    return Ne

def temp_profile(z, T0=1000., z0=100.):
    """This function creates a tempreture profile using arc tan functions for test purposes.

    Parameters
    ----------
    z : array_like
        An array of z values in km.
    T0 : array_like
        The value of the lowest tempretures in K.
    z0 : array_like
        The middle value of the atan functions along alitutude. In km.

    Returns
    -------
    Te : array_like
        The electron density profile in K. 1700*(atan((z-z0)2*exp(1)/400-exp(1))+1)/2 +T0
    Ti : array_like
        The ion density profile in K. 500*(atan((z-z0)2*exp(1)/400-exp(1))+1)/2 +T0
    """
    zall = (z-z0)*2.*np.exp(1)/400. -np.exp(1)
    atanshp = (np.tanh(zall)+1.)/2
    Te = 1700*atanshp+T0
    Ti = 500*atanshp+T0

    return (Te,Ti)
