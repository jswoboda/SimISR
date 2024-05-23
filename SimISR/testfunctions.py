#!python

import numpy as np
from IonoContainer import IonoContainer

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


def make_test_ex(
    testv=False,
    testtemp=False,
    N_0=1e11,
    z_0=250.0,
    H_0=50.0,
    coords=None,
    times=np.array([[0, 1e6]]),
):
    """ This function will create a test ionoclass with an electron density that follows a chapman function.
    
    Parameters
    ----------
    testv : bool
        If false then all of the velocity values will be zero. 
    testtemp : bool 
        If true then a tempreture profile will be used. If not  Te and Ti are a constant of 2000 k.
    N_0 : float 
        The peak electron density.
    Z_0 : float
        The peak density location.
    H_0 : float 
        A single float of the scale height in km.
    coords : array_like
        A list of coordinates that the data will be created over.
    times : array_like
        A list of times the data will be created over.
    
    Returns 
    -------
    Icont : ionocontainer
        An instance that is a test.
    """
    if coords is None:
        xvec = np.arange(-250.0, 250.0, 20.0)
        yvec = np.arange(-250.0, 250.0, 20.0)
        zvec = np.arange(50.0, 900.0, 2.0)
        # Mesh grid is set up in this way to allow for use in MATLAB with a simple reshape command
        xx, zz, yy = np.meshgrid(xvec, zvec, yvec)
        coords = np.zeros((xx.size, 3))
        coords[:, 0] = xx.flatten()
        coords[:, 1] = yy.flatten()
        coords[:, 2] = zz.flatten()
        zzf = zz.flatten()
    else:
        zzf = coords[:, 2]
    #    H_0 = 50.0 #km scale height
    #    z_0 = 250.0 #km
    #    N_0 = 10**11

    # Make electron density
    Ne_profile = chapman_func(zzf, H_0, z_0, N_0)
    # Make temperture background
    if testtemp:
        (Te, Ti) = temp_profile(zzf)
    else:
        Te = np.ones_like(zzf) * 2000.0
        Ti = np.ones_like(zzf) * 1500.0

    # set up the velocity
    (Nlocs, ndims) = coords.shape
    Ntime = len(times)
    vel = np.zeros((Nlocs, Ntime, ndims))

    if testv:
        vel[:, :, 2] = np.repeat(zzf[:, np.newaxis], Ntime, axis=1) / 5.0
    species = ["O+", "e-"]
    # put the parameters in order
    params = np.zeros((Ne_profile.size, len(times), 2, 2))
    params[:, :, 0, 1] = np.repeat(Ti[:, np.newaxis], Ntime, axis=1)
    params[:, :, 1, 1] = np.repeat(Te[:, np.newaxis], Ntime, axis=1)
    params[:, :, 0, 0] = np.repeat(Ne_profile[:, np.newaxis], Ntime, axis=1)
    params[:, :, 1, 0] = np.repeat(Ne_profile[:, np.newaxis], Ntime, axis=1)

    Icont1 = IonoContainer(
        coordlist=coords,
        paramlist=params,
        times=times,
        sensor_loc=np.zeros(3),
        ver=0,
        coordvecs=["x", "y", "z"],
        paramnames=None,
        species=species,
        velocity=vel,
    )
    return Icont1
