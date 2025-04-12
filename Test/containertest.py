from pathlib import Path
import dateutil.parser
from datetime import datetime,timedelta,timezone
import numpy as np
import xarray as xr
import pandas as pd
from SimISR.testfunctions import chapman_func, temp_profile
from SimISR import IonoContainer,make_iline_specds


def example_iono():
    # make a cube

    xy = np.linspace(-1000,1000,200)
    z = np.linspace (50,1000,100)

    xm,ym,zm = np.meshgrid(xy,xy,z)
    coords = np.zeros((xm.size, 3))
    coords[:, 0] = xm.flatten()
    coords[:, 1] = ym.flatten()
    coords[:, 2] = zm.flatten()
    N_0=1e11
    z_0=250.0
    H_0=50.0

    Ne_profile = chapman_func(zm.flatten(), H_0, z_0, N_0)
    (Te, Ti) = temp_profile(zm.flatten())
    times=np.array([[0, 1e6]])

    (Nlocs, ndims) = coords.shape
    Ntime = len(times)
    vel = np.zeros((Nlocs, Ntime, ndims))
    species = ["O+", "e-"]
    # put the parameters in order
    params = np.zeros((Ne_profile.size, len(times), 2, 2))
    params[:, :, 0, 1] = np.repeat(Ti[:, np.newaxis], Ntime, axis=1)
    params[:, :, 1, 1] = np.repeat(Te[:, np.newaxis], Ntime, axis=1)
    params[:, :, 0, 0] = np.repeat(Ne_profile[:, np.newaxis], Ntime, axis=1)
    params[:, :, 1, 0] = np.repeat(Ne_profile[:, np.newaxis], Ntime, axis=1)

    iono = IonoContainer(coords,params,times,species = species,velocity=vel)
    return iono
def example_xr():

    xy = np.linspace(-500,500,50)
    z = np.linspace (50,1000,100)

    xm,ym,zm = np.meshgrid(xy,xy,z)

    coords = np.zeros((xm.size, 3))
    coords[:, 0] = xm.flatten()
    coords[:, 1] = ym.flatten()
    coords[:, 2] = zm.flatten()
    N_0=1e11
    z_0=250.0
    H_0=50.0

    Ne_profile = chapman_func(zm.flatten(), H_0, z_0, N_0)
    (Te, Ti) = temp_profile(zm.flatten())
    Te = Te[:,np.newaxis].repeat(2,axis=1)
    Ti = Ti[:,np.newaxis].repeat(2,axis=1)
    dt1 = dateutil.parser.parse("2024-01-01T15:24:00Z").replace(tzinfo=timezone.utc)
    times=pd.date_range(dt1, freq=timedelta(hours=1),periods=2)
    d1 = ['locs','time']
    # d2 = ['locs','time','spdims']

    Ne_all = Ne_profile[...,np.newaxis].repeat(2,axis=1)
    Ni_non = np.zeros_like(Ne_all)
    idict = {"ne":(d1,Ne_all,{"units": "m^-3"}),
            "nO+":(d1,Ne_all,{"units": "m^-3"}),
            "nH+":(d1,Ni_non,{"units": "m^-3"}),
            "nHe+":(d1,Ni_non,{"units": "m^-3"}),
            "nO2+":(d1,Ni_non,{"units": "m^-3"}),
            "nN0+":(d1,Ni_non,{"units": "m^-3"}),
            "nN+":(d1,Ni_non,{"units": "m^-3"}),
            "Te":(d1,Te,{"units": "K"}),
            "Ti":(d1,Ti,{"units": "K"}),
            "vdop":(d1,Ni_non,{"units": "m/s"})}
           # "vel":(d2,vel)}
        # vx=(d1,vel[...,0]))
    coords = dict(x=(['locs'],xm.flatten()),
        y=(['locs'],ym.flatten()),
        z=(['locs'],zm.flatten()),time=times)
        #spdims=[0,1,2])

    attrs = dict(originlla=np.array([42.6195,-71.49173,146.0]),
                species = ["O+", "e-"])
    i_ds = xr.Dataset(idict,coords=coords,attrs=attrs)

    return i_ds

def example_xr1d():

    xy = np.zeros(1)
    z = np.linspace(100,1000,100)

    xm,ym,zm = np.meshgrid(xy,xy,z)

    coords = np.zeros((xm.size, 3))
    coords[:, 0] = xm.flatten()
    coords[:, 1] = ym.flatten()
    coords[:, 2] = zm.flatten()
    N_0=1e11
    z_0=250.0
    H_0=50.0

    Ne_profile = chapman_func(zm.flatten(), H_0, z_0, N_0)
    (Te, Ti) = temp_profile(zm.flatten())
    Te = Te[:,np.newaxis].repeat(2,axis=1)
    Ti = Ti[:,np.newaxis].repeat(2,axis=1)

    times=pd.date_range("2024-01-01T15:24:00Z", freq=timedelta(hours=1),periods=2)
    d1 = ['locs','time']
    # d2 = ['locs','time','spdims']

    Ne_all = Ne_profile[...,np.newaxis].repeat(2,axis=1)
    Ni_non = np.zeros_like(Ne_all)
    idict = {"ne":(d1,Ne_all,{"units": "m^-3"}),
            "nO+":(d1,Ne_all,{"units": "m^-3"}),
            "nH+":(d1,Ni_non,{"units": "m^-3"}),
            "nHe+":(d1,Ni_non,{"units": "m^-3"}),
            "nO2+":(d1,Ni_non,{"units": "m^-3"}),
            "nN0+":(d1,Ni_non,{"units": "m^-3"}),
            "nN+":(d1,Ni_non,{"units": "m^-3"}),
            "Te":(d1,Te,{"units": "K"}),
            "Ti":(d1,Ti,{"units": "K"}),
            "vdop":(d1,Ni_non,{"units": "m/s"})}

    coords = dict(x=(['locs'],xm.flatten(),{"units": "km"}),
        y=(['locs'],ym.flatten(),{"units": "km"}),
        z=(['locs'],zm.flatten(),{"units": "km"}),time=times)
        #spdims=[0,1,2])

    attrs = dict(originlla=np.array([42.6195,-71.49173,146.0]),
                species = ["O+", "e-"])
    i_ds = xr.Dataset(idict,coords=coords,attrs=attrs)

    return i_ds

def create_spectrum(i_ds):

    cf = 440.2e6
    sf = 50e3
    nfft = int(2**10)
    f_vec = np.fft.fftshift(np.fft.fftfreq(nfft,d=1/sf))
    spec_args = dict(centerFrequency=440.2e6,sampfreq=sf,f=f_vec)
    ilineds = make_iline_specds(i_ds,spec_args)
    return ilineds

if __name__ == '__main__':
    iono=example_xr1d()
    param_file = Path("onedionosphere_params.nc")
    if param_file.exists():
        param_file.unlink()
    spec_file = Path("onedionosphere_spec.nc")
    if spec_file.exists():
        spec_file.unlink()
    iono.to_netcdf(str(param_file))
    iono_spec = create_spectrum(iono)
    iono_spec.to_netcdf((spec_file))
    print('Saved data')
