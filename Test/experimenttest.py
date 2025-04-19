#!/usr/bin/env python

from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from SimISR import Experiment, read_config_yaml
from SimISR.testfunctions import chapman_func, temp_profile
from SimISR import make_iline_specds



def experiment_setup(exp_file,test_dir,start_time):
    """"""
    exp_1 = read_config_yaml(exp_file,'experiment')
    exp_obj = Experiment(**exp_1)
    exp_obj.setup_channels(test_dir,start_time)

    return exp_obj

def experiment_close(exp_obj):
    """"""

    exp_obj.close_channels()

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
