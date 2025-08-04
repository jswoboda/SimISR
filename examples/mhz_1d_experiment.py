#!/usr/bin/env python

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from SimISR.testfunctions import chapman_func, temp_profile
from SimISR import make_iline_specds
from SimISR.CoordTransforms import sphereical2Cartisian

from SimISR import experiment_setup,experiment_close,run_exp




def example_ionosphere(coords):
    """Creates a data set that is modeled after a simple champman function ionosphere.

    Parameters
    ----------
    coords : ndarray
        x,y,z coordinates in km of the ionosphere dataset.

    Returns
    -------
    i_ds : xarray.DataSet
        The dataset of ionospheric parameters that will be used for SimISR.
    """
    xm, ym, zm = np.split(coords, 3, axis=-1)
    N_0 = 1e11
    z_0 = 250.0
    H_0 = 50.0

    Ne_profile = chapman_func(zm.flatten(), H_0, z_0, N_0)
    (Te, Ti) = temp_profile(zm.flatten())
    Te = Te[:, np.newaxis].repeat(2, axis=1)
    Ti = Ti[:, np.newaxis].repeat(2, axis=1)

    times = pd.date_range("2024-01-01T15:24:00Z", freq=pd.Timedelta(hours=1), periods=2)
    d1 = ["locs", "time"]
    # d2 = ['locs','time','spdims']

    Ne_all = Ne_profile[..., np.newaxis].repeat(2, axis=1)
    Ni_non = np.zeros_like(Ne_all)
    idict = {
        "ne": (d1, Ne_all, {"units": "m^-3"}),
        "nO+": (d1, Ne_all, {"units": "m^-3"}),
        "nH+": (d1, Ni_non, {"units": "m^-3"}),
        "nHe+": (d1, Ni_non, {"units": "m^-3"}),
        "nO2+": (d1, Ni_non, {"units": "m^-3"}),
        "nN0+": (d1, Ni_non, {"units": "m^-3"}),
        "nN+": (d1, Ni_non, {"units": "m^-3"}),
        "Te": (d1, Te, {"units": "K"}),
        "Ti": (d1, Ti, {"units": "K"}),
        "vdop": (d1, Ni_non, {"units": "m/s"}),
    }

    coords = dict(
        x=(["locs"], xm.flatten(), {"units": "km"}),
        y=(["locs"], ym.flatten(), {"units": "km"}),
        z=(["locs"], zm.flatten(), {"units": "km"}),
        time=times,
    )
    # spdims=[0,1,2])

    attrs = dict(originlla=np.array([42.6195, -71.49173, 146.0]), species=["O+", "e-"])
    i_ds = xr.Dataset(idict, coords=coords, attrs=attrs)

    return i_ds


def example_xr1d():

    xy = np.zeros(1)
    z = np.linspace(100, 1000, 500)

    xm, ym, zm = np.meshgrid(xy, xy, z)

    coords = np.zeros((xm.size, 3))
    coords[:, 0] = xm.flatten()
    coords[:, 1] = ym.flatten()
    coords[:, 2] = zm.flatten()

    return example_ionosphere(coords)


def get_zenith_coords():
    """Creates an xyz coordinate system for the Zenith antenna at Millstone Hill.

    Returns
    -------
    coords : ndarray
        xyz coordinates for the Millstone Hill Zenith antenna.
    """
    r_vec = np.arange(100, 1000, 2.0)
    az_vec = np.array([178.0])
    el_vec = np.array([88.0])
    (
        rm,
        am,
        elm,
    ) = np.meshgrid(r_vec, az_vec, el_vec)
    coords = np.zeros((rm.size, 3))
    coords[:, 0] = rm.flatten()
    coords[:, 1] = am.flatten()
    coords[:, 2] = elm.flatten()

    cart_coords = sphereical2Cartisian(coords)
    return cart_coords


def example_xr1d_zenith():
    """Create an example ionosphere specifically for Millstone Hill Zenith.

    Returns
    -------
    example_iono : xarray.DataSet
        A data set that has been made so it aligns with the angle for the Millstone Hill Zenith antenna.
    """
    cart_coords = get_zenith_coords()
    return example_ionosphere(cart_coords)


def create_spectrum(i_ds):
    """This will make the ion line spectra for the plasma parameters within the xarray data set input specifically for Millstone Hill UHF.

    Parameters
    ----------
    i_ds : xarray.DataSet
        Holds the plasma parameters to be turned into ionline spectra.

    Returns
    -------
    ilineds : xarray.DataSet
        Holds the ion-line spectra.
    """
    cf = 440.2e6
    sf = 50e3
    nfft = int(2**10)
    f_vec = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sf))
    spec_args = dict(centerFrequency=cf, sampfreq=sf, f=f_vec)
    ilineds = make_iline_specds(i_ds, spec_args)
    return ilineds


def run_full(expfile, test_dir):
    """Creates the data, the experiment object and runs SimISR.

    expfile : str
        The experiment file that will be used as a template.
    test_dir : str
        Directory where the data will be located.

    """
    start_time = "2024-01-01T15:24:00Z"
    exp_obj = experiment_setup(expfile, test_dir, start_time)
    i_ds = example_xr1d_zenith()
    spec_ds = create_spectrum(i_ds)

    run_exp(exp_obj,spec_ds)

    experiment_close(exp_obj)


if __name__ == "__main__":

    curfile = Path(__file__)
    expfile = str(curfile.parent.parent.joinpath("config", "experiments", "mhzexp.yml"))

    test_dir = "~/DATA/SimISR/version2tests/experimentclass"
    print(f"Experiment file: {expfile}")
    print(f"Output directory: {test_dir}")
    run_full(expfile, test_dir)
