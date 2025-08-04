#!/usr/bin/env python
import re
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import numpy as np
import xarray as xr
import pandas as pd
from SimISR import make_iline_specds
from SimISR.CoordTransforms import sphereical2Cartisian

from SimISR import experiment_setup,experiment_close,run_exp




def example_iri2016(t_st, t_end, t_step):
    """Creates an xarray dataset from the iri2016

    Parameters
    ----------
    t_st : Datetime
        Start time of parameters
    t_end : Datetime
        End time of the parameters
    t_step : TimeDelta
        Step size of the IRI simulation in time.

    Returns
    -------
    sim : xarray.DataSet
        The xarray dataset out of IRI2016.

    """
    import iri2016.profile as iri

    alt_km_range = (100, 1000, 2.0)
    glat = 42.6195
    glon = -71.49173

    sim = iri.timeprofile((t_st, t_end), t_step, alt_km_range, glat, glon)
    sim.attrs.update({"alt": 146.0})

    return sim


def example_iri2020(t_st, t_end, t_step):
    """Creates an xarray dataset from the iri2020

    Parameters
    ----------
    t_st : str
        Start time of parameters as an ISO string
    t_end : str
        End time of the parameters as an ISO string
    t_step : float
        Step size of the IRI simulation in number of hours

    Returns
    -------
    sim : xarray.DataSet
        The xarray dataset out of IRI2016.

    """
    import iri2020.times as time_profile

    # t_st = "2020-06-01T00:00:00"
    # t_end = "2020-06-01T02:00:00"
    # t_step = 1.0


        # latitude: 9.39538
        # longitude: 167.479181
        # altitude: 2.0
    alt_km_range = (100, 1000, 2.0)
    glat = 42.6195
    glon = -71.49173
    sim = time_profile.main((t_st, t_end, t_step), alt_km_range, glat, glon)
    sim.attrs.update({"alt": 146.0})
    return sim


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


def iri_to_MHz(iri_prof):
    """Adjusts the xarray data from IRI to work with the SimISR format for Millstone Hill Zenith.

    Parameters
    ----------
    iri_prof : xarray.DataSet
        The xarray dataset out of IRI2016 or 2020.

    Returns
    -------
    xrout : xarray.DataSet
        The reformated dataset for SimISR spectral code.
    """
    cart_coords = get_zenith_coords()
    xrout = iri_tp_to_simisr(iri_prof, cart_coords)

    return xrout


def iri_tp_to_simisr(iri_prof, coord_vecs):
    """Adjusts the xarray data from IRI to work with the SimISR format. The IRI is assumed to be a time altitude profile.

    Parameters
    ----------
    iri_prof : xarray.DataSet
        The xarray dataset out of IRI2016 or 2020.
    coord_vecs : ndarray
        Needs to be the same size as the altitude vector.

    Returns
    -------
    xrout : xarray.DataSet
        The reformated dataset for SimISR spectral code.
    """
    time = iri_prof.time
    lla = [float(iri_prof.glat), float(iri_prof.glon), iri_prof.attrs["alt"]]

    xm, ym, zm = np.split(coord_vecs, 3, axis=-1)
    iri_interp = iri_prof.interp(
        alt_km=zm.flatten(), method="linear", kwargs={"fill_value": "extrapolate"}
    )
    dsname = list(iri_interp.keys())
    species_names = [iname for iname in dsname if re.match(r"n\w+\+", iname)]
    species = []
    for ispec in species_names:
        numden = iri_interp[ispec].to_numpy() / iri_interp["ne"].to_numpy()
        if np.any(numden > 1e-6):
            species.append(ispec[1:])
        else:
            iri_interp.drop_vars(ispec)

    species.append("e-")
    attrs = {"originlla": np.array(lla), "species": species}

    coords = dict(
        x=(["locs"], xm.flatten(), {"units": "km"}),
        y=(["locs"], ym.flatten(), {"units": "km"}),
        z=(["locs"], zm.flatten(), {"units": "km"}),
        time=time,
    )

    dims = ("time", "locs")
    dims = ("locs", "time")
    data_dict = iri_interp.transpose().to_dict()["data_vars"]

    idict = {}
    for iname, iitem in data_dict.items():

        # if iitem['dims']==('time', 'alt_km'):
        if iitem["dims"] == ("alt_km", "time"):
            idict[iname] = (dims, iitem["data"])

    xrout = xr.Dataset(idict, coords=coords, attrs=attrs)
    if "glat" in xrout:
        xrout.drop_vars(["glat"])
    if "glon" in xrout:
        xrout.drop_vars(["glon"])

    xrout = xrout.assign_coords(time=pd.DatetimeIndex(xrout.time.data,tz=pytz.utc))
    return xrout


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
    utc_tz = pytz.timezone("UTC")
    st_str = "2024-11-17T15:24:00"
    et_str = "2024-11-17T17:24:00"
    t_2020 = .1
    st = datetime.fromisoformat(st_str).replace(tzinfo=utc_tz)
    et = datetime.fromisoformat(et_str).replace(tzinfo=utc_tz)
    t_step = timedelta(minutes=5)
    exp_obj = experiment_setup(expfile, test_dir, st.strftime("%Y-%m-%dT%H%M%SZ"))
    # iri2016_data = example_iri2016(st, et, t_step)
    iri2020_data = example_iri2020(st_str, et_str, t_2020)
    test_path = Path(test_dir)
    iri2020_data.to_netcdf(str(test_path.joinpath("iri_out.h5")))
    i_ds = iri_to_MHz(iri2020_data)

    spec_ds = create_spectrum(i_ds)
    i_ds_cp = i_ds.copy()
    t1 = i_ds_cp['time'].data
    i_ds_cp = i_ds_cp.assign_coords(time=t1.tz_localize(None))
    i_ds_cp.to_netcdf(str(test_path.joinpath("ionoparams.nc")))

    run_exp(exp_obj,spec_ds)
    spec_ds_cp = spec_ds.copy()
    t1 = spec_ds_cp['time'].data
    spec_ds_cp = spec_ds_cp.assign_coords(time=t1.tz_localize(None))
    spec_ds_cp.to_netcdf(str(test_path.joinpath("spectrum.nc")))

    experiment_close(exp_obj)


if __name__ == "__main__":

    curfile = Path(__file__)
    expfile = str(curfile.parent.parent.joinpath("config", "experiments", "mhzexp.yml"))
    test_path = Path("~/DATA/SimISR/version2tests/iri2020")
    test_dir = str(test_path.expanduser())
    print(f"Experiment file: {expfile}")
    print(f"Output directory: {test_dir}")
    run_full(expfile, test_dir)
