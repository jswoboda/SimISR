#!/usr/bin/env python
"""This module holds a number of functions to help with running SimISR."""
from pathlib import Path
from .radarobjs import read_config_yaml, Experiment
from .createData import RadarDataCreate


def experiment_setup(exp_file, test_dir, start_time=None, end_time=None):
    """Sets Gets the experiment set up but doing some quick editing of the example experiment file.

    Parameters
    ----------
    exp_file : str
        Name of the experiment file used as the prototype.
    test_dir : str
        Location of the directory.
    start_time : str
        ISO time string.

    Returns
    -------
    exp_obj : SimISR.Experiment
        Experiment class object.
    """

    exp_1 = read_config_yaml(exp_file, "experiment")

    # add the starttime
    exp_obj = Experiment(**exp_1)
    exp_obj.set_times(start_time, end_time)
    if start_time is None:
        start_time = exp_obj.exp_start
    exp_obj.setup_channels(test_dir,start_time)

    return exp_obj


def experiment_close(exp_obj):
    """Closes the files at the end of the experiment.

    Parameters
    ----------
    exp_obj : SimISR.Experiment
        Experiment class object to be closed.
    """

    exp_obj.close_channels()


def run_exp(exp_obj, spec_ds):
    """Run the experiment given all of the channels

    Parameters
    ----------
    exp_obj : Experiment
        Experiment object with the channels set up.
    spec_ds : xarray.Dataset
        The dataset holding the spectra that will be used for SimISR.

    """
    rdr = RadarDataCreate(exp_obj)
    phys_ds = rdr.spatial_set_up(spec_ds.coords, spec_ds.attrs["originlla"])

    savepath = Path(exp_obj.save_directory)

    phys_ds.to_netcdf(str(savepath.joinpath("phys_setup.nc")))
    chan_names = list(exp_obj.iline_chans.keys())

    for ichan in chan_names:
        rx_name = ichan.split("-")[0]
        rdr.write_chan(spec_ds, phys_ds, rx_name, ichan)
