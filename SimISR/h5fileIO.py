#!python
"""

fileIO.py
Holds functions for file input/output
@author: John Swoboda

"""


import h5py
import numpy as np
from pathlib import Path


def save_dict_to_hdf5(dic, filename):
    """Save dictionary to h5 file.

    Saves a dictionary where the keys are labels for data sets and items are numpy
    arrays or other dictionaries which will then make another level of labels. This
    is a pass through function for recursively_save_dict_contents_to_group.

    Parameters
    ----------
    dic : Dictionary
        Dictionary to be saved.

    filename : str
        File name that the data will be saved to.
    """

    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", dic)


def load_dict_from_hdf5(filename):
    """Load dictionary from h5 file.

    Loads a dictionary where the keys are labels for data sets and items are numpy
    arrays or other dictionaries which will then make another level of labels. This
    is a pass through function for recursively_load_dict_contents_from_group.

    Parameters
    ----------
    filename : str
        File name that the data will be saved to.

    Returns
    -------
    dic : Dictionary
        Dictionary to be read.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f'{filename}, no such file exists')

    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/")


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Save dictionary to h5 file.

    Saves a dictionary where the keys are labels for data sets and items are numpy
    arrays or other dictionaries which will then make another level of labels.

    Parameters
    ----------
    h5file : h5py.File
        File object to write data.

    path : str
        File name that the data will be saved to.

    dic : Dictionary
        Dictionary to be saved.
    """
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        # print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            # print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(
            item, (np.int64, np.float64, str, float, np.float32, np.float16, int)
        ):
            # print( 'here' )
            h5file[path + key] = item
            if isinstance(item, str):
                utf8_type = h5py.string_dtype("utf-8", len(item))
                item = np.array(item.encode("utf-8"), dtype=utf8_type)
            if not h5file[path + key][()] == item:
                raise ValueError(
                    "The data representation in the HDF5 file does not match the original dict."
                )
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype("|S9")
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError(
                    "The data representation in the HDF5 file does not match the original dict."
                )
        elif item is None:
            continue
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        # other types cannot be saved and will result in an error
        else:
            # print(item)
            raise ValueError("Cannot save %s type." % type(item))


def recursively_load_dict_contents_from_group(h5file, path):
    """Load dictionary from h5 file.

    Loads a dictionary where the keys are labels for data sets and items are numpy
    arrays or other dictionaries which will then make another level of labels.

    Parameters
    ----------
    h5file : h5py.File
        File object to write data.

    path : str
        File name that the data will be saved to.

    Returns
    -------
    dic : Dictionary
        Dictionary to be read.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans
