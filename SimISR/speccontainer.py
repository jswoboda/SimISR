import numpy as np
import xarray as xr
from ISRSpectrum import Specinit, PLspecinit



def make_pline_specds(i_ds,spec_args):
    """

    i_ds : xarray.Dataset
        A dataset for the ionosphere parameters.
    spec_args : dict
        Dictionary containing the arguments for the the spectrum.
    """

    attrs = i_ds.attrs.copy()

    coords = i_ds.coords.copy()
    idx_coords = coords.copy().to_index()

    sp1 = PLspecinit(**spec_args)

def make_iline_specds(i_ds,spec_args,spec_types=['i_line']):
    """Make an xarray dataset with the spectra data.

    Parameters
    ----------
    i_ds : xarray.Dataset
        A dataset for the ionosphere parameters.
    spec_args : dict
        Dictionary containing the arguments for the the spectrum
    spec_types : list
        List of the types of spectrums, either ion-line 'i_line' or 'p_line' for plasma line.

    Returns
    -------
    sp_ds : xarray.Dataset
        The data set with the same location and time coordinates but now with a frequency vector.
    """

    # Copy the coordinates and attrs so we don't have issues with overwritting.
    attrs = i_ds.attrs.copy()
    coords = i_ds.coords.copy()
    idx_coords = coords.copy().to_index()

    # Get the main spectrum
    sp1 = Specinit(**spec_args)
    # Add the frequency vector
    coords['freqs'] = sp1.f
    nfreq = len(sp1.f)
    sr_num = int(spec_args['sampfreq'])
    sr_den = 1
    attrs['sr_num'] = sr_num
    attrs['sr_den'] = sr_den

    species = attrs['species']
    n_sp = len(species)

    ne= i_ds.ne.to_numpy()
    nloc,ntime = ne.shape

    dims = ['locs','time','freqs']
    rc_arr = np.empty((nloc,ntime))
    he_arr = np.empty((nloc,ntime))
    sp_arr = np.empty((nloc,ntime,nfreq))
    data_block = np.empty((n_sp,2))
    d_dict = {'iline':(dims, sp_arr), 'rcs':(dims[:-1], rc_arr), 'he':(dims[:-1], he_arr)}
    sp_ds = xr.Dataset(d_dict,coords=coords,attrs=attrs)

    klist = list(i_ds.keys())
    den_names = ['ne']*len(species)
    t_names = ['Ti']*len(species)
    t_names[-1] = 'Te'
    for inum,isp in enumerate(species[:-1]):
        den_names[inum] = 'n'+isp
        if 't'+isp in klist:
            t_names[inum] = 't'+isp

    for idx in idx_coords:
        data_block[-1,0] = i_ds['ne'].loc[idx].data
        data_block[-1,1] = i_ds['Te'].loc[idx].data
        for inum,(iden,itemp) in enumerate(zip(den_names,t_names)):
            data_block[inum,0] = i_ds[iden].loc[idx].data
            data_block[inum,1] = i_ds[itemp].loc[idx].data
        __,sp_ds.iline.loc[idx],sp_ds.rcs.loc[idx], htmp = sp1.getspecsep(data_block, species, rcsflag=True, heflag=True)
        sp_ds.he.loc[idx] = htmp*1e-3

    return sp_ds
