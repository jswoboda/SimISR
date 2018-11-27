#!/usr/bin/env python

from datetime import datetime
import calendar
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pyglow.pyglow import Point
from SimISR.IonoContainer import IonoContainer
from SimISR.utilFunctions import getdefualtparams, getdefualtparams, makeconfigfile
from SimISR.analysisplots import analysisdump
from SimISR.runsim import main as runsimisr
from SimISR import Path


def pyglowinput(latlonalt=[65.1367, -147.4472, 250.00], dn_list=[datetime(2015, 3, 21, 8, 00), datetime(2015, 3, 21, 20, 00)], z=None):


    if z is None:
        z = sp.linspace(50., 1000., 200)
    dn_diff = sp.diff(dn_list)
    dn_diff_sec = dn_diff[-1].seconds
    timelist = sp.array([calendar.timegm(i.timetuple()) for i in dn_list])
    time_arr = sp.column_stack((timelist, sp.roll(timelist, -1)))
    time_arr[-1, -1] = time_arr[-1, 0]+dn_diff_sec

    v=[]
    coords = sp.column_stack((sp.zeros((len(z), 2), dtype=z.dtype), z))
    all_spec = ['O+', 'NO+', 'O2+', 'H+', 'HE+']
    Param_List = sp.zeros((len(z), len(dn_list),len(all_spec),2))
    for idn, dn in enumerate(dn_list):
        for iz, zcur in enumerate(z):
            latlonalt[2] = zcur
            pt = Point(dn, *latlonalt)
            pt.run_igrf()
            pt.run_hwm93()
            pt.run_msis()
            pt.run_iri()

            # so the zonal pt.u and meriodinal winds pt.v  will coorispond to x and y even though they are
            # supposed to be east west and north south. Pyglow does not seem to have
            # vertical winds.
            v.append([pt.u, pt.v, 0])

            for is1, ispec in enumerate(all_spec):
                Param_List[iz, idn, is1, 0] = pt.ni[ispec]*1e6

            Param_List[iz, idn, :, 1] = pt.Ti

            Param_List[iz, idn, -1, 0] = pt.ne*1e6
            Param_List[iz, idn, -1, 1] = pt.Te
    Param_sum = Param_List[:, :, :, 0].sum(0).sum(0)
    spec_keep = Param_sum > 0.
    species = sp.array(all_spec)[spec_keep[:-1]].tolist()
    species.append('e-')
    Param_List[:, :] = Param_List[:, :, spec_keep]
    Iono_out = IonoContainer(coords, Param_List, times = time_arr, species=species)
    return Iono_out


def makedatasets(maindir='~/DATA/Ion_Comp_Exp/', nomult=sp.arange(1, 5, dtype=float), baud_len=[7, 14, 21]):
    """
        This will make all of the data sets.
    """
    basepath = Path(maindir).expanduser()
    basepath.mkdir(exist_ok=True)
    iono_orig = pyglowinput()
    nion = len(iono_orig.Species)-1
    npulses = 1000
    (sensdict, simparams) = getdefualtparams()
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['RangeLims'] = [80., 750.]
    simparams['Tint'] = ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = 12*tint
    ts = simparams['t_s']
    simparams['beamrate'] = 1
    simparams['outangles'] = [[0]]
    simparams['species'] = ['O+', 'NO+', 'O2+', 'H+', 'e-']
    origtime = iono_orig.Time_Vector[0,0]

    for imult in nomult:
        curiono = iono_orig.copy()
        curiono.Param_List[:, :, 1, 0] *= imult
        ratio = curiono.Param_List[:, :, -1, 0]/curiono.Param_List[:, :, :-1, 0].sum(-1)
        ratio_ar = sp.repeat(ratio[:, :, sp.newaxis], nion, axis=-1)
        curiono.Param_List[:, :, :-1, 0] *= ratio_ar
        figout = plotiono(curiono)[0]
        for ibaud in baud_len:
            newpath = basepath.joinpath('NOMult{0:02d}baud{1:02d}'.format(int(imult), int(ibaud)))
            newpath.joinpath('Origparams').mkdir(exist_ok=True, parents=True)
            figname = newpath.joinpath('InputParams.png')
            figout.savefig(str(figname))
            dataname = newpath.joinpath('Origparams', '{0:d}Params.h5'.format(origtime))
            curiono.saveh5(str(dataname))
            simparams['Pulselength'] = ibaud*ts
            sfile = str(newpath/'startfile.h5')
            iono_orig.saveh5(sfile)
            simparams['startfile'] = sfile
            makeconfigfile(str(newpath/'chem_test.yml'), [23465.], 'millstone', simparams)
def plotiono(ionoin):

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    (figmplf, axmat) = plt.subplots(1, 2, figsize=(10, 5), facecolor='w', sharey=True)
    species = ionoin.Species
    zvec = ionoin.Cart_Coords[:, 2]
    params = ionoin.Param_List[:, 0]
    maxden = 10**sp.ceil(sp.log10(params[:, -1, 0].max()))
    for iplot, ispec in enumerate(species):
        axmat[0].plot(params[:, iplot, 0], zvec, label=ispec +'Density')
        axmat[1].plot(params[:, iplot, 1], zvec, label=ispec+'Temperture')

    axmat[0].set_title('Number Density')
    axmat[0].set_xscale('log')
    axmat[0].set_ylim([50, 1000])
    axmat[0].set_xlim([maxden*1e-5, maxden])
    axmat[0].set_xlabel(r'Densities in m$^{-3}$')
    axmat[0].set_ylabel('Alt in km')
    axmat[0].legend()

    axmat[1].set_title('Temperture')
    axmat[1].set_xlim([100., 2500.])
    axmat[1].set_xlabel(r'Temp in $^{\circ}$ K')
    axmat[1].set_ylabel('Alt in km')
    axmat[1].legend()
    plt.tight_layout()
    return figmplf,axmat


if __name__ == '__main__':
    exppath = Path('~/DATA/Ion_Comp_Exp').expanduser()
    drpath = Path('~/Dropbox (MIT)/').expanduser()
    funclist = ['spectrums', 'radardata','fitting']
    list1 = [i for i in exppath.glob('NO*')]
    configlist = [i.joinpath('chem_test.yml') for i in exppath.glob('NO*')]

    for idata,icon in zip(list1,configlist):
        runsimisr(funclist,str(idata),str(icon),True)
        
    for idata,icon in zip(list1,configlist):
        analysisdump(str(idata),str(icon),params = ['Ne','Nepow','Te','Ti'])

    for ipath in list1:
        idir=ipath.joinpath('AnalysisPlots')
        idest = drpath.joinpath(*ipath.parts[-2:])
        shutil.copytree(str(idir),str(idest))
#     main()
