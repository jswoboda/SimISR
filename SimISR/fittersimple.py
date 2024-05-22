#!/usr/bin/env python
"""
fittersimple.py
This module will implement a simple fitter for ISR spectra.
@author: John Swoboda
"""

import scipy as sp
import scipy.optimize
from pathlib import Path
from SimISR.IonoContainer import IonoContainer, makeionocombined
from SimISR.utilFunctions import readconfigfile, update_progress,spect2acf
import SimISR.specfunctions as specfuncs

PVALS = [1e11, 2.1e3, 1.1e3, 0.]
SVALS = [1.1e3, 1e11, 2.1e3, 0.]
SIMVALUES = sp.array([[PVALS[0],PVALS[2]],[PVALS[0],PVALS[1]]])
curloc = Path(__file__).resolve().parent
defcon = curloc/'statsbase.ini'

def runfitter(paramdict, SNR, n_pulses, n_runs, Niratio, x_0=SVALS):
    """
        paramdict
    """

    data = SIMVALUES
    z = sp.linspace(50., 1e3, 50)
    nz = len(z)
    params = sp.tile(data[sp.newaxis, sp.newaxis, :, :], (nz, 1, 1, 1))
    coords = sp.column_stack((sp.ones(nz), sp.ones(nz), z))
    species = ['O+', 'e-']
    times = sp.array([[0, 1e9]])
    vel = sp.zeros((nz, 1, 3))

    (sensdict, simparams) = readconfigfile(defcon)

    species = paramdict['species']
    nspecies = len(species)
    ni = nspecies-1
    Icont1 = IonoContainer(coordlist=coords, paramlist=params, times=times,
                           sensor_loc=sp.zeros(3), ver=0, coordvecs=['x', 'y', 'z'],
                           paramnames=None, species=species, velocity=vel)

    omeg, outspecs = specfuncs.ISRSspecmake(Icont1, sensdict, int(simparams['numpoints']))

    tau, acf = spect2acf(omeg, outspecs)

    t_log = sp.logical_and(tau<simparams['Pulselength'],tau>=0)
    plen = t_log.sum()
    amb_dict = simparams['amb_dict']
    amb_mat_flat = sp.zeros_like(amb_dict['WttMatrix'])
    eyemat = sp.eye(plen)
    amb_mat_flat[:plen, t_log] = eyemat
    fitmode = simparams['fitmode']
    simparams['amb_dict']['WttMatrix'] = amb_mat_flat

    Nloc, Nt = acf.shape[:2]
    # output Files
    fittedarray = sp.zeros((Nloc, Nt, nparams+1))*sp.nan
    fittederror = sp.zeros((Nloc, Nt, nparams+1))*sp.nan
    fittedcov = sp.zeros((Nloc, Nt, 4, 4))*sp.nan
    funcevals = sp.zeros((Nloc, Nt))
    for iloc, ilocvec in acf:
        for itn, iacf in ilocvec:
            curlag = sp.dot(amb_mat_flat, iacf)
            std_lag = sp.absolute(curlag)(1.+1./SNR)/sp.sqrt(n_pulses)
            covar_lag = sp.diag(sp.power(std_lag,2.))

            curlag = curlag + std_lag*sp.random.randn(curlag.shape)
            d_func = (curlag, sensdict, simparams, Niratio)
            # Only fit Ti, Te, Ne and Vi

            # change variables because of new fit mode

            # Perform the fitting
            optresults = scipy.optimize.least_squares(fun=specfuncs.ISRSfitfunction, x0=x_0,
                                                      method='lm', verbose=0, args=d_func)
            x_res = optresults.x.real
            # Derive data for the ions using output from the fitter and ion
            # species ratios which are assumed to be given.
            ionstuff = sp.zeros(ni*2-1)
            ionstuff[:2*ni:2] = x_res[1]*Niratio
            ionstuff[1:2*ni-1:2] = x_res[0]
            # change variables because of new fit mode
            if fitmode == 1:
                x_res[2] = x_res[2]*x_res[0]
            fittedarray[iloc, itn] = sp.append(ionstuff,
                                               sp.append(x_res, Ne_start[iloc, itime]))
            funcevals[iloc, itn] = optresults.nfev
            #                fittedarray[iloc,itime] = sp.append(optresults.x,Ne_start[iloc,itime])
            resid = optresults.cost
            jac = optresults.jac
            # combine the rows because of the complex conjugates
            jacc = jac[0::2]+jac[1::2]
            try:
                # Derive covariances for the ions using output from the fitter and ion species ratios which are assumed to be given.
                #covf = sp.linalg.inv(sp.dot(jac.transpose(),jac))*resid/dof


                covf = sp.linalg.inv(sp.dot(sp.dot(jacc.transpose(),
                                                   sp.linalg.inv(covar_lag)), jacc))

                # change variables because of new fit mode
                if fitmode == 1:
                    # is this right?
                    covf[2] = covf[2]*x_res[0]
                    covf[:,2] = covf[:,2]*x_res[0]
                vars_vec = sp.diag(covf).real
                ionstuff = sp.zeros(ni*2-1)
                ionstuff[:2*ni:2] = vars_vec[1]*Niratio
                ionstuff[1:2*ni-1:2] = vars_vec[0]
                vars_vec = sp.append(ionstuff, vars_vec)
                fittedcov[iloc, itn] = covf
            except:#sp.linalg.LinAlgError('singular matrix'):
                vars_vec = sp.ones(nparams)*float('nan')

        #                if len(vars_vec)<fittederror.shape[-1]-1:
        #                    pdb.set_trace()
        fittederror[iloc, itn, :-1] = vars_vec



    return(fittedarray, fittederror, funcevals, fittedcov)
def run_sim(args_commd):



    fitoutput = fitterone.fitdata(specfuncs.ISRSfitfunction,
                              fitterone.simparams['startfile'], fittimes=fitlist,
                              printlines=printlines)
    (fitteddata, fittederror, funcevals, fittedcov) = fitoutput


    fittederronly = sp.sqrt(fittederror)
    paramnames = []
    species = fitterone.simparams['species']
    # Seperate Ti and put it in as an element of the ionocontainer.
    Ti = fitteddata[:, :, 1]

    nTi = fittederronly[:, :, 1]

    nTiTe = fittedcov[:, :, 0, 1]
    nTiNe = fittedcov[:, :, 0, 2]
    nTiVi = fittedcov[:, :, 0, 3]
    nTeNe = fittedcov[:, :, 1, 2]
    nTeVi = fittedcov[:, :, 1, 3]
    nNeVi = fittedcov[:, :, 2, 3]
    cov_list = [nTiTe[:, :, sp.newaxis], nTiNe[:, :, sp.newaxis],
                nTiVi[:, :, sp.newaxis], nTeNe[:, :, sp.newaxis],
                nTeVi[:, :, sp.newaxis], nNeVi[:, :, sp.newaxis]]
    cov_list_names = ['nTiTe', 'nTiNe', 'nTiVi', 'nTeNe', 'nTeVi','nNeVi']
    paramlist = sp.concatenate([fitteddata, Ti[:, :, sp.newaxis], fittederronly,
                                nTi[:, :, sp.newaxis], funcevals[:, :, sp.newaxis]]
                               + cov_list, axis=2)

    ionoout = IonoContainer(Ionoin.Cart_Coords, paramlist.real, timevec, ver=newver,
                            coordvecs=Ionoin.Coord_Vecs, paramnames=paranamsf,
                            species=species)

    ionoout.saveh5(str(outfile))

def parse_command_line(str_input=None):
    # if str_input is None:
    parser = argparse.ArgumentParser()
    # else:
    #     parser = argparse.ArgumentParser(str_input)
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="prints debug output and additional detail.")
    parser.add_argument("-c", "--config", dest="config", default=str(defcon),
                        help="Use configuration file <config>.")
if __name__ == '__main__':
    """
        Main way run from command line
    """
    args_commd = parse_command_line()

    if args_commd.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)

    run_sim(args_commd)
