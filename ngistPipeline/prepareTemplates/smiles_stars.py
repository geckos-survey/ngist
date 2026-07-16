import glob
import logging
import os

import numpy as np
import pandas as pd

from astropy.io import fits
from ppxf.ppxf_util import gaussian_filter1d, log_rebin
from printStatus import printStatus


def prepareSpectralTemplateLibrary(
    config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid,
):
    """
    Prepares the spectral template library. The templates are loaded from disk,
    shortened to meet the spectral range in consideration, convolved to meet the
    resolution of the observed spectra (according to the LSF), log-rebinned, and
    normalised. In addition, they are sorted in a three-dimensional array
    sampling the parameter space in age, metallicity and alpha-enhancement.
    """
    printStatus.running("Preparing the stellar population templates")
    cvel = 299792.458

    # SSP model library
    sp_models = glob.glob(
        os.path.join(config["GENERAL"]["TEMPLATE_DIR"], config[module_used]["LIBRARY"])
        +"*/"+ "*"
    )

    sp_models.sort()
    ntemplates = len(sp_models)

    # Load wavelength and flux
    data = np.loadtxt(sp_models[0])

    lam = data[:, 0]
    ssp_data = data[:, 1]
    lamRange_spmod = np.array([lam[0], lam[-1]])
    cdelt1 = lam[1] - lam[0]

    # Determine length of templates
    template_overhead = np.zeros(2)
    if lmin - lamRange_spmod[0] > 150.0:
        template_overhead[0] = 150.0
    else:
        template_overhead[0] = lmin - lamRange_spmod[0] - 5
    if lamRange_spmod[1] - lmax > 150.0:
        template_overhead[1] = 150.0
    else:
        template_overhead[1] = lamRange_spmod[1] - lmax - 5

    # Shorten templates to size of data
    lamRange_lin = lam

    # Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
    constr = np.array([lmin - template_overhead[0], lmax + template_overhead[1]])
    idx_lam = np.where(
        np.logical_and(lamRange_lin > constr[0], lamRange_lin < constr[1])
    )[0]
    lamRange_spmod = np.array([lamRange_lin[idx_lam[0]], lamRange_lin[idx_lam[-1]]])
    # Shorten data to size of new lamRange
    ssp_data = ssp_data[idx_lam]

    # Convolve templates to same resolution as data
    if (
        len(
            np.where(
                LSF_Data(lamRange_lin[idx_lam]) - LSF_Templates(lamRange_lin[idx_lam])
                < 0.0
            )[0]
        )
        != 0
    ):
        message = (
            "According to the specified LSF's, the resolution of the "
            + "templates is lower than the resolution of the data. Exit!"
        )
        printStatus.updateFailed("Preparing the stellar population templates")
        print("             " + message)
        logging.critical(message)
        exit(1)
    else:
        FWHM_dif = np.sqrt(
            LSF_Data(lamRange_lin[idx_lam]) ** 2
            - LSF_Templates(lamRange_lin[idx_lam]) ** 2
        )
        sigma = FWHM_dif / 2.355 / cdelt1

    # Create an array to store the templates
    sspNew, _, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale)

    # Do NOT sort the templates in any way

    # Load templates, convolve and log-rebin them
    templates = np.empty((sspNew.size, ntemplates))
    for j, file in enumerate(sp_models):
        #hdu = fits.open(file)
        #ssp_data = np.squeeze(hdu[0].data)[idx_lam]
        dat = np.loadtxt(file)
        ssp_data = dat[:, 1][idx_lam]
        ssp_data = gaussian_filter1d(ssp_data, sigma)
        templates[:, j], logLam_spmod, _ = log_rebin(
            lamRange_spmod, ssp_data, velscale=velscale
        )

    # Normalise templates in such a way to get mass-weighted results
    if config[module_used]["NORM_TEMP"] == "MASS":
        templates = templates / np.mean(templates)

    # Normalise templates in such a way to get light-weighted results
    if config[module_used]["NORM_TEMP"] == "LIGHT":
        for i in range(templates.shape[1]):
            templates[:, i] = templates[:, i] / np.mean(templates[:, i], axis=0)

    printStatus.updateDone("Preparing the stellar population templates")
    logging.info("Prepared the stellar population templates")

    # For alpha get the value from the filename
    alpha_grid = np.array([
        (1 if t[0] == 'p' else -1) * int(t[1:]) / 10
        for t in (s.split('aFe')[1][:3] for s in sp_models)
    ])
    
    # For metallicity get the values from the filename
    smiles_library_filename = os.path.join(config["GENERAL"]["TEMPLATE_DIR"],
                                           config[module_used]["LIBRARY"],
                                           "sMILES_Library_Params.txt")
    df = pd.read_csv(smiles_library_filename, delim_whitespace=True)
    # Build dictionary: MILESID -> [Fe/H]
    feh_lookup = dict(zip(df["MILESID"], df["[Fe/H]"]))
    feh_grid = np.array([
        feh_lookup[s.split('/')[-1].split('_')[0]]
        for s in sp_models
    ])
    metal_grid = feh_grid + 0.66154*alpha_grid + 0.20465*(alpha_grid**2)
    
    # make fake grids
    nAges = 89
    nMetal = 9
    nAlpha = np.unique(alpha_grid).size
    ncomb = nAges * nMetal * nAlpha
    
    #for stellar libraries, replace age with log Teff
    teff_grid = dict(zip(df["MILESID"], df["Teff(K)"]))
    logAge_grid = np.log10(teff_grid)

    if sortInGrid == True:
        return (
            templates,
            [lamRange_spmod[0], lamRange_spmod[1]],
            logLam_spmod,
            ntemplates,
            logAge_grid,
            metal_grid,
            alpha_grid,
            ncomb,
            nAges,
            nMetal,
            nAlpha,
        )
    else:
        return (
            templates,
            [lamRange_spmod[0], lamRange_spmod[1]],
            logLam_spmod,
            ntemplates,
            np.Nan,
            np.Nan,
            np.Nan,
            np.Nan,
            np.Nan,
            np.Nan,
            np.Nan,
        )
