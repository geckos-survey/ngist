import glob
import logging
import os

import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin, varsmooth
from printStatus import printStatus


def age_metal_alpha(passedFiles):
    """
    Function to extract the values of age, metallicity, and alpha-enhancement
    from standard MILES filenames. Note that this function can automatically
    distinguish between template libraries that do or do not include
    alpha-enhancement.
    """

    out = np.zeros((len(passedFiles), 3))
    out[:, :] = np.nan

    files = []
    for i in range(len(passedFiles)):
        files.append(passedFiles[i].split("/")[-1])

    for num, s in enumerate(files):
        # Ages
        age_start = s.find("T")
        age_end = s.find("_M")
        age = float(s[age_start + 1 : age_end])

        # Metals
        metal_start = s.find("MH")
        metal_end = s.find("_Kr")
        metal = float(s[metal_start + 1 : metal_end])

        # Alpha
        # = There is *NO* alpha defined in XSL SSP
        alpha = 0.0

        out[num, :] = age, metal, alpha

    Age = np.unique(out[:, 0])
    Metal = np.unique(out[:, 1])
    Alpha = np.unique(out[:, 2])
    nAges = len(Age)
    nMetal = len(Metal)
    nAlpha = len(Alpha)
    ncomb = nAges * nMetal * nAlpha

    metal_str = []
    alpha_str = []
    for i in range(len(Metal)):
        if Metal[i] > 0:
            mm = "p" + "{:.2f}".format(np.abs(Metal[i])) + "T"
        elif Metal[i] < 0:
            mm = "m" + "{:.2f}".format(np.abs(Metal[i])) + "T"
        metal_str.append(mm)

    alpha_str = ["baseFe"]

    return (
        np.log10(Age),
        Metal,
        Alpha,
        metal_str,
        alpha_str,
        nAges,
        nMetal,
        nAlpha,
        ncomb,
    )


def prepareSpectralTemplateLibrary(
    config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid
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
    ssp_models = glob.glob(
        os.path.join(config["GENERAL"]["TEMPLATE_DIR"], config[module_used]["LIBRARY"])
        + "*.fits"
    )

    ssp_models.sort()
    ntemplates = len(ssp_models)

    # Read data
    hdu_sspmod = fits.open(ssp_models[0])
    ssp_flux = np.squeeze(hdu_sspmod[0].data)
    ssp_head = hdu_sspmod[0].header
    ssp_lam = 10**((np.arange(0,ssp_flux.shape[0])- ssp_head["CRPIX1"]) * ssp_head["CDELT1"] + ssp_head["CRVAL1"])*10
    lamRange_lin = [ssp_lam[0], ssp_lam[-1]]
    
    # Determine length of templates
    template_overhead = np.zeros(2)
    if lmin - lamRange_lin[0] > 150.0:
        template_overhead[0] = 150.0
    else:
        template_overhead[0] = lmin - lamRange_lin[0] - 5
    if lamRange_lin[1] - lmax > 150.0:
        template_overhead[1] = 150.0
    else:
        template_overhead[1] = lamRange_lin[1] - lmax - 5

    # Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
    constr = np.array([lmin - template_overhead[0], lmax + template_overhead[1]])
    idx_lam = np.where(
        np.logical_and(ssp_lam > constr[0], ssp_lam < constr[1])
    )[0]
    
    # Shorten data to size of new lamRange
    ssp_lam_mod = ssp_lam[idx_lam]
    ssp_flux_mod = ssp_flux[idx_lam]

    # Convolve templates to same resolution as data
    if (
        len(
            np.where(
                LSF_Data(ssp_lam[idx_lam]) - LSF_Templates(ssp_lam[idx_lam])
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
        #from cappellari sps_util, use this because the wavelength bin size is variable
        fwhm_diff2 = (LSF_Data(ssp_lam_mod) ** 2 - LSF_Templates(ssp_lam_mod) ** 2).clip(0)  # NB: clip if fwhm_tem too large!
        sigma = np.sqrt(fwhm_diff2)/np.sqrt(4*np.log(4))
        
    # Create an array to store the templates
    sspNew, _ , _ = log_rebin(ssp_lam_mod, ssp_flux_mod, velscale=velscale)

    # Do NOT sort the templates in any way
    if sortInGrid == False:
        # Load templates, convolve and log-rebin them
        templates = np.empty((sspNew.size, ntemplates))
        for j, file in enumerate(ssp_models):
            hdu = fits.open(file)
            ssp_tmp_flux = np.squeeze(hdu[0].data)[idx_lam]
            
            ###########
            # smooth LSF from cappellari sps_util
            flux_ssp_lsf = varsmooth(ssp_lam_mod, ssp_tmp_flux, sigma)
        
            # log Rebin
            templates[:, j], logLam_ssp, _ = log_rebin(
                ssp_lam_mod, flux_ssp_lsf, velscale=velscale)
            

        # Normalise templates in such a way to get mass-weighted results
        if config[module_used]["NORM_TEMP"] == "MASS":
            templates = templates / np.mean(templates)

        # Normalise templates in such a way to get light-weighted results
        if config[module_used]["NORM_TEMP"] == "LIGHT":
            for i in range(templates.shape[1]):
                templates[:, i] = templates[:, i] / np.mean(templates[:, i], axis=0)

        printStatus.updateDone("Preparing the stellar population templates")
        logging.info("Prepared the stellar population templates")

        return (
            templates,
            [ssp_lam_mod[0], ssp_lam_mod[-1]],
            logLam_ssp,
            ntemplates,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # Sort the templates in a cube of age, metal, alpha for the SFH module
    elif sortInGrid == True:
        # Extract ages, metallicities and alpha from the templates
        (
            logAge,
            metal,
            alpha,
            metal_str,
            alpha_str,
            nAges,
            nMetal,
            nAlpha,
            ncomb,
        ) = age_metal_alpha(ssp_models)

        templates = np.zeros((sspNew.size, nAges, nMetal, nAlpha))
        templates[:, :, :, :] = np.nan

        # Arrays to store properties of the models
        logAge_grid = np.empty((nAges, nMetal, nAlpha))
        metal_grid = np.empty((nAges, nMetal, nAlpha))
        alpha_grid = np.empty((nAges, nMetal, nAlpha))

        # Sort the templates in the cube of age, metal, alpha
        # This sorts for alpha
        for i, a in enumerate(alpha_str):
            # This sorts for metals
            for k, mh in enumerate(metal_str):
                files = [s for s in ssp_models if (mh in s and a in s)]
                # This sorts for ages
                for j, filename in enumerate(files):
                    hdu = fits.open(filename)
                    ssp_tmp_flux = np.squeeze(hdu[0].data)[idx_lam]
                    ###########
                    # smooth LSF from cappellari sps_util
                    flux_ssp_lsf = varsmooth(ssp_lam_mod, ssp_tmp_flux, sigma)
        
                      # log Rebin
                    sspNew, logLam2, _ = log_rebin(
                    ssp_lam_mod, flux_ssp_lsf, velscale=velscale)
                    
                    logAge_grid[j, k, i] = logAge[j]
                    metal_grid[j, k, i] = metal[k]
                    alpha_grid[j, k, i] = alpha[i]

                    # Normalise templates for light-weighted results
                    if config[module_used]["NORM_TEMP"] == "LIGHT":
                        templates[:, j, k, i] = sspNew / np.mean(sspNew)
                    else:
                        templates[:, j, k, i] = sspNew

        # Normalise templates for mass-weighted results
        if config[module_used]["NORM_TEMP"] == "MASS":
            templates = templates / np.mean(templates)

        printStatus.updateDone("Preparing the stellar population templates")
        logging.info("Prepared the stellar population templates")

        return (
            templates,
            [ssp_lam_mod[0], ssp_lam_mod[-1]],
            logLam2,
            ntemplates,
            logAge_grid,
            metal_grid,
            alpha_grid,
            ncomb,
            nAges,
            nMetal,
            nAlpha,
        )
