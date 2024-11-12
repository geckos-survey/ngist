import glob
import logging
import os

import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import gaussian_filter1d, log_rebin
from printStatus import printStatus


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
    sp_models = glob.glob(
        os.path.join(config["GENERAL"]["TEMPLATE_DIR"], config[module_used]["LIBRARY"])
        + "*.fits"
    )

    sp_models.sort()
    ntemplates = len(sp_models)

    # Read data
    hdu_spmod = fits.open(sp_models[0])
    ssp_data = np.squeeze(hdu_spmod[0].data)
    ssp_head = hdu_spmod[0].header
    lamRange_spmod = ssp_head["CRVAL1"] + np.array(
        [0.0, ssp_head["CDELT1"] * (ssp_head["NAXIS1"] - 1)]
    )

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
    # Reconstruct full original lamRange
    lamRange_lin = np.arange(
        lamRange_spmod[0], lamRange_spmod[-1] + ssp_head["CDELT1"], ssp_head["CDELT1"]
    )
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
        sigma = FWHM_dif / 2.355 / ssp_head["CDELT1"]

    # Create an array to store the templates
    sspNew, _, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale)

    # Do NOT sort the templates in any way
    if (
        sortInGrid == False
    ):  # Currently this is the only option for IndoUS - only for use with stellar kinematics
        # Load templates, convolve and log-rebin them
        templates = np.empty((sspNew.size, ntemplates))
        for j, file in enumerate(sp_models):
            hdu = fits.open(file)
            ssp_data = np.squeeze(hdu[0].data)[idx_lam]
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

        return (
            templates,
            [lamRange_spmod[0], lamRange_spmod[1]],
            logLam_spmod,
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
        ) = age_metal_alpha(sp_models)

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
                files = [s for s in sp_models if (mh in s and a in s)]
                # This sorts for ages
                for j, filename in enumerate(files):
                    hdu = fits.open(filename)
                    ssp = np.squeeze(hdu[0].data)[idx_lam]
                    ssp = gaussian_filter1d(ssp, sigma)
                    sspNew, logLam2, _ = log_rebin(
                        lamRange_spmod, ssp, velscale=velscale
                    )

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
            [lamRange_spmod[0], lamRange_spmod[1]],
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
