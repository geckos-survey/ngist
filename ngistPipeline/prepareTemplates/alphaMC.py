import glob
import os
import numpy as np
import logging
import astropy.units as u

from astropy.io import fits
from printStatus import printStatus
from ppxf.ppxf_util import log_rebin, varsmooth

def feh_zh_conversion(FeH, alphaFe):

    #[Fe/H], [a/Fe], [Z/H] lists from MJ Park
    
    FeH_arr = [-2.5, -2.5, -2.5, -2.5, -2.5, -2.25, -2.25, -2.25, -2.25, -2.25, -2.0,\
            -2.0, -2.0, -2.0, -2.0, -1.75, -1.75, -1.75, -1.75, -1.75, -1.5, -1.5,\
            -1.5, -1.5, -1.5, -1.25, -1.25, -1.25, -1.25, -1.25, -1.0, -1.0, -1.0,\
            -1.0, -1.0, -0.75, -0.75, -0.75, -0.75, -0.75, -0.5, -0.5, -0.5, -0.5,\
            -0.5, -0.25, -0.25, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25,\
            0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5]

    alphaFe_arr = [-0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, \
                0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, \
                0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, \
                0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, \
                0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6, -0.2, 0.0, 0.2, 0.4, 0.6]

    ZH_arr = [-2.627,-2.500,-2.353,-2.190,-2.016, -2.377, -2.250, -2.103, -1.940, -1.766,\
            -2.127,-2.000,-1.853,-1.690,-1.516, -1.877, -1.750, -1.603, -1.441, \
            -1.266, -1.627,-1.500,-1.353,-1.191,-1.016, -1.377, -1.250, -1.103, \
            -0.941, -0.766, -1.127,-1.000,-0.853,-0.690,-0.516, -0.877, -0.750, \
            -0.603, -0.440, -0.266, -0.627,-0.500,-0.353,-0.190,-0.016, -0.377, \
            -0.250, -0.103, 0.060, 0.234, -0.127,0.000,0.147,0.310,0.484, 0.123, \
            0.250,0.397,0.560,0.734, 0.373,0.500,0.647,0.810,0.987]
    
    # find the best match
    ZH = ZH_arr[np.where((FeH_arr == FeH) & (alphaFe_arr == alphaFe))[0][0]] \
        if np.any((FeH_arr == FeH) & (alphaFe_arr == alphaFe)) else np.nan

    return ZH

def age_metal_alpha(logage_grid, metal_grid, alpha_grid):
    '''
    extract the age and metallicity from the name of file
    modified from miles_util.py of PPXF
    '''
    logAge = np.unique(logage_grid)
    Metal = np.unique(metal_grid)
    Alpha = np.unique(alpha_grid)
    nAges = len(logAge)
    nMetal = len(Metal)
    nAlpha = len(Alpha)
    ncomb = nAges * nMetal * nAlpha
    return logAge, Metal, Alpha, nAges, nMetal, nAlpha, ncomb


def vactoair(vacwl):
    """Calculate the approximate wavelength in air for vacuum wavelengths.

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses an approximate formula from the IDL astronomy library
    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro

    """
    wave2 = vacwl * vacwl
    n = 1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2 * wave2)

    # Do not extrapolate to very short wavelengths.
    if not isinstance(vacwl, np.ndarray):
        if vacwl < 2000:
            n = 1.0
    else:
        ignore = np.where(vacwl < 2000)
        n[ignore] = 1.0

    return vacwl / n


def prepareSpectralTemplateLibrary(config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid):
    """
    Prepares the spectral template library. The templates are loaded from disk,
    shortened to meet the spectral range in consideration, convolved to meet the
    resolution of the observed spectra (according to the LSF), log-rebinned, and
    normalised. In addition, they are sorted in a three-dimensional array
    sampling the parameter space in age, metallicity and alpha-enhancement.
    """
    printStatus.running("Preparing the stellar population templates")
    cvel  = 299792.458

    # SSP model library
    sp_models = glob.glob(os.path.join(config['GENERAL']['TEMPLATE_DIR'],config[module_used]["LIBRARY"]) + \
#                          'ssp_final_mistv2.5_c3kv2.3vt10allfal_250722.fits')
                           'ssp_final_mistv2.5_c3kv2.3vt10allfal_250722_nGIST_geckos.fits')

    ssp_conroy = fits.open(sp_models[0])
    wave = ssp_conroy[0].data
    orig_templates = ssp_conroy[1].data
    grids = ssp_conroy[2].data


    # transfer from vacuum to air wave
    wave_air = vactoair(wave)
    for j in range(len(orig_templates)):
        orig_templates[j, :] = np.interp(wave, wave_air, orig_templates[j, :])

    # Select wavelength between 3000 and 10000 Angstrom
    hires_mask = (wave > 3000) & (wave < 10000)
    wave = wave[hires_mask]
    orig_templates = orig_templates[:, hires_mask]

    # Change the unit of templates to from dHz to dlambda
    orig_templates = orig_templates * cvel * 1e13 / wave ** 2

    ntemplates = len(orig_templates)
    ssp_data = orig_templates[0, :]
    lamRange_spmod = wave[[0, -1]]

    # Determine length of templates
    template_overhead = np.zeros(2)
    if lmin - lamRange_spmod[0] > 150.:
        template_overhead[0] = 150.
    else:
        template_overhead[0] = lmin - lamRange_spmod[0] - 5
    if lamRange_spmod[1] - lmax > 150.:
        template_overhead[1] = 150.
    else:
        template_overhead[1] = lamRange_spmod[1] - lmax - 5

    # Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
    constr = np.array([ lmin - template_overhead[0], lmax + template_overhead[1] ])
    idx_lam = np.where( np.logical_and(wave > constr[0], wave < constr[1] ) )[0]
    lamRange_spmod = np.array([ wave[idx_lam[0]], wave[idx_lam[-1]] ])

    # Shorten data to size of new lamRange
    new_wave = wave[idx_lam]
    ssp_data = ssp_data[idx_lam]

    # Convolve templates to same resolution as data
    if (
        len(
            np.where(
                LSF_Data(new_wave) - LSF_Templates(new_wave)
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
        fwhm_diff2 = (LSF_Data(new_wave) ** 2 - LSF_Templates(new_wave) ** 2).clip(0)  # NB: clip if fwhm_tem too large!
        sigma = np.sqrt(fwhm_diff2)/np.sqrt(4*np.log(4))

    # Create an array to store the templates
    sspNew, _, _ = log_rebin(new_wave, ssp_data, velscale=velscale)

    # Extract ages, metallicities and alpha from the templates
    orig_logage_grid = np.array([x[0] for x in grids]) - 9
    orig_metal_grid = np.array([x[1] for x in grids])
    orig_alpha_grid = np.array([x[2] for x in grids])

    # Do NOT sort the templates in any way
    if sortInGrid == False:
        # Load templates, convolve and log-rebin them
        templates = np.empty((sspNew.size, ntemplates))
        for j in range(ntemplates):
            ssp_data = orig_templates[j, :][idx_lam]
            ssp_data = varsmooth(new_wave, ssp_data, sigma)
            templates[:, j], logLam_spmod, _ = log_rebin(
                new_wave, ssp_data, velscale=velscale
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
            nAges,
            nMetal,
            nAlpha,
            ncomb,
        ) = age_metal_alpha(orig_logage_grid, orig_metal_grid, orig_alpha_grid)

        templates = np.zeros((sspNew.size, nAges, nMetal, nAlpha))
        templates[:, :, :, :] = np.nan

        # Arrays to store properties of the models
        logAge_grid = np.empty((nAges, nMetal, nAlpha))
        metal_grid = np.empty((nAges, nMetal, nAlpha))
        alpha_grid = np.empty((nAges, nMetal, nAlpha))

        # Load the templates
        for j in range(ntemplates):

            idx_j = np.where(logAge == orig_logage_grid[j])[0][0]
            idx_k = np.where(metal == orig_metal_grid[j])[0][0]
            idx_i = np.where(alpha == orig_alpha_grid[j])[0][0]
            ssp = orig_templates[j, :][idx_lam]
            ssp_data = varsmooth(new_wave, ssp_data, sigma)
            sspNew, logLam2, _ = log_rebin(
                new_wave, ssp, velscale=velscale
            )

        
            # convert FeH to ZH - currently not being used
            # metal_out = feh_zh_conversion(orig_metal_grid[j],orig_alpha_grid[j])
            metal_out = orig_metal_grid[j] # default Fe/H

            logAge_grid[idx_j, idx_k, idx_i] = orig_logage_grid[j]
            metal_grid[idx_j, idx_k, idx_i] = metal_out
            alpha_grid[idx_j, idx_k, idx_i] = orig_alpha_grid[j]

            # Normalise templates for light-weighted results
            if config[module_used]["NORM_TEMP"] == "LIGHT":
                templates[:, idx_j, idx_k, idx_i] = sspNew / np.mean(sspNew)
            else:
                templates[:, idx_j, idx_k, idx_i] = sspNew
    
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
