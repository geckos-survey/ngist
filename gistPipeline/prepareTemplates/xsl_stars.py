import glob
import logging
import os

import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin, varsmooth
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

    # XSL spectral library
    # only select the merged and scl files
    xsl_stars = glob.glob(
        os.path.join(config["GENERAL"]["TEMPLATE_DIR"], config[module_used]["LIBRARY"])
        + "*_merged.fits") + glob.glob(
        os.path.join(config["GENERAL"]["TEMPLATE_DIR"], config[module_used]["LIBRARY"])
        + "*_scl.fits")
    
    ntemplates = len(xsl_stars)

    # Read data
    hdu_xsl = fits.open(xsl_stars[0])
    #read flux 
    flux = hdu_xsl[1].data['FLUX_DR']
    #read wavelength
    lam = hdu_xsl[1].data['WAVE'] * 10 # to convert to Angstrom

    lamRange_lin = np.array([lam[0], lam[-1]])

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
        np.logical_and(lam > constr[0], lam < constr[1])
    )[0]
    
    # Shorten data to size of new lamRange
    lam_mod = lam[idx_lam]
    flux_mod = flux[idx_lam]

    # Convolve templates to same resolution as data
    if (
        len(
            np.where(
                LSF_Data(lam[idx_lam]) - LSF_Templates(lam[idx_lam])
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
        fwhm_diff2 = (LSF_Data(lam_mod) ** 2 - LSF_Templates(lam_mod) ** 2).clip(0)  # NB: clip if fwhm_tem too large!
        sigma = np.sqrt(fwhm_diff2)/np.sqrt(4*np.log(4))
        
    # Create an array to store the templates
    sspNew, _ , _ = log_rebin(lam_mod, flux_mod, velscale=velscale)


    # Load templates, convolve and log-rebin them
    templates = np.empty((sspNew.size, ntemplates))
    ngtemplates = 0
    for j, file in enumerate(xsl_stars):
        hdu = fits.open(file)

        ###########

        # Define the columns based on file name
        keyword1 = '_ncl.'
        keyword2 = '_ncge.'
        keyword3 = '_scl.'
    
        if keyword1 in file:
            flux_tmp = hdu[1].data['FLUX_DR']
        elif keyword2 in file:
            flux_tmp = hdu[1].data['FLUX']
        elif keyword3 in file:
            flux_tmp = hdu[1].data['FLUX_SC']
        else:
            flux_tmp = hdu[1].data['FLUX_DR']
        
        lam_tmp = hdu[1].data['WAVE'] * 10 

        ###########
        # cut wavelength and flux spectrum
        lam_tmp = lam_tmp[idx_lam]
        flux_tmp = flux_tmp[idx_lam]
        ###########

        # only select templates without NaNs
        if (len(np.where(np.isfinite(flux_tmp) == 0)[0])) == 0:
            
            ###########
            # smooth LSF from cappellari sps_util
            flux_lsf = varsmooth(lam_tmp, flux_tmp, sigma)
        
            # log Rebin
            templates[:, ngtemplates], logLam_flux, _ = log_rebin(
                lam_tmp, flux_lsf, velscale=velscale)
            ngtemplates += 1

    # reshape templates to remove the empty rows at the end due to bad templates    
    new_templates = templates[:,0:ngtemplates]

    # Normalise templates in such a way to get mass-weighted results
    if config[module_used]["NORM_TEMP"] == "MASS":
        new_templates = new_templates / np.nanmean(new_templates)

    # Normalise templates in such a way to get light-weighted results
    if config[module_used]["NORM_TEMP"] == "LIGHT":
        for i in range(new_templates.shape[1]):
            new_templates[:, i] = new_templates[:, i] / np.nanmean(new_templates[:, i], axis=0)

    printStatus.updateDone("Preparing the stellar population templates")
    logging.info("Prepared the stellar population templates")

    return (
        new_templates,
        [lam_mod[0], lam_mod[-1]],
        logLam_flux,
        ngtemplates,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
