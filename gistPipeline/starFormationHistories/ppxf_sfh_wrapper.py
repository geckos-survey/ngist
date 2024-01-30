import glob
import logging
import os
import time

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from multiprocess import Process, Queue
# Then use system installed version instead
from ppxf.ppxf import ppxf
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.prepareTemplates import _prepareTemplates

from concurrent.futures import ThreadPoolExecutor, as_completed

# Physical constants
C = 299792.458  # speed of light in km/s


"""
PURPOSE:
  This module performs the extraction of non-parametric star-formation histories
  by full-spectral fitting.  Basically, it acts as an interface between pipeline
  and the pPXF routine from Cappellari & Emsellem 2004
  (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""

def robust_sigma(y, zero=False):
     """
     Biweight estimate of the scale (standard deviation).
     Implements the approach described in
     "Understanding Robust and Exploratory Data Analysis"
     Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417
     Added for sigma-clipping method
     """
     y = np.ravel(y)
     d = y if zero else y - np.median(y)

     mad = np.median(np.abs(d))
     u2 = (d/(9.0*mad))**2  # c = 9
     good = u2 < 1.0
     u1 = 1.0 - u2[good]
     num = y.size * ((d[good]*u1**2)**2).sum()
     den = (u1*(1.0 - 5.0*u2[good])).sum()
     sigma = np.sqrt(num/(den*(den - 1.0)))  # see note in above reference

     return sigma

def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """

    for (
        templates,
        galaxy,
        noise,
        velscale,
        start,
        goodPixels_sfh,
        mom,
        offset,
        degree,
        mdeg,
        regul_err,
        doclean,
        fixed,
        velscale_ratio,
        npix,
        ncomb,
        nbins,
        i,
        optimal_template_in,
    ) in iter(inQueue.get,'STOP'):
        (
            sol,
            w_row,
            bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
        ) = run_ppxf(templates,
            galaxy,
            noise,
            velscale,
            start,
            goodPixels_sfh,
            mom,
            offset,
            degree,
            mdeg,
            regul_err,
            doclean,
            fixed,
            velscale_ratio,
            npix,
            ncomb,
            nbins,
            i,
            optimal_template_in,
        )

        outQueue.put(
            (
                i,
                sol,
                w_row,
                bestfit,
                optimal_template,
                mc_results,
                formal_error,
                spectral_mask,
                snr_postfit,
            )
        )

def run_ppxf_firsttime(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul_err,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb,
    nbins,
    i,
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
        # Call PPXF for first time to get optimal template
    #print("Running pPXF for the first time")
    #logging.info("Using the new 3-step pPXF implementation")
    pp = ppxf(
        templates,
        log_bin_data,
        log_bin_error,
        velscale,
        start,
        goodpixels=goodPixels,
        plot=False,
        quiet=True,
        moments=nmoments,
        degree=-1,
        vsyst=offset,
        mdegree=mdeg,
        regul = 1./regul_err,
        fixed=fixed,
        velscale_ratio=velscale_ratio,
    )

    normalized_weights = pp.weights / np.sum( pp.weights )
    optimal_template   = np.zeros( templates.shape[0] )
    for j in range(0, templates.shape[1]):
        optimal_template = optimal_template + templates[:,j]*normalized_weights[j]

    return optimal_template

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul_err,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb,
    nbins,
    i,
    optimal_template_in,
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    printStatus.progressBar(i, nbins, barLength=50)

    try:

        if len(optimal_template_in) > 1:
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise=np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                fixed=fixed,
                velscale_ratio=velscale_ratio,
            )
            # Find a proper estimate of the noise
            #noise_orig = biweight_location(log_bin_error[goodPixels])
            #goodpixels is one shorter than log_bin_error
            noise_orig = np.mean(log_bin_error[goodPixels])
            noise_est = robust_sigma(pp_step1.galaxy[goodPixels]-pp_step1.bestfit[goodPixels])

            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error*(noise_est/noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

            ################ 2 ##################
            # Second Call PPXF - use best-fitting template, determine outliers
            if doclean == True:
                pp_step2 = ppxf(
                    optimal_template_in,
                    log_bin_data,
                    noise_new,
                    velscale,
                    start,
                    goodpixels=goodPixels,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    degree=-1,
                    vsyst=offset,
                    mdegree=mdeg,
                    fixed=fixed,
                    velscale_ratio=velscale_ratio,
                    clean=True,
                )

                # update goodpixels
                goodPixels = pp_step2.goodpixels

                # repeat noise scaling # Find a proper estimate of the noise
                noise_orig = biweight_location(log_bin_error[goodPixels])
                noise_est = robust_sigma(pp_step2.galaxy[goodPixels]-pp_step2.bestfit[goodPixels])

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error*(noise_est/noise_orig)
                noise_new_std = robust_sigma(noise_new)

                # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
                noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

            ################ 3 ##################
            # Third Call PPXF - use all templates, get best-fit
            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                regul = 1./regul_err,
                fixed=fixed,
                velscale_ratio=velscale_ratio,
            )

        #update goodpixels again
        goodPixels = pp.goodpixels

        #make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # Calculate the true S/N from the residual
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmean(pp.galaxy[goodPixels]/noise_est)

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
        w_row   = np.reshape(weights, ncomb)

        # # Do MC-Simulations - Amelia - this is not currently implemented. Add back in later.
        # sol_MC     = np.zeros((nsims,nmoments))
        mc_results = np.zeros(nmoments)
        #
        # for o in range(0, nsims):
        #     # Add noise to bestfit:
        #     #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
        #     #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
        #     noisy_bestfit = pp.bestfit  +  np.random.normal(0, 1, len(log_bin_data)) * np.std( log_bin_data[goodPixels] - pp.bestfit[goodPixels] )
        #
        #     mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
        #             quiet=True, moments=nmoments, degree=-1, mdegree=mdeg, velscale_ratio=velscale_ratio, vsyst=offset, bias=0.0)
        #     sol_MC[o,:] = mc.sol[:]
        #
        # if nsims != 0:
        #     mc_results = np.nanstd( sol_MC, axis=0 )
        # print(pp.sol[:])

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
        )

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)



# ## *****************************************************************************
# #        noise_i = noise_i * np.sqrt(  / len(goodPixels) )
# #        regul_err =
#
#         pp = ppxf(templates, galaxy_i, noise_i, velscale, start, goodpixels=goodPixels, plot=False, quiet=True,\
#               moments=nmom, degree=-1, vsyst=dv, mdegree=mdeg, regul=1./regul_err, fixed=fixed, velscale_ratio=velscale_ratio)
#
# #        if i == 0:
# #            print()
# #            print( i, pp.chi2 )
# #            print( len( goodPixels ) )
# #            print( np.sqrt(2 * len(goodPixels)) )
# #            print()
#
#         weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
#         w_row   = np.reshape(weights, ncomb)
#
#         # Correct the formal errors assuming that the fit is good
#         formal_error = pp.error * np.sqrt(pp.chi2)
#
#         return(pp.sol, w_row, pp.bestfit, formal_error)
#
#     except:
#         return(np.nan, np.nan, np.nan, np.nan)



def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin.
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan

    for i in range( nbins ):
        mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
        mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])

    return(mean)


def save_sfh(
    mean_result,
    ppxf_result,
    w_row,
    mc_results,
    formal_error,
    logAge_grid,
    metal_grid,
    alpha_grid,
    ppxf_bestfit,
    logLam,
    goodPixels,
    velscale,
    logLam1,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
    npix,
    config,
    spectral_mask,
    optimal_template_comb,
    snr_postfit,
):
    """ Save all results to disk. """

    # ========================
    # SAVE KINEMATICS
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with stellar kinematics
    cols = []
    cols.append(fits.Column(name="AGE", format="D", array=mean_result[:, 0]))
    cols.append(fits.Column(name="METAL", format="D", array=mean_result[:, 1]))
    cols.append(fits.Column(name="ALPHA", format="D", array=mean_result[:, 2]))

    if config["SFH"]["FIXED"] == False:
        cols.append(fits.Column(name="V", format="D", array=kin[:, 0]))
        cols.append(fits.Column(name="SIGMA", format="D", array=kin[:, 1]))
        if np.any(kin[:, 2]) != 0:
            cols.append(fits.Column(name="H3", format="D", array=kin[:, 2]))
        if np.any(kin[:, 3]) != 0:
            cols.append(fits.Column(name="H4", format="D", array=kin[:, 3]))
        if np.any(kin[:, 4]) != 0:
            cols.append(fits.Column(name="H5", format="D", array=kin[:, 4]))
        if np.any(kin[:, 5]) != 0:
            cols.append(fits.Column(name="H6", format="D", array=kin[:, 5]))

        cols.append(
            fits.Column(name="FORM_ERR_V", format="D", array=formal_error[:, 0])
        )
        cols.append(
            fits.Column(name="FORM_ERR_SIGMA", format="D", array=formal_error[:, 1])
        )
        if np.any(formal_error[:, 2]) != 0:
            cols.append(
                fits.Column(name="FORM_ERR_H3", format="D", array=formal_error[:, 2])
            )
        if np.any(formal_error[:, 3]) != 0:
            cols.append(
                fits.Column(name="FORM_ERR_H4", format="D", array=formal_error[:, 3])
            )
        if np.any(formal_error[:, 4]) != 0:
            cols.append(
                fits.Column(name="FORM_ERR_H5", format="D", array=formal_error[:, 4])
            )
        if np.any(formal_error[:, 5]) != 0:
            cols.append(
                fits.Column(name="FORM_ERR_H6", format="D", array=formal_error[:, 5])
            )

    # Add True SNR calculated from residual
    cols.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))


    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "SFH"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE WEIGHTS AND GRID
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh-weights.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols = []
    cols.append(
        fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "WEIGHTS"

    logAge_row = np.reshape(logAge_grid, ncomb)
    metal_row = np.reshape(metal_grid, ncomb)
    alpha_row = np.reshape(alpha_grid, ncomb)

    # Table HDU with grids
    cols = []
    cols.append(fits.Column(name="LOGAGE", format="D", array=logAge_row))
    cols.append(fits.Column(name="METAL", format="D", array=metal_row))
    cols.append(fits.Column(name="ALPHA", format="D", array=alpha_row))
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gridHDU.name = "GRID"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    gridHDU = _auxiliary.saveConfigToHeader(gridHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, gridHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "NAGES", value=nAges)
    fits.setval(outfits_sfh, "NMETAL", value=nMetal)
    fits.setval(outfits_sfh, "NALPHA", value=nAlpha)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits"
    )
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE BESTFIT
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh-bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-bestfit.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with SFH bestfit
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with SFH logLam
    cols = []

    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))

    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with SFH goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["SFH"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1", value=1.0)
    fits.setval(outfits_sfh, "CRVAL1", value=logLam1[0])
    fits.setval(outfits_sfh, "CDELT1", value=logLam1[1] - logLam1[0])

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-bestfit.fits"
    )
    logging.info("Wrote: " + outfits_sfh)


def extractStarFormationHistories(config):
    """
    Starts the computation of non-parametric star-formation histories with
    pPXF.  A spectral template library sorted in a three-dimensional grid of
    age, metallicity, and alpha-enhancement is loaded.  Emission-subtracted
    spectra are used for the fit. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called.
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    # Prepare template library
    velscale = fits.open(
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_BinSpectra.fits"
    )[0].header["VELSCALE"]
    velscale_ratio = 2

    (
        templates,
        lamRange_temp,
        logLam_template,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config['SFH']['LMIN'],
        config['SFH']['LMAX'],
        velscale/velscale_ratio,
        LSF_Data,
        LSF_Templates,
        'SFH',
        sortInGrid=True,
    )
    templates = templates.reshape( (templates.shape[0], ntemplates) )


    # Read spectra
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + '_gas-cleaned_'+config['GAS']['LEVEL']+'.fits'
        )
        == True
    ):
        logging.info(
            "Using emission-subtracted spectra at "
            + os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + '_gas-cleaned_'+config['GAS']['LEVEL']+'.fits'
        )
        printStatus.done("Using emission-subtracted spectra")

        hdu = fits.open(
            os.path.join(config['GENERAL']['OUTPUT'],
            config['GENERAL']['RUN_ID'])+'_gas-cleaned_'+config['GAS']['LEVEL']+'.fits'
        )
        # Adding a bit in to also load the BinSpectra.fits to grab the error spectrum, even if using the cleaned gas specrum
        # But sometimes this isn't always the right shape. So really, you want the error saved to the _gas_cleaned_BIN.fits hdu
        #hdu2 = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')

    else:
        logging.info(
            "Using regular spectra without any emission-correction at "
            + os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_BinSpectra.fits"
        )
        printStatus.done("Using regular spectra without any emission-correction")
        hdu = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_BinSpectra.fits"
        )

    galaxy = np.array( hdu[1].data.SPEC )
    logLam = hdu[2].data.LOGLAM
    idx_lam = np.where( np.logical_and( np.exp(logLam) > config['SFH']['LMIN'], np.exp(logLam) < config['SFH']['LMAX'] ) )[0]
    galaxy = galaxy[:,idx_lam]
    #galaxy = galaxy/np.median(galaxy) # Amelia added to normalise normalize flux. Do we use this again?
    logLam = logLam[idx_lam]
    nbins = galaxy.shape[0]
    npix = galaxy.shape[1]
    ubins = np.arange(0, nbins)
    noise = np.full(npix, config['SFH']['NOISE'])
    dv = (np.log(lamRange_temp[0]) - logLam[0])*C
    #bin_err = np.array( hdu2[1].data.ESPEC.T ) #This will almost certainly not work, as galaxy array isn't transposed
    bin_err = np.array( hdu[1].data.ESPEC.T ) #This will almost certainly not work, as galaxy array isn't transposed. Does this still need to be transposed?
    bin_data = np.array( hdu[1].data.SPEC.T ) # Amelia this doens't bode well
    bin_data = bin_data[idx_lam,:]
    bin_err = bin_err[idx_lam,:]
    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    #noise = np.ones((npix,nbins))
    noise = bin_err # is actual noise, not variance
    nsims = config['SFH']['MC_PPXF']

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["SFH"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        # Set fixed option to True
        fixed = [True] * config["KIN"]["MOM"]

        # Read PPXF results
        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits"
        )[1].data
        start = np.zeros((nbins, config["KIN"]["MOM"]))
        for i in range(nbins):
            start[i, :] = np.array(ppxf_data[i][: config["KIN"]["MOM"]])

    # Do *NOT* fix kinematics to those obtained previously
    elif config["SFH"]["FIXED"] == False:
        logging.info(
            "Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties."
        )
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, 2))
        for i in range(nbins):
            start[i, :] = np.array([0.0, config["KIN"]["SIGMA"]])

    # Define goodpixels
    goodPixels_sfh = _auxiliary.spectralMasking(config, config['SFH']['SPEC_MASK'], logLam)

    # Define output arrays
    ppxf_result = np.zeros((nbins,6    ))
    w_row = np.zeros((nbins,ncomb))
    ppxf_bestfit = np.zeros((nbins,npix))
    optimal_template = np.zeros((nbins,templates.shape[0]))
    mc_results = np.zeros((nbins,6))
    formal_error = np.zeros((nbins,6))
    spectral_mask = np.zeros((nbins,bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:,:],axis=1)
    comb_espec = np.nanmean(bin_err[:,:],axis=1)
    #comb_spec = comb_spec/np.nanmedian(comb_spec) # Amelia added to mormalise normalize spectrum
    #comb_espec = comb_espec/np.nanmedian(comb_espec) # and the error spectrum
    optimal_template_init = [0]

    optimal_template_out = run_ppxf_firsttime(
        templates,
        comb_spec ,
        comb_espec,
        velscale,
        start[0,:],
        goodPixels_sfh,
        config['SFH']['MOM'],
        offset,-1,
        config['SFH']['MDEG'],
        config['SFH']['REGUL_ERR'],
        config["SFH"]["DOCLEAN"],
        fixed,
        velscale_ratio,
        ncomb,
        nsims,
        nbins,
        0,
        optimal_template_init,
    )

    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

    # ====================


    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        
        # Define run_ppxf_wrapper function to be run in parallel
        def worker(i):
            return run_ppxf(
                templates,
                    bin_data[:,i],
                    noise[:,i],
                    velscale,
                    start[i,:],
                    goodPixels_sfh,
                    config['SFH']['MOM'],
                    offset,
                    -1,
                    config['SFH']['MDEG'],
                    config['SFH']['REGUL_ERR'],
                    config["SFH"]["DOCLEAN"],
                    fixed,
                    velscale_ratio,
                    npix,
                    ncomb,
                    nbins,
                    i,
                    optimal_template_comb,
            )

        # Create a pool of threads
        with ThreadPoolExecutor(max_workers=min(32, config["GENERAL"]["NCPU"]+4)) as executor:

            # Use a list comprehension to create a list of Future objects
            futures = [executor.submit(worker, i) for i in range(nbins)]

            # Iterate over the futures as they complete
            for future in as_completed(futures):
                # Get the result from the future
                result = future.result()

                # Get the index of the future in the list
                i = futures.index(future)

                # Assign the results to the arrays
                (ppxf_result[i,:config['SFH']['MOM']],
                w_row[i,:],
                ppxf_bestfit[i,:],
                optimal_template[i,:],
                mc_results[i,:config['SFH']['MOM']],
                formal_error[i,:config['SFH']['MOM']],
                spectral_mask[i,:],
                snr_postfit[i],) = result

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False: # Amelia you haven't tested this yet. Come back to.
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            (
                kin[i,:config['SFH']['MOM']],
                w_row[i,:],
                bestfit[i,:],
                formal_error[i,:config['SFH']['MOM']],
                snr_postfit[i],
            ) = run_ppxf(
                templates,
                galaxy[i,:],
                noise,
                velscale,
                start[i,:],
                goodPixels_sfh,
                config['SFH']['MOM'],
                dv,
                -1,
                config['SFH']['MDEG'],
                config['SFH']['REGUL_ERR'],
                config["KIN"]["DOCLEAN"],
                fixed,
                velscale_ratio,
                npix,
                ncomb,
                nbins,
                i,
                optimal_template_init,
            )
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print(
        "             Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]

    if len(idx_error) != 0:
        printStatus.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
        )
        print("             " + str(idx_error))
        logging.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
            + str(idx_error)
        )
    else:
        print("             " + "There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Calculate mean age, metallicity and alpha
    mean_results = mean_agemetalalpha(
        w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins
    )

    # Save to file

    save_sfh(
        mean_results,
        ppxf_result,
        w_row,
        mc_results,
        formal_error,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ppxf_bestfit,
        logLam,
        goodPixels_sfh,
        velscale,
        logLam,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
        npix,
        config,
        spectral_mask,
        optimal_template_comb,
        snr_postfit,
    )

    # Return
    return None
