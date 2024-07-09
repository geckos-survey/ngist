import glob
import logging
import os
import time
import extinction

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from multiprocess import Process, Queue
from packaging import version
# Then use system installed version instead
from ppxf.ppxf import ppxf
import ppxf as ppxf_package
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.prepareTemplates import _prepareTemplates

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
        EBV_init,
        logLam,
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
            EBV,
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
            EBV_init,
            logLam,
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
                EBV,
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
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    # Call PPXF for first time to get optimal template
    print("Running pPXF for the first time")
    # normalise galaxy spectra and noise
    median_log_bin_data = np.nanmedian(log_bin_data)
    log_bin_error /= median_log_bin_data
    log_bin_data /= median_log_bin_data

    pp = ppxf(
        templates,
        log_bin_data,
        log_bin_error,
        velscale,
        start,
        goodpixels=goodPixels,
        plot=False,
        quiet=False,
        moments=nmoments,
        degree=-1,
        vsyst=offset,
        mdegree=mdeg,
        regul = 1./regul_err,
        fixed=fixed,
        velscale_ratio=velscale_ratio,
    )

    # Templates shape is currently [Wavelength, nAge, nMet, nAlpha]. Reshape to [Wavelength, ncomb] to create optimal template
    reshaped_templates = templates.reshape((templates.shape[0], ncomb))
    normalized_weights = pp.weights / np.sum( pp.weights )
    optimal_template   = np.zeros( reshaped_templates.shape[0] )
    for j in range(0, reshaped_templates.shape[1]):
        optimal_template = optimal_template + reshaped_templates[:,j]*normalized_weights[j]

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
    EBV_init,
    logLam,
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

            # Normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error /= median_log_bin_data
            log_bin_data /= median_log_bin_data

            # Here add in the extra, 0th step to estimate the dust and print out the E(B-V) map
            # Call PPXF, using an extinction law, no polynomials.
            pp_step0 = ppxf(templates, log_bin_data, log_bin_error, velscale, lam=np.exp(logLam), goodpixels=goodPixels,
                      degree=-1, mdegree=-1, vsyst=offset, velscale_ratio=velscale_ratio,
                      moments=nmoments, start=start, plot=False, reddening=EBV_init,
                      regul=0, quiet=True, fixed=fixed)

            # Take care about the version of ppxf used.
            # For ppxf versions > 8.2.1, pp.reddening = Av,
            # For ppxf versions < 8.2.1, pp.reddening = E(B-V)

            Rv = 4.05
            if version.parse(ppxf_package.__version__) >= version.parse('8.2.1'):
                Av = pp_step0.reddening
                EBV = Av/Rv
            else:
                EBV = pp_step0.reddening
                Av =  EBV * Rv

                # The following is for if we decide  we want to extinction-correct the spectra in the future.
                # Uses a config['SFH']['DUST_CORR'] = True keyword added to the MasterConfig.yaml file
                # log_bin_data1 = extinction.remove(extinction.calzetti00(np.exp(logLam), Av, Rv), log_bin_data)/np.median(log_bin_data)
                # log_bin_data = (log_bin_data1/np.median(log_bin_data1))*np.median(log_bin_data)
                # log_bin_error1 = extinction.remove(extinction.calzetti00(np.exp(logLam), Av, Rv), log_bin_error)/np.median(log_bin_error)
                # log_bin_error = (log_bin_error1/np.median(log_bin_error1))*np.median(log_bin_error)
                # ext_curve = extinction.apply(extinction.calzetti00(np.exp(logLam), Av, Rv), np.ones_like(log_bin_data))
            # # If dust_corr key is False
            # else:
            #     EBV = 0

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
        reshaped_templates = templates.reshape((templates.shape[0], ncomb)) #
        normalized_weights = pp.weights / np.sum( pp.weights ) #
        optimal_template   = np.zeros( reshaped_templates.shape[0] )

        for j in range(0, reshaped_templates.shape[1]):
            optimal_template = optimal_template + reshaped_templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)
        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum() # Take from 1D list to nD array (nAges, nMet, nAlpha)
        w_row   = np.reshape(weights, ncomb)

        # # Do MC-Simulations - this is not currently implemented. Add back in later.
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

        # add normalisation factor back in main results
        pp.bestfit *= median_log_bin_data

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            EBV,
        )

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)



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



def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins, tag):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin.
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan
    if tag == 'all':
        for i in range( nbins ):
            mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
            mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
            mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])
    elif tag =='young':
        for i in range(nbins):
            a = np.array(w_row[i])
            a[216::] = 0 # I've tried to set all weights above the age limit to zero here, so they shouldn't contribute to the metallicity.
            # Here, in MILES_safe_github, logAge[216] = 0.3 = 2 Gyr. I've called everything younger than this young and everything greater old.
            mean[i,0] = np.sum(a * ageGrid.ravel())   / np.sum(a) # This is the mean age, it will be somewhat meaningless.
            mean[i,1] = np.sum(a * metalGrid.ravel()) / np.sum(a) # mean met. This should be the mean met of young stars only. 
            mean[i,2] = np.sum(a * alphaGrid.ravel()) / np.sum(a) # this is just alpha. Not in use. 
    elif tag =='old':
        for i in range(nbins):
            a = np.array(w_row[i])
            a[0:216] = 0 # I've tried to set all weights below the age limit to zero here, so they shouldn't contribute to the metallicity.
            # Here, in MILES_safe_github, logAge[216] = 0.3 = 2 Gyr. I've called everything younger than this young and everything greater old.
            mean[i,0] = np.sum(a * ageGrid.ravel())   / np.sum(a) # This is the mean age, it will be somewhat meaningless.
            mean[i,1] = np.sum(a * metalGrid.ravel()) / np.sum(a) # mean met. This should be the mean met of old stars only. 
            mean[i,2] = np.sum(a * alphaGrid.ravel()) / np.sum(a) # this is just alpha. Not in use.             
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
    EBV,
    tag,
):
    """ Save all results to disk. """

    # ========================
    # SAVE KINEMATICS
    if tag =='ALL':
        outfits_sfh = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_sfh.fits"
        )
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    elif tag =='YOUNG':
        outfits_sfh = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_sfh_young.fits"
        )
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_young.fits")  

    elif tag =='OLD':
        outfits_sfh = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_sfh_old.fits"
        )
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_old.fits")   

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

    # Add E(B-V) derived from pPXF 0th step with reddening but no polynomials
    cols.append(fits.Column(name="EBV", format="D", array=EBV[:]))

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
    if tag =='ALL':
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

    # Read spectra
    if (
        (config['SFH']['SPEC_EMICLEAN'] == True)
        and
        (os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + '_gas-cleaned_'+config['GAS']['LEVEL']+'.fits') == True)
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
    logLam = logLam[idx_lam]
    nbins = galaxy.shape[0]
    npix = galaxy.shape[1]
    ubins = np.arange(0, nbins)
    dv = (np.log(lamRange_temp[0]) - logLam[0])*C
    #bin_err = np.array( hdu2[1].data.ESPEC.T )
    bin_err = np.array( hdu[1].data.ESPEC.T )
    bin_data = np.array( hdu[1].data.SPEC.T )
    bin_data = bin_data[idx_lam,:]
    bin_err = bin_err[idx_lam,:]
    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    #noise = np.full(npix, config['SFH']['NOISE'])
    #noise = np.ones((npix,nbins))
    noise = bin_err # is actual noise, not variance
    nsims = config['SFH']['MC_PPXF']

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["SFH"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        
        #check if moments KIN == SFH
        if config["SFH"]["MOM"] != config["KIN"]["MOM"]:
            printStatus.running("Moments not the same in KIN and SFH module")
            printStatus.running("Ignoring SFH MOMENTS, using KIN MOMENTS")
        
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
    EBV = np.zeros(nbins)

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:,:],axis=1)
    comb_espec = np.nanmean(bin_err[:,:],axis=1)
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
        npix,
        ncomb,
        nbins,
        optimal_template_init,
    )

    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

    # ====================
    EBV_init = 0.1 # PHANGS value initial guess

    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        # Create Queues
        inQueue = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [
            Process(target=workerPPXF, args=(inQueue, outQueue))
            for _ in range(config["GENERAL"]["NCPU"])
        ]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue
        for i in range(nbins):
            inQueue.put(
                (
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
                    EBV_init,
                    logLam,
                )
            )


        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]

        # send stop signal to stop iteration
        for _ in range(config["GENERAL"]["NCPU"]):
            inQueue.put("STOP")

        # stop processes
        for p in ps:
            p.join()

        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):

            index[i] = ppxf_tmp[i][0]
            ppxf_result[i,:config['SFH']['MOM']] = ppxf_tmp[i][1]
            w_row[i,:] = ppxf_tmp[i][2]
            #Here we are un-dereddening the bestfit spectra becuase it looks nicer in Mapviewer. If you want the dereddened spectra, then do the opposite of this
            # Rv = 4.05
            # if config['SFH']['DUST_CORR'] == 'True': # If you've added the dust correction, unapply it to make the mapviewer output look more normal
            #     ppxf_bestfit[i,:] = extinction.apply(extinction.calzetti00(np.exp(logLam), ppxf_tmp[i][9], Rv), ppxf_tmp[i][3])#  * (1/(ppxf_tmp[i][3]/bin_data[:,i])) #/np.median(bin_data[:,i])#/np.median(ppxf_tmp[i][3]) OR np.log(bin_data[:,i])??
            # #log_bin_data = (log_bin_data1/np.median(log_bin_data1))*np.median(log_bin_data) # Don't know if I need this line?
            # else:
            ppxf_bestfit[i,:] = ppxf_tmp[i][3]

            optimal_template[i,:] = ppxf_tmp[i][4]
            mc_results[i,:config['SFH']['MOM']] = ppxf_tmp[i][5]
            formal_error[i,:config['SFH']['MOM']] = ppxf_tmp[i][6]
            spectral_mask[i,:] = ppxf_tmp[i][7]
            snr_postfit[i] = ppxf_tmp[i][8]
            EBV[i] = ppxf_tmp[i][9]

        # Sort output
        argidx = np.argsort( index )
        ppxf_result = ppxf_result[argidx,:]
        w_row = w_row[argidx,:]
        ppxf_bestfit = ppxf_bestfit[argidx,:]
        optimal_template = optimal_template[argidx,:]
        mc_results = mc_results[argidx,:]
        formal_error = formal_error[argidx,:]
        spectral_mask = spectral_mask[argidx,:]
        snr_postfit = snr_postfit[argidx]
        EBV = EBV[argidx]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            (
                ppxf_result[i,:config['SFH']['MOM']],
                w_row[i,:],
                ppxf_bestfit[i,:],
                optimal_template[i,:],
                mc_results[i,:config['SFH']['MOM']],
                formal_error[i,:config['SFH']['MOM']],
                spectral_mask[i,:],
                snr_postfit[i],
                EBV[i],
            ) = run_ppxf(
                templates,
                bin_data[i,:],
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
                optimal_template_comb,
                EBV_init,
                logLam,
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
    print('Determining the mean met of ALL, YOUNG and OLD stellar populations')
    mean_results = mean_agemetalalpha(
        w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins, tag='all'
    )
    mean_results_young = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins, tag='young')

    mean_results_old = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins, tag='old')
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
        EBV,
        tag = 'ALL',
    )

    # Now save the young SPs to file 
    save_sfh(
        mean_results_young,
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
        EBV,
        tag = 'YOUNG',
    )
    # Now save the old SPs to file
    save_sfh(
        mean_results_old,
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
        EBV,
        tag = 'OLD',
    )
    # Return
    return None
