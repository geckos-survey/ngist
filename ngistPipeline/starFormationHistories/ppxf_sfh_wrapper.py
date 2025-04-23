import logging
import os
import time

# import extinction
import h5py
import numpy as np
import ppxf as ppxf_package
from astropy.io import fits
from astropy.stats import biweight_location
from joblib import Parallel, delayed, dump, load
from packaging import version
from ppxf.ppxf import ppxf
from printStatus import printStatus
from tqdm import tqdm

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

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
    printStatus.running("Running pPXF for the first time")
    # normalise galaxy spectra and noise
    median_log_bin_data = np.nanmedian(log_bin_data)
    log_bin_error = log_bin_error / median_log_bin_data
    log_bin_data = log_bin_data / median_log_bin_data
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
    nsims,
    logAge_grid,
    metal_grid,
    alpha_grid
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    # printStatus.progressBar(i, nbins, barLength=50)

    try:

        if len(optimal_template_in) > 1:

            # Normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

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

        # Currently only apply MC described by Pessa et al. 2023 (https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.147P/abstract)
        if nsims > 0:

            w_row_MC = np.zeros((nsims, w_row.shape[0]))

            for o in range(0, nsims):
                # Add noise to input spectrum "log_bin_data":
                #   - MC iterated spectrum is created by a gaussian random sampling with the mean of galaxy spectrum "log_bin_data" and sigma of "noise_new"
                #   - no regularization is applied for this step
                log_bin_data_iter = np.random.normal(loc=log_bin_data, scale=noise_new)
                mc_iter = ppxf(
                    templates,
                    log_bin_data_iter,
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
                    velscale_ratio=velscale_ratio
                )
                weights_mc_iter   = mc_iter.weights.reshape(templates.shape[1:])/mc_iter.weights.sum()
                w_row_MC[o, :]    = np.reshape(weights_mc_iter, ncomb)

            # Calculate mean and error of weights and weighted properties from MC realizations
            w_row_MC_mean         = np.nanmean(w_row_MC, axis=0)
            w_row_MC_err          = (np.nanpercentile(w_row_MC, q=84, axis=0) - np.nanpercentile(w_row_MC, q=16, axis=0))/2
            mean_results_MC_array = mean_agemetalalpha(w_row_MC, 10**logAge_grid, metal_grid, alpha_grid, nsims)
            mean_results_MC_mean  = np.nanmean(mean_results_MC_array, axis=0)
            mean_results_MC_err  = (np.nanpercentile(mean_results_MC_array, q=84, axis=0) - np.nanpercentile(mean_results_MC_array, q=16, axis=0))/2

            mc_results = {
                "w_row_MC_iter": w_row_MC,
                "w_row_MC_mean": w_row_MC_mean,
                "w_row_MC_err": w_row_MC_err,
                "mean_results_MC_iter": mean_results_MC_array,
                "mean_results_MC_mean":  mean_results_MC_mean,
                "mean_results_MC_err":  mean_results_MC_err
            }

        else:
            mc_results = {
                "w_row_MC_iter": np.nan,
                "w_row_MC_mean": np.nan,
                "w_row_MC_err": np.nan,
                "mean_results_MC_iter": np.nan,
                "mean_results_MC_mean":  np.nan,
                "mean_results_MC_err":  np.nan
            }
        # add normalisation factor back in main results
        pp.bestfit = pp.bestfit * median_log_bin_data
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

    except Exception as e:
        # Handle any other type of exception
        print(f"An error occurred: {e}")
    #     mc_results_nan = {
    #             "w_row_MC_iter": np.nan,
    #             "w_row_MC_mean": np.nan,
    #             "w_row_MC_err": np.nan,
    #             "mean_results_MC_iter": np.nan,
    #             "mean_results_MC_mean":  np.nan,
    #             "mean_results_MC_err":  np.nan
    #         }
    #     return( np.nan, np.nan, np.nan, np.nan, mc_results_nan, np.nan, np.nan, np.nan, np.nan)



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
    mean_result_MC_mean,
    mean_result_MC_err,
    ppxf_result,
    w_row,
    w_row_MC_iter,
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
):
    """ Save all results to disk. """

    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # Table HDU with stellar kinematics
    # Define the initial columns
    columns = [
        fits.Column(name="AGE", format="D", array=mean_result[:, 0]),  # Age column
        fits.Column(name="METAL", format="D", array=mean_result[:, 1]),  # Metallicity column
        fits.Column(name="ALPHA", format="D", array=mean_result[:, 2])  # Alpha column
    ]

    # If MC_PPXF is enabled, add additional columns
    if config['SFH']['MC_PPXF'] > 0:
        # Define MC columns (mean values)
        mc_columns = [
            fits.Column(name=f"{name}_MC", format="D", array=mean_result_MC_mean[:,i])
            for i, name in enumerate(["AGE", "METAL", "ALPHA"])  # Loop over AGE, METAL, and ALPHA
        ]
        # Define MC error columns
        err_columns = [
            fits.Column(name=f"ERR_{name}_MC", format="D", array=mean_result_MC_err[:,i])
            for i, name in enumerate(["AGE", "METAL", "ALPHA"])  # Loop over AGE, METAL, and ALPHA
        ]
        # Add MC columns and error columns to the main list
        columns.extend(mc_columns + err_columns)

    # If FIXED is False, add kinematic columns
    if config["SFH"]["FIXED"] == False:
        # Define kinematic columns (V and SIGMA)
        kinematic_columns = [
            fits.Column(name=name, format="D", array=ppxf_result[:, i])
            for i, name in enumerate(["V", "SIGMA"])  # Loop over V and SIGMA
        ]
        # Add kinematic columns to the main list
        columns.extend(kinematic_columns)

        # Add higher-order kinematic columns (H3, H4, H5, H6) if they exist
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(ppxf_result[:, i+2]) != 0:  # Check if the column exists
                columns.append(fits.Column(name=name, format="D", array=ppxf_result[:, i+2]))

        # Define formal error columns for kinematic parameters
        error_columns = [
            fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i])
            for i, name in enumerate(["V", "SIGMA"])  # Loop over V and SIGMA
        ]
        # Add formal error columns to the main list
        columns.extend(error_columns)

        # Add formal error columns for higher-order kinematic parameters
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(formal_error[:, i+2]) != 0:  # Check if the column exists
                columns.append(fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i+2]))

    # Add SNR_POSTFIT column to the main list
    columns.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))

    # Add E(B-V) derived from pPXF 0th step with reddening but no polynomials
    columns.append(fits.Column(name="EBV", format="D", array=EBV[:]))

    # Create the HDUs
    priHDU = fits.PrimaryHDU()
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SFH")

    # Save the configuration to the headers
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE WEIGHTS AND GRID
    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh-weights.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols_weights = [fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights), name="WEIGHTS")

    # Reshape the grids
    logAge_row, metal_row, alpha_row = map(np.reshape, [logAge_grid, metal_grid, alpha_grid], [ncomb]*3)

    # Table HDU with grids
    cols_grid = [fits.Column(name=name, format="D", array=array) 
                 for name, array in zip(["LOGAGE", "METAL", "ALPHA"], [logAge_row, metal_row, alpha_row])]
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_grid), name="GRID")

    # Create HDU list and write to file
    HDUList = fits.HDUList([_auxiliary.saveConfigToHeader(hdu, config["SFH"]) for hdu in [priHDU, dataHDU, gridHDU]])
    HDUList.writeto(outfits_sfh, overwrite=True)

    # Set additional header values
    for name, value in zip(["NAGES", "NMETAL", "NALPHA"], [nAges, nMetal, nAlpha]):
        fits.setval(outfits_sfh, name, value=value)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits"
    )
    logging.info("Wrote: " + outfits_sfh)
    # ========================
    # SAVE MC RESULTS OF WEIGHTS AND GRID
    if config['SFH']['MC_PPXF'] > 0:

        outfits_sfh = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_sfh-weights_mc.fits"
        )
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights_mc.fits")

        # Primary HDU
        priHDU = fits.PrimaryHDU()

        # Table HDU with weights from MC results
        dataHDU_list = []
        for iter_i in range(config['SFH']['MC_PPXF']):
            cols = [fits.Column(name="WEIGHTS_MC", format=str(w_row_MC_iter.shape[2]) + "D", array=w_row_MC_iter[:, iter_i, :])]
            dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
            dataHDU.name = "MC_ITER_%s" % iter_i
            dataHDU_list.append(dataHDU)

        # Create HDU list and write to file
        priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
        HDUList = fits.HDUList([priHDU] + dataHDU_list)
        HDUList.writeto(outfits_sfh, overwrite=True)

        fits.setval(outfits_sfh, "NAGES", value=nAges)
        fits.setval(outfits_sfh, "NMETAL", value=nMetal)
        fits.setval(outfits_sfh, "NALPHA", value=nAlpha)

        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights_mc.fits"
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
    Args:
    - config: dictionary containing configuration parameters
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    # Prepare template library
    
    # Open the HDF5 file
    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5", 'r') as f:
        # Read the VELSCALE attribute from the file
        velscale = f.attrs['VELSCALE']
        
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

    # Define file paths
    gas_cleaned_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + '_gas-cleaned_'+config['GAS']['LEVEL']+'.fits'
    bin_spectra_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5"

    # Check if emission-subtracted spectra file exists
    if (config['SFH']['SPEC_EMICLEAN'] == True) and os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        # Open the FITS file
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            # Read the LOGLAM data from the file
            logLam = hdul[2].data['LOGLAM']

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config['SFH']['LMIN'], np.exp(logLam) < config['SFH']['LMAX']))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            bin_data = hdul[1].data['SPEC'].T[idx_lam, :]
            bin_err = hdul[1].data['ESPEC'].T[idx_lam, :]
            logLam = logLam[idx_lam]
            nbins = bin_data.shape[1]
            npix = bin_data.shape[0]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            # Read the LOGLAM data from the file
            logLam = f['LOGLAM'][:]

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config['SFH']['LMIN'], np.exp(logLam) < config['SFH']['LMAX']))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            bin_data = f['SPEC'][idx_lam, :]
            bin_err = f['ESPEC'][idx_lam, :]
            logLam = logLam[idx_lam]

    # Define additional variables
    nbins = bin_data.shape[1]
    npix = bin_data.shape[0]
    ubins = np.arange(nbins)
    noise = np.full(npix, config['SFH']['NOISE'])
    dv = (np.log(lamRange_temp[0]) - logLam[0])*C

    # Apply the selection to the logLam array


    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    noise = bin_err  # is actual noise, not variance
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
            + "_kin.fits", mem_map=True
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
    formal_error = np.zeros((nbins,6))
    spectral_mask = np.zeros((nbins,bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)
    EBV = np.zeros(nbins)

    # Define output arrays of MC realizations
    if nsims > 0:
        logging.info('MC realizations will be applied with %s iterations to estimate weights uncertainties.' % nsims)
    w_row_MC_iter        = np.zeros((nbins,nsims,ncomb))
    w_row_MC_mean        = np.zeros((nbins,ncomb))
    w_row_MC_err         = np.zeros((nbins,ncomb))
    mean_results_MC_iter = np.zeros((nbins,nsims,3))
    mean_results_MC_mean = np.zeros((nbins,3))
    mean_results_MC_err  = np.zeros((nbins,3))

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

        # Prepare the folder where the memmap will be dumped
        memmap_folder = "/scratch" if os.access("/scratch", os.W_OK) else config["GENERAL"]["OUTPUT"]

        # dump the arrays and load as memmap
        templates_filename_memmap = memmap_folder + "/templates_memmap.tmp"
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode='r')
        
        bin_data_filename_memmap = memmap_folder + "/bin_data_memmap.tmp"
        dump(bin_data, bin_data_filename_memmap)
        bin_data = load(bin_data_filename_memmap, mmap_mode='r')
        
        noise_filename_memmap = memmap_folder + "/noise_memmap.tmp"
        dump(noise, noise_filename_memmap)
        noise = load(noise_filename_memmap, mmap_mode='r')

        # Define a function to encapsulate the work done in the loop
        def worker(chunk, templates):
            results = []
            for i in chunk:
                result = run_ppxf(
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
                    config['SFH']['MC_PPXF'],
                    logAge_grid,
                    metal_grid,
                    alpha_grid
                )
                results.append(result)
            return results

        # Use joblib to parallelize the work
        max_nbytes = "1M" # max array size before memory mapping is triggered
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)]
        parallel_configs = {"n_jobs": config["GENERAL"]["NCPU"], "max_nbytes": max_nbytes, "temp_folder": memmap_folder, "mmap_mode": "c", "return_as":"generator"}
        ppxf_tmp = list(tqdm(Parallel(**parallel_configs)(delayed(worker)(chunk, templates) for chunk in chunks),
                        total=len(chunks), desc="Processing chunks", ascii=" #", unit="chunk"))

        # Flatten the results
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        # Unpack results
        for i in range(0, nbins):
            ppxf_result[i,:config['SFH']['MOM']] = ppxf_tmp[i][0]
            w_row[i,:] = ppxf_tmp[i][1]
            ppxf_bestfit[i,:] = ppxf_tmp[i][2]
            optimal_template[i,:] = ppxf_tmp[i][3]
            w_row_MC_iter[i,:,:] = ppxf_tmp[i][4]["w_row_MC_iter"]
            w_row_MC_mean[i,:] = ppxf_tmp[i][4]["w_row_MC_mean"]
            w_row_MC_err[i,:] = ppxf_tmp[i][4]["w_row_MC_err"]
            mean_results_MC_iter[i,:,:] = ppxf_tmp[i][4]["mean_results_MC_iter"]
            mean_results_MC_mean[i,:]  = ppxf_tmp[i][4]["mean_results_MC_mean"]
            mean_results_MC_err[i,:]  = ppxf_tmp[i][4]["mean_results_MC_err"]
            formal_error[i,:config['SFH']['MOM']] = ppxf_tmp[i][5]
            spectral_mask[i,:] = ppxf_tmp[i][6]
            snr_postfit[i] = ppxf_tmp[i][7]
            EBV[i] = ppxf_tmp[i][8]

        # Remove the memory-mapped files
        os.remove(templates_filename_memmap)
        os.remove(bin_data_filename_memmap)
        os.remove(noise_filename_memmap)
        
        printStatus.updateDone("Running PPXF in parallel mode", progressbar=False)
        

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            (
                ppxf_result[i,:config['SFH']['MOM']],
                w_row[i,:],
                ppxf_bestfit[i,:],
                optimal_template[i,:],
                mc_results_i,
                formal_error[i,:config['SFH']['MOM']],
                spectral_mask[i,:],
                snr_postfit[i],
                EBV[i],
            ) = run_ppxf(
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
                config['SFH']['MC_PPXF'],
                logAge_grid,
                metal_grid,
                alpha_grid
            )
            w_row_MC_iter[i,:,:] = mc_results_i["w_row_MC_iter"]
            w_row_MC_mean[i,:] = mc_results_i["w_row_MC_mean"]
            w_row_MC_err[i,:] = mc_results_i["w_row_MC_err"]
            mean_results_MC_iter[i,:,:] = mc_results_i["mean_results_MC_iter"]
            mean_results_MC_mean[i,:] = mc_results_i["mean_results_MC_mean"]
            mean_results_MC_err[i,:] = mc_results_i["mean_results_MC_err"]
        printStatus.updateDone("Running PPXF in serial mode", progressbar=False)

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
        mean_results_MC_mean,
        mean_results_MC_err,
        ppxf_result,
        w_row,
        w_row_MC_iter,
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
    )

    # Return
    return None
