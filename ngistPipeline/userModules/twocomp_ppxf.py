import code
import logging
import os
import time

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from multiprocess import Process, Queue
from ppxf.ppxf import ppxf
from printStatus import printStatus

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE:
  This module executes the analysis of stellar kinematics in the pipeline.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""


def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    d = y if zero else y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad)) ** 2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1**2) ** 2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

    return sigma


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for (
        templates,
        bin_data,
        noise,
        velscale,
        start,
        bias,
        goodPixels_ppxf,
        nmoments,
        adeg,
        mdeg,
        reddening,
        doclean,
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        i,
        ntemplates,
        optimal_template_in,
    ) in iter(inQueue.get, "STOP"):
        (
            sol,
            ppxf_reddening,
            bestfit,
            optimal_template1,
            optimal_template2,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            tweights,
        ) = run_twocomp_ppxf(
            templates,
            bin_data,
            noise,
            velscale,
            start,
            bias,
            goodPixels_ppxf,
            nmoments,
            adeg,
            mdeg,
            reddening,
            doclean,
            logLam,
            offset,
            velscale_ratio,
            nsims,
            nbins,
            i,
            ntemplates,
            optimal_template_in,
        )

        outQueue.put(
            (
                i,
                sol,
                ppxf_reddening,
                bestfit,
                optimal_template1,
                optimal_template2,                
                mc_results,
                formal_error,
                spectral_mask,
                snr_postfit,
                tweights,
            )
        )


def run_twocomp_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    bias,
    goodPixels,
    nmoments,
    adeg,
    mdeg,
    reddening,
    doclean,
    logLam,
    offset,
    velscale_ratio,
    nsims,
    nbins,
    i,
    ntemplates,
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics.
    """
    printStatus.progressBar(i, nbins, barLength=50)


    try:
        # normalise galaxy spectra and noise
        median_log_bin_data = np.nanmedian(log_bin_data)
        log_bin_error /= median_log_bin_data
        log_bin_data /= median_log_bin_data

        # Call PPXF for first time to get optimal template
        if len(optimal_template_in) == 1:
            printStatus.running("Running pPXF for the first time")

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
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

            # Make the unconvolved optimal stellar template
            normalized_weights = pp.weights / np.sum( pp.weights )
            optimal_template1   = np.zeros( templates.shape[0] )
            for j in range(0, templates.shape[1]):
                optimal_template1 = optimal_template1 + templates[:,j]*normalized_weights[j]
            #output requires two optimal templates
            optimal_template2 = optimal_template1

            weights = pp.weights

        else:
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)
            # use first guess start
            start_tmp = start[0,:]

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start_tmp,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

            # Find a proper estimate of the noise
            noise_orig = biweight_location(log_bin_error[goodPixels])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels] - pp_step1.bestfit[goodPixels]
            )

            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 2 ##################
            # Second Call PPXF - use best-fitting template, determine outliers
            # only do this if doclean is set
            
            use_local_templ = True
            templates_step2 = []
            if use_local_templ:
                templates_step2 = templates[:,0:ntemplates[0]]
            else:
                templates_step2 = optimal_template_in

            if doclean == True:
                pp_step2 = ppxf(
                    templates_step2,
                    log_bin_data,
                    noise_new,
                    velscale,
                    start_tmp,
                    goodpixels=goodPixels,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    degree=adeg,
                    mdegree=mdeg,
                    reddening=reddening,
                    lam=np.exp(logLam),
                    velscale_ratio=velscale_ratio,
                    vsyst=offset,
                    clean=True,
                )

                # update goodpixels
                goodPixels = pp_step2.goodpixels

                # repeat noise scaling # Find a proper estimate of the noise
                noise_orig = biweight_location(log_bin_error[goodPixels])
                noise_est = robust_sigma(
                    pp_step2.galaxy[goodPixels] - pp_step2.bestfit[goodPixels]
                )

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error * (noise_est / noise_orig)
                noise_new_std = robust_sigma(noise_new)

                # Find non-zero weights
                wnonzero_weights_step2 = np.squeeze(np.where(pp_step2.weights > 0))
                optimal_templates_local = templates[:,wnonzero_weights_step2]
                
            # A fix for the noise issue where a single high S/N spaxel
            # causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 3 ##################
            # Third Call PPXF - use all templates, get best-fit
            
            # Check if we want to use all templates, or just local
            if use_local_templ:
                templates_use = optimal_templates_local
            else:
                templates_use = templates

            # Create step 3 templates
            ntemplates_step3 = templates_use.shape[1]
            templates_step3 = np.zeros( (templates_use.shape[0],ntemplates_step3*2) )

            # Fill step 3 template with either all or local templates
            templates_step3[:,:ntemplates_step3] = templates_use
            templates_step3[:,ntemplates_step3:] = templates_use            
            
            # kinematic constraints sigma0 >  sigma1
            #A_ineq = [[0, 1, 0, -1]]  # -sigma0 + sigma1 <= 0
            #b_ineq = [0]
            #constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

            # kinematic constraints vel0 <  vel1
            #A_ineq = [[-1, 0, 1, 0]]  # -vel0 + vel1 <= 0
            #b_ineq = [0]
            #constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

            # kinematic constraints vel0 <  vel1 & sigma0 > sigma 1
            A_ineq = [[-1, 0, 1, 0],  # -vel0 + vel1 <= 0
                      [0, 1, 0, -1]]  #  sigma0 > sigma 1
            b_ineq = [0, 0]
            constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

            input_component = np.arange(np.int32(templates_step3.shape[1]))
            input_component[:] = 0
            input_component[:ntemplates_step3] = 0
            input_component[ntemplates_step3:] = 1
            
            # template weight constraints
            #fraction=0.5
            #A_eq = np.zeros(np.int32(templates_step3.shape[1]))
            #A_eq[:ntemplates_step3] = fraction-1
            #A_eq[ntemplates_step3:] = fraction
            #b_eq = [0]
            #constr_templ = {"A_eq": [A_eq], "b_eq": b_eq}
            
            # Inequality. Requires ,linear_method='lsq_lin' or linear_method='cvxopt'
            #fraction=0.10
            #A_ineq = np.zeros((2,np.int32(templates_step3.shape[1])))
            #A_ineq[0,:ntemplates_step3] = fraction-1
            #A_ineq[1,:ntemplates_step3] = fraction

            #A_ineq[0,ntemplates_step3:] = fraction
            #A_ineq[1,ntemplates_step3:] = fraction-1
            #b_ineq = [0,0]
            #constr_templ = {"A_ineq": A_ineq, "b_ineq": b_ineq}

            # get better default guess from previous fit
            start[0,0] = pp_step2.sol[0]+25
            start[1,0] = pp_step2.sol[0]-25
            start[0,1] = pp_step2.sol[1]-5
            start[1,1] = pp_step2.sol[1]+5

            pp = ppxf(
                templates_step3,
                log_bin_data,
                noise_new,
                velscale,
                start,
                bias,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=[nmoments,nmoments],
                degree=adeg,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                component=input_component,
                constr_kinem=constr_kinem, #,constr_templ=constr_templ) #,linear_method='lsq_lin')
            )
            
            # Make the unconvolved optimal stellar template
            normalized_weights = pp.weights / np.sum( pp.weights )
            optimal_template1   = np.zeros( templates_step3.shape[0] )
            optimal_template2   = np.zeros( templates_step3.shape[0] )

            for j in range(0, ntemplates_step3):
                optimal_template1 = optimal_template1 + templates_step3[:,j]*normalized_weights[j]
            for j in range(ntemplates_step3,templates_step3.shape[1]):
                optimal_template2 = optimal_template2 + templates_step3[:,j]*normalized_weights[j]

            # transfer weights step 3 back into original templates format if local templ are used instead of all templates
            if use_local_templ:
                weights = np.zeros( templates.shape[1])
                weights[0:ntemplates[0]][wnonzero_weights_step2] = pp.weights[0:ntemplates_step3]
                weights[ntemplates[0]:][wnonzero_weights_step2] = pp.weights[ntemplates_step3:]             
            else:
                weights = pp.weights

        # update goodpixels again
        goodPixels = pp.goodpixels

        # make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # Calculate the true S/N from the residual
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmean(pp.galaxy[goodPixels]/noise_est)

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error
        formal_error[0] = pp.error[0] * np.sqrt(pp.chi2)
        formal_error[1] = pp.error[1] * np.sqrt(pp.chi2)
                    

        # Do MC-Simulations
        # will not work for multicomp
        sol_MC = np.zeros((nsims, nmoments))
        mc_results = np.zeros(nmoments)

        if nsims != 0:
            mc_results = np.nanstd(sol_MC, axis=0)

        # add normalisation factor back in main results
        pp.bestfit *= median_log_bin_data
        if pp.reddening is not None:
            pp.reddening *= median_log_bin_data

        return(
            pp.sol[:],
            pp.reddening,
            pp.bestfit,
            optimal_template1,
            optimal_template2,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            weights,
        )

    except:
        return ([np.nan,np.nan], np.nan, np.nan, np.nan, np.nan, [np.nan,np.nan], [np.nan,np.nan],np.nan, np.nan, np.nan)


def save_ppxf(
    config,
    ppxf_result,
    ppxf_reddening,
    mc_results,
    formal_error,
    ppxf_bestfit,
    logLam,
    goodPixels,
    optimal_template1,
    optimal_template2,
    logLam_template,
    npix,
    spectral_mask,
    optimal_template_comb,
    bin_data,
    snr_postfit,
    template_weights,
):
    """Saves all results to disk."""
    # ========================
    # SAVE RESULTS
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_twocomp_kin.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    # Output for Component number 1    
    cols.append(fits.Column(name="V_C1", format="D", array=ppxf_result[:, 0, 0]))
    cols.append(fits.Column(name="SIGMA_C1", format="D", array=ppxf_result[:, 0, 1]))
    if np.any(ppxf_result[:, 0, 2]) != 0:
        cols.append(fits.Column(name="H3_C1", format="D", array=ppxf_result[:, 0, 2]))
    if np.any(ppxf_result[:, 0, 3]) != 0:
        cols.append(fits.Column(name="H4_C1", format="D", array=ppxf_result[:, 0, 3]))
    if np.any(ppxf_result[:, 0, 4]) != 0:
        cols.append(fits.Column(name="H5_C1", format="D", array=ppxf_result[:, 0, 4]))
    if np.any(ppxf_result[:, 0, 5]) != 0:
        cols.append(fits.Column(name="H6_C1", format="D", array=ppxf_result[:, 0, 5]))
    
    if np.any(mc_results[:, 0, 0]) != 0:
        cols.append(fits.Column(name="ERR_V_C1", format="D", array=mc_results[:, 0, 0]))
    if np.any(mc_results[:, 0, 1]) != 0:
        cols.append(fits.Column(name="ERR_SIGMA_C1", format="D", array=mc_results[:, 0, 1]))
    if np.any(mc_results[:, 0, 2]) != 0:
        cols.append(fits.Column(name="ERR_H3_C1", format="D", array=mc_results[:, 0, 2]))
    if np.any(mc_results[:, 0, 3]) != 0:
        cols.append(fits.Column(name="ERR_H4_C1", format="D", array=mc_results[:, 0, 3]))
    if np.any(mc_results[:, 0, 4]) != 0:
        cols.append(fits.Column(name="ERR_H5_C1", format="D", array=mc_results[:, 0, 4]))
    if np.any(mc_results[:, 0, 5]) != 0:
        cols.append(fits.Column(name="ERR_H6_C1", format="D", array=mc_results[:, 0, 5]))#

    cols.append(fits.Column(name="FORM_ERR_V_C1", format="D", array=formal_error[:, 0, 0]))
    cols.append(
        fits.Column(name="FORM_ERR_SIGMA_C1", format="D", array=formal_error[:, 0, 1])
    )
    if np.any(formal_error[:, 0, 2]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H3_C1", format="D", array=formal_error[:, 0, 2])
        )
    if np.any(formal_error[:, 0, 3]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H4_C1", format="D", array=formal_error[:, 0, 3])
        )
    if np.any(formal_error[:, 0, 4]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H5_C1", format="D", array=formal_error[:, 0, 4])
        )
    if np.any(formal_error[:, 0, 5]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H6_C1", format="D", array=formal_error[:, 0, 5])
        )#

    # Output for Component number 2
    cols.append(fits.Column(name="V_C2", format="D", array=ppxf_result[:, 1, 0]))
    cols.append(fits.Column(name="SIGMA_C2", format="D", array=ppxf_result[:, 1, 1]))
    if np.any(ppxf_result[:, 1, 2]) != 0:
        cols.append(fits.Column(name="H3_C2", format="D", array=ppxf_result[:, 1, 2]))
    if np.any(ppxf_result[:, 1, 3]) != 0:
        cols.append(fits.Column(name="H4_C2", format="D", array=ppxf_result[:, 1, 3]))
    if np.any(ppxf_result[:, 1, 4]) != 0:
        cols.append(fits.Column(name="H5_C2", format="D", array=ppxf_result[:, 1, 4]))
    if np.any(ppxf_result[:, 1, 5]) != 0:
        cols.append(fits.Column(name="H6_C2", format="D", array=ppxf_result[:, 1, 5]))

    if np.any(mc_results[:, 1, 0]) != 0:
        cols.append(fits.Column(name="ERR_V_C2", format="D", array=mc_results[:, 1, 0]))
    if np.any(mc_results[:, 1, 1]) != 0:
        cols.append(fits.Column(name="ERR_SIGMA_C2", format="D", array=mc_results[:, 1, 1]))
    if np.any(mc_results[:, 1, 2]) != 0:
        cols.append(fits.Column(name="ERR_H3_C2", format="D", array=mc_results[:, 1, 2]))
    if np.any(mc_results[:, 1, 3]) != 0:
        cols.append(fits.Column(name="ERR_H4_C2", format="D", array=mc_results[:, 1, 3]))
    if np.any(mc_results[:, 1, 4]) != 0:
        cols.append(fits.Column(name="ERR_H5_C2", format="D", array=mc_results[:, 1, 4]))
    if np.any(mc_results[:, 1, 5]) != 0:
        cols.append(fits.Column(name="ERR_H6_C2", format="D", array=mc_results[:, 1, 5]))

    cols.append(fits.Column(name="FORM_ERR_V_C2", format="D", array=formal_error[:, 1, 0]))
    cols.append(
        fits.Column(name="FORM_ERR_SIGMA_C2", format="D", array=formal_error[:, 1, 1])
    )
    if np.any(formal_error[:, 1, 2]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H3_C2", format="D", array=formal_error[:, 1, 2])
        )
    if np.any(formal_error[:, 1, 3]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H4_C2", format="D", array=formal_error[:, 1, 3])
        )
    if np.any(formal_error[:, 1, 4]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H5_C2", format="D", array=formal_error[:, 1, 4])
        )
    if np.any(formal_error[:, 1, 5]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H6_C2", format="D", array=formal_error[:, 1, 5])
        )        

    ## Add reddening if parameter is used
    if np.any(np.isnan(ppxf_reddening)) != True:
        cols.append(fits.Column(name="REDDENING", format="D", array=ppxf_reddening[:]))

    ## Add True SNR calculated from residual
    cols.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "KIN_DATA"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["UMOD"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["UMOD"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin.fits")
    logging.info("Wrote: " + outfits_ppxf)

    # ========================
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_twocomp_kin-bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-bestfit.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=ppxf_bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with PPXF logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Table HDU with ??? --> unclear what this is?
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["UMOD"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["UMOD"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["UMOD"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["UMOD"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["UMOD"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, specHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-bestfit.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_twocomp_kin-optimalTemplates.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-optimalTemplates.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATES1",
            format=str(optimal_template1.shape[1]) + "D",
            array=optimal_template1,
        )
    )
    dataHDU1 = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU1.name = "OPTIMAL_TEMPLATES1"

    # Extension 2: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATES2",
            format=str(optimal_template2.shape[1]) + "D",
            array=optimal_template2,
        )
    )
    dataHDU2 = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU2.name = "OPTIMAL_TEMPLATES2"

    # Extension 3: Table HDU with logLam_templates
    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM_TEMPLATE"

    # Extension 4: Table HDU with logLam_templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATE_ALL", format="D", array=optimal_template_comb
        )
    )
    combHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    combHDU.name = "OPTIMAL_TEMPLATE_ALL"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["UMOD"])
    dataHDU1 = _auxiliary.saveConfigToHeader(dataHDU1, config["UMOD"])
    dataHDU2 = _auxiliary.saveConfigToHeader(dataHDU2, config["UMOD"])    
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["UMOD"])
    combHDU = _auxiliary.saveConfigToHeader(combHDU, config["UMOD"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, combHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-optimalTemplates.fits"
    )
    logging.info("Wrote: " + outfits)

    # ============================
    # SAVE SPECTRAL MASK RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_twocomp_kin-SpectralMask.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-SpectralMask.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="SPECTRAL_MASK",
            format=str(spectral_mask.shape[1]) + "D",
            array=spectral_mask,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "SPECTRAL_MASK"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["UMOD"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["UMOD"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_twocomp_kin-SpectralMask.fits"
    )
    logging.info("Wrote: " + outfits)

    # ============================
    # SAVE WEIGHTS and TEMPLATE Properties
    outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_twocomp_kin-weightTemplates.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_twocomp_kin-weightTemplates.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with flux_1
    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name='WEIGHT_C1' , format='D', array=template_weights[:,0,0]))
    cols.append(fits.Column(name='FLUX_C1', format='D', array=template_weights[:,0,1]))
    cols.append(fits.Column(name='LOGAGE_C1', format='D', array=template_weights[:,0,2]))
    cols.append(fits.Column(name='METAL_C1', format='D', array=template_weights[:,0,3]))
    cols.append(fits.Column(name='ALPHA_C1', format='D', array=template_weights[:,0,4]))

    cols.append(fits.Column(name='WEIGHT_C2' , format='D', array=template_weights[:,1,0]))
    cols.append(fits.Column(name='FLUX_C2', format='D', array=template_weights[:,1,1]))
    cols.append(fits.Column(name='LOGAGE_C2', format='D', array=template_weights[:,1,2]))
    cols.append(fits.Column(name='METAL_C2', format='D', array=template_weights[:,1,3]))
    cols.append(fits.Column(name='ALPHA_C2', format='D', array=template_weights[:,1,4]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'WEIGHT_DATA'
    
    # Create HDU list and write to file
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['UMOD'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['UMOD'])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_twocomp_kin-weightTemplates.fits')
    logging.info("Wrote: "+outfits)



def extractStellarKinematics(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the GIST conventions.
    """
    # Read data from file
    hdu = fits.open(
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_BinSpectra.fits"
    )
    bin_data = np.array(hdu[1].data.SPEC.T)
    bin_err = np.array(hdu[1].data.ESPEC.T)
    logLam = np.array(hdu[2].data.LOGLAM)
    idx_lam = np.where(
        np.logical_and(
            np.exp(logLam) > config["UMOD"]["LMIN"],
            np.exp(logLam) < config["UMOD"]["LMAX"],
        )
    )[0]
    bin_data = bin_data[idx_lam, :]
    bin_err = bin_err[idx_lam, :]
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)
    velscale = hdu[0].header["VELSCALE"]

    # Define bias value
    if config["UMOD"]["BIAS"] == 'Auto': # 'Auto' setting: bias=None
        bias = None
    elif config["UMOD"]["BIAS"] != 'Auto':
        bias = config["UMOD"]["BIAS"]

    # Read LSF information

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "UMOD")  # added input of module

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
        config["UMOD"]["LMIN"],
        config["UMOD"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        'UMOD',
        sortInGrid=True,
    )

    #save a copy for the initial pPXF run
    templates_def = templates.copy()
    templates_def = templates_def.reshape( (templates_def.shape[0], ntemplates) )

    # Select Two Sets of Templates
    # two options:  1) use all templates
    #               2) use range in age and metallicity

    templ_choice = 1

    if templ_choice == 1:
        printStatus.running("Using Full set of templates for both components")
       
        #now combine templates and stelpop grids 
        templates1 = templates 

        #select range of templates (== all here)
        min_age, max_age, min_z, max_z, min_a, max_a = -0.5, 1.2, -1.49, 0.4, 0.0, 0.4

        wtempl = np.where( (logAge_grid >= min_age) & (logAge_grid <= max_age) & \
            (metal_grid >= min_z) & (metal_grid <= max_z) & \
            (alpha_grid >= min_a) & (alpha_grid <= max_a))

        templates1 = templates[:,wtempl[0],wtempl[1],wtempl[2]]
        logAge_tgrid1 = logAge_grid[wtempl[0],wtempl[1],wtempl[2]]
        metal_tgrid1 = metal_grid[wtempl[0],wtempl[1],wtempl[2]]
        alpha_tgrid1 = alpha_grid[wtempl[0],wtempl[1],wtempl[2]]

        templates2 = templates[:,wtempl[0],wtempl[1],wtempl[2]]
        logAge_tgrid2 = logAge_grid[wtempl[0],wtempl[1],wtempl[2]]
        metal_tgrid2 = metal_grid[wtempl[0],wtempl[1],wtempl[2]]
        alpha_tgrid2 = alpha_grid[wtempl[0],wtempl[1],wtempl[2]]

        #now combine templates and stelpop grids
        ntemplates_c1 = templates1[0,:].size
        ntemplates_c2 = templates2[0,:].size

        templates1 = templates1.reshape( (templates1.shape[0], ntemplates_c1) )    
        templates2 = templates2.reshape( (templates2.shape[0], ntemplates_c2) )

        templates = np.column_stack([templates1, templates2])
        ntemplates =  templates[0,:].size

        printStatus.updateDone("Selected number of templates: " + str(ntemplates))
        templates = templates.reshape( (templates.shape[0], ntemplates) )

        #make the complete template age, metal, and alpha grids (needed later for output)
        logAge_tgrid = np.concatenate((logAge_tgrid1,logAge_tgrid2))
        metal_tgrid = np.concatenate((metal_tgrid1,metal_tgrid2))
        alpha_tgrid = np.concatenate((alpha_tgrid1,alpha_tgrid2))

    else:

        #must select equal number of template 1 and template 2!
        #select the first range of templates
        min_age1, max_age1, min_z1, max_z1, min_a1, max_a1 = -0.5, 1.2, -1.49, 0.4, 0.0, 0.0

        wtempl = np.where( (logAge_grid >= min_age1) & (logAge_grid <= max_age1) & \
            (metal_grid >= min_z1) & (metal_grid <= max_z1) & \
            (alpha_grid >= min_a1) & (alpha_grid <= max_a1))

        wage = np.where( (logAge_grid[:,0,0] >= min_age1) & (logAge_grid[:,0,0] <= max_age1))[0]
        wmetal = np.where((metal_grid[0,:,0] >= min_z1) & (metal_grid[0,:,0] <= max_z1))[0]
        walpha = np.where((alpha_grid[0,0,:] >= min_a1) & (alpha_grid[0,0,:] <= max_a1))[0]
        
        print("wage",logAge_grid[np.squeeze(wage),0,0])
        print("wmetal",metal_grid[0,np.squeeze(wmetal),0])
        print("walpha",alpha_grid[0,0,np.squeeze(walpha)])

        templates1 = templates[:,wtempl[0],wtempl[1],wtempl[2]]
        logAge_tgrid1 = logAge_grid[wtempl[0],wtempl[1],wtempl[2]]
        metal_tgrid1 = metal_grid[wtempl[0],wtempl[1],wtempl[2]]
        alpha_tgrid1 = alpha_grid[wtempl[0],wtempl[1],wtempl[2]]

        #templates1 = templates[:,np.int(wage[0]):np.int(wage[-1]+1),
        #np.int(wmetal[0]):np.int(wmetal[-1]+1),
        #np.int(walpha[0]):np.int(walpha[-1]+1)]

        #select the second range of templates
        #min_age2, max_age2, min_z2, max_z2, min_a2, max_a2 = -0.5, 1.2, -1.49, 0.4, 0.4, 0.4
        min_age2, max_age2, min_z2, max_z2, min_a2, max_a2 = -0.5, 1.2, -1.49, 0.4, 0.0, 0.0
        wage = np.where( (logAge_grid[:,0,0] >= min_age2) & (logAge_grid[:,0,0] <= max_age2))[0]
        wmetal = np.where((metal_grid[0,:,0] >= min_z2) & (metal_grid[0,:,0] <= max_z2))[0]
        walpha = np.where((alpha_grid[0,0,:] >= min_a2) & (alpha_grid[0,0,:] <= max_a2))[0]
        
        wtempl = np.where( (logAge_grid >= min_age2) & (logAge_grid <= max_age2) & \
            (metal_grid >= min_z2) & (metal_grid <= max_z2) & \
            (alpha_grid >= min_a2) & (alpha_grid <= max_a2))
        
        print("wage",logAge_grid[np.squeeze(wage),0,0])
        print("wmetal",metal_grid[0,np.squeeze(wmetal),0])
        print("walpha",alpha_grid[0,0,np.squeeze(walpha)])

        templates2 = templates[:,wtempl[0],wtempl[1],wtempl[2]]
        logAge_tgrid2 = logAge_grid[wtempl[0],wtempl[1],wtempl[2]]
        metal_tgrid2 = metal_grid[wtempl[0],wtempl[1],wtempl[2]]
        alpha_tgrid2 = alpha_grid[wtempl[0],wtempl[1],wtempl[2]]

        #now combine templates and stelpop grids
        ntemplates_c1 = templates1[0,:].size
        ntemplates_c2 = templates2[0,:].size

        templates1 = templates1.reshape( (templates1.shape[0], ntemplates_c1) )    
        templates2 = templates2.reshape( (templates2.shape[0], ntemplates_c2) )

        templates = np.column_stack([templates1, templates2])
        ntemplates =  templates[0,:].size

        printStatus.updateDone("Selected number of templates: " + str(ntemplates))
        templates = templates.reshape( (templates.shape[0], ntemplates) )

        #make the complete template age, metal, and alpha grids (needed later for output)
        logAge_tgrid = np.concatenate((logAge_tgrid1,logAge_tgrid2))
        metal_tgrid = np.concatenate((metal_tgrid1,metal_tgrid2))
        alpha_tgrid = np.concatenate((alpha_tgrid1,alpha_tgrid2))

        #end of template selection

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0]) * C
    # noise  = np.ones((npix,nbins))
    noise = bin_err  # is actual noise, not variance
    nsims = config["UMOD"]["MC_PPXF"]

    # Initial guesses
    start = np.zeros((nbins, 2, 2))
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )
        == True
    ):
        printStatus.done(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        logging.info(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        guess = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )[1].data
        # ignore guess values for now
        #start[:, 0] = guess.V
        #start[:, 1] = guess.SIGMA
        start[:,0,0] = 0
        start[:,1,0] = 0
        start[:,0,1] = 50
        start[:,1,1] = 75
    else:
        # Use the same initial guess for all bins, as stated in MasterConfig
        printStatus.done(
            "Using V and SIGMA from the MasterConfig file as initial guesses"
        )
        logging.info("Using V and SIGMA from the MasterConfig file as initial guesses")
        #ignore guess values for now
        #start[:, 0] = 0.0
        #start[:, 1] = config["UMOD"]["SIGMA"]
        start[:,0,0] = 0
        start[:,1,0] = 0
        start[:,0,1] = 50
        start[:,1,1] = 75        

    #extra start for first pPXF fit
    start_def = np.zeros((nbins,2))
    start_def[:,0] = 0
    start_def[:,1] = 75

    # Define goodpixels
    goodPixels_ppxf = _auxiliary.spectralMasking(
        config, config["UMOD"]["SPEC_MASK"], logLam
    )

    # Array to store results of ppxf
    ppxf_result = np.zeros((nbins,2,6))
    ppxf_reddening = np.zeros(nbins)
    ppxf_bestfit = np.zeros((nbins, npix))
    optimal_template1 = np.zeros((nbins,templates.shape[0]))
    optimal_template2 = np.zeros((nbins,templates.shape[0])) 
    mc_results = np.zeros((nbins,2,6))
    formal_error = np.zeros((nbins,2,6))
    spectral_mask = np.zeros((nbins, bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)
    tweights = np.zeros((nbins,ntemplates))
    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:, :], axis=1)
    comb_espec = np.nanmean(bin_err[:, :], axis=1)
    optimal_template_init = [0]

    (
        tmp_ppxf_result,
        tmp_ppxf_reddening,
        tmp_ppxf_bestfit,
        optimal_template_out,
        tmp_optimal_template_out,
        tmp_mc_results,
        tmp_formal_error,
        tmp_spectral_mask,
        tmp_snr_postfit,
        tmp_tweights,
    ) = run_twocomp_ppxf(
        templates_def,
        comb_spec,
        comb_espec,
        velscale,
        start_def[0, :],
        bias,
        goodPixels_ppxf,
        config["UMOD"]["MOM"],
        config["UMOD"]["ADEG"],
        config["UMOD"]["MDEG"],
        config["UMOD"]["REDDENING"],
        config["UMOD"]["DOCLEAN"],
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        0,
        [ntemplates_c1,ntemplates_c2],
        optimal_template_init,
    )
    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

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
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    bias,
                    goodPixels_ppxf,
                    config["UMOD"]["MOM"],
                    config["UMOD"]["ADEG"],
                    config["UMOD"]["MDEG"],
                    config["UMOD"]["REDDENING"],
                    config["UMOD"]["DOCLEAN"],
                    logLam,
                    offset,
                    velscale_ratio,
                    nsims,
                    nbins,
                    i,
                    [ntemplates_c1,ntemplates_c2],
                    optimal_template_comb,
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
            ppxf_result[i, 0, : config["UMOD"]["MOM"]] = ppxf_tmp[i][1][0]
            ppxf_result[i, 1, : config["UMOD"]["MOM"]] = ppxf_tmp[i][1][1]            
            ppxf_reddening[i] = ppxf_tmp[i][2]
            ppxf_bestfit[i, :] = ppxf_tmp[i][3]
            optimal_template1[i, :] = ppxf_tmp[i][4]
            optimal_template2[i, :] = ppxf_tmp[i][5]            
            mc_results[i,0, : config["UMOD"]["MOM"]] = ppxf_tmp[i][6][0]
            mc_results[i,1, : config["UMOD"]["MOM"]] = ppxf_tmp[i][6][1]
            formal_error[i,0, : config["UMOD"]["MOM"]] = ppxf_tmp[i][7][0]
            formal_error[i,1, : config["UMOD"]["MOM"]] = ppxf_tmp[i][7][1]            
            spectral_mask[i, :] = ppxf_tmp[i][8]
            snr_postfit[i] = ppxf_tmp[i][9]
            tweights[i,:] = ppxf_tmp[i][10]            

        # Sort output
        argidx = np.argsort(index)
        ppxf_result = ppxf_result[argidx, :, :]
        ppxf_reddening = ppxf_reddening[argidx]
        ppxf_bestfit = ppxf_bestfit[argidx, :]
        optimal_template1 = optimal_template1[argidx, :]
        optimal_template2 = optimal_template2[argidx, :]        
        mc_results = mc_results[argidx, :, :]
        formal_error = formal_error[argidx, :, :]
        spectral_mask = spectral_mask[argidx, :]
        snr_postfit = snr_postfit[argidx]
        tweights = tweights[argidx,:]
        
        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            # for i in range(1, 2):
            (
                tmp_result,
                ppxf_reddening[i],
                ppxf_bestfit[i, :],
                optimal_template1[i, :],
                optimal_template2[i, :],
                tmp_mc_result,
                tmp_formal_error,
                spectral_mask[i, :],
                snr_postfit[i],
                tweights[i,:],
            ) = run_twocomp_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                bias,
                goodPixels_ppxf,
                config["UMOD"]["MOM"],
                config["UMOD"]["ADEG"],
                config["UMOD"]["MDEG"],
                config["UMOD"]["REDDENING"],
                config["UMOD"]["DOCLEAN"],
                logLam,
                offset,
                velscale_ratio,
                nsims,
                nbins,
                i,
                [ntemplates_c1,ntemplates_c2],
                optimal_template_comb,
            )

            ppxf_result[i,0,:config['UMOD']['MOM']] = tmp_result[0]
            ppxf_result[i,1,:config['UMOD']['MOM']] = tmp_result[1]
            mc_results[i,0,:config['UMOD']['MOM']] = tmp_mc_result #needs to be done properly
            mc_results[i,1,:config['UMOD']['MOM']] = tmp_mc_result #needs to be done properly
            formal_error[i,0,:config['UMOD']['MOM']] = tmp_formal_error[0]
            formal_error[i,1,:config['UMOD']['MOM']] = tmp_formal_error[1]

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
    idx_error = np.where(np.isnan(ppxf_result[:, 0]) == True)[0]
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

    #calculate the flux, mean age, metal, alpha of each template
    template_weights = np.zeros((nbins,2,5))
    for h in range(0, nbins):

        #first calculate normalised weights per component
        nw1 = np.squeeze(tweights[h,:ntemplates_c1] / np.nansum(tweights[h,:ntemplates_c1]))
        nw2 = np.squeeze(tweights[h,ntemplates_c1:] / np.nansum(tweights[h,ntemplates_c1:]))
        
        sel1 = np.squeeze(np.where(nw1 > 0)[0])
        sel2 = np.squeeze(np.where(nw2 > 0)[0])

        #weights        
        template_weights[h,0,0] = np.nansum(tweights[h,:ntemplates_c1])
        template_weights[h,1,0] = np.nansum(tweights[h,ntemplates_c1:])
        #flux
        #age
        template_weights[h,0,2] = np.nansum((logAge_tgrid[:ntemplates_c1]*nw1)[sel1])
        template_weights[h,1,2] = np.nansum((logAge_tgrid[ntemplates_c1:]*nw2)[sel2])
        #metal
        template_weights[h,0,3] = np.nansum((metal_tgrid[:ntemplates_c1]*nw1)[sel1])
        template_weights[h,1,3] = np.nansum((metal_tgrid[ntemplates_c1:]*nw2)[sel2])
        #alpha
        template_weights[h,0,4] = np.nansum((alpha_tgrid[:ntemplates_c1]*nw1)[sel1])
        template_weights[h,1,4] = np.nansum((alpha_tgrid[ntemplates_c1:]*nw2)[sel2])        

        #calculate mean weights



    # Save stellar kinematics to file
    save_ppxf(
        config,                         
        ppxf_result,
        ppxf_reddening,
        mc_results,
        formal_error,
        ppxf_bestfit,
        logLam,
        goodPixels_ppxf,
        optimal_template1,
        optimal_template2,
        logLam_template,
        npix,
        spectral_mask,
        optimal_template_comb,
        bin_data,
        snr_postfit,                
        template_weights,
    )

    # Return

    return None
