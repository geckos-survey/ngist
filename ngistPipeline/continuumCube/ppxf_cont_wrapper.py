import logging
import os
import time

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import extinction
import shutil
import tempfile

from astropy.io import fits
from astropy.stats import biweight_location
from joblib import Parallel, delayed, dump, load
from packaging import version
from ppxf.ppxf import ppxf
from printStatus import printStatus
from tqdm import tqdm

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

import warnings
warnings.filterwarnings("ignore")

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE:
  This module creates a continuum and line-only cube.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""

def plot_ppxf_cont(pp, x, i, outfig_ppxf, snrCubevar=-99, snrResid=-99,
                  goodpixelsPre=[], norm=False, EBV=None,
                  poly_type='apoly'):

    mpl.rcParams['path.simplify'] = False
    mpl.rcParams['path.simplify_threshold'] = 0.0

    fig, (ax2, axpoly, axdust) = plt.subplots(
        3, 1, figsize=(13, 4.8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1],
                     'hspace': 0})

    if norm == True:
        median_norm = np.nanmedian(pp.galaxy[pp.goodpixels])
    else:
        median_norm = 1

    stars_bestfit = pp.bestfit
    galaxy = pp.galaxy
    resid = galaxy - stars_bestfit
    goodpixels = pp.goodpixels

    ll, rr = np.min(x), np.max(x)

    sig3 = np.percentile(abs(resid[goodpixels]), 99.73)

    if np.nanmax(stars_bestfit) > 2.59:
        mx = 3.49
    else:
        mx = 2.49
    mn = -0.49

    # Residuals on the full wavelength grid, excluding masked pixels
    resid_plot = np.full(len(resid), np.nan, dtype=float)
    resid_plot[goodpixels] = resid[goodpixels]

    # Main spectrum panel
    ax2.axhline(0.0, color='lightgray', linewidth=0.8,
                antialiased=False, zorder=0)

    resid_plot = np.full(len(resid), np.nan, dtype=float)
    resid_plot[goodpixels] = resid[goodpixels]

    # Stepped spectrum, residual, and best-fit model
    ax2.plot(x, galaxy, color='black', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=2)

    ax2.plot(x, resid_plot, color='LimeGreen', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=2)

    ax2.plot(x, stars_bestfit, color='red', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=3)

    if len(goodpixelsPre) > 0:

        # Current masked regions: pink, with green residuals
        padded = np.r_[-1, goodpixels, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightpink')

            ax2.plot(x[left:right + 1], resid[left:right + 1],
                     color='green', linewidth=0.1,
                     drawstyle='steps-mid', alpha=0.5,
                     antialiased=False,
                     solid_joinstyle='miter',
                     solid_capstyle='butt', zorder=2)

        for k in goodpixels[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightpink', linewidth=0.5,
                     antialiased=False)

        # Previous masked regions: grey only
        padded = np.r_[-1, goodpixelsPre, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightgray')

        for k in goodpixelsPre[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightgray', linewidth=0.5,
                     antialiased=False)

    else:

        # Current masked regions: grey, with green residuals
        padded = np.r_[-1, goodpixels, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightgray')

            ax2.plot(x[left:right + 1], resid[left:right + 1],
                     color='green', linewidth=0.1,
                     drawstyle='steps-mid', alpha=0.5,
                     antialiased=False,
                     solid_joinstyle='miter',
                     solid_capstyle='butt', zorder=2)

        for k in goodpixels[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightgray', linewidth=0.5,
                     antialiased=False)

    ax2.set_ylabel('Flux [normalised]')
    ax2.set_ylim(mn, mx)
    ax2.tick_params(direction='in', which='both')
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Polynomial panel
    if poly_type == 'mpoly':
        polynomial = pp.mpoly
        poly_min = 0.7
        poly_max = 1.3
        reference_value = 1.0
        poly_label = 'm-poly'

    elif poly_type == 'apoly':
        polynomial = pp.apoly
        poly_min = -0.55
        poly_max = 0.55
        reference_value = 0.0
        poly_label = 'a-poly'

    else:
        raise ValueError("poly_type must be either 'mpoly' or 'apoly'")

    # If no polynomial was fitted, use a constant reference curve
    if polynomial is None or np.size(polynomial) == 0:
        polynomial = np.full(len(x), reference_value, dtype=float)

    if EBV == 0:
        poly_min = np.nanmin(polynomial)
        poly_max = np.nanmax(polynomial)
        padding = 0.25 * max(poly_max - poly_min, 1.0)
        poly_min -= padding
        poly_max += padding

    axpoly.plot(x, polynomial, color='orchid', linewidth=0.8,
                antialiased=False)

    axpoly.axhline(reference_value, color='black', linestyle='--',
                   linewidth=0.7, antialiased=False, zorder=0)

    axpoly.set_ylabel(poly_label)
    axpoly.set_ylim(poly_min, poly_max)
    axpoly.set_xlabel('wavelength [Ang]')
    axpoly.tick_params(direction='in', which='both')
    axpoly.minorticks_on()
    axpoly.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Dust attenuation panel
    if EBV is not None and np.isfinite(EBV) and EBV >= 0:

        rv = 4.05
        av = rv * float(EBV)

        # extinction.calzetti00 returns A_lambda in magnitudes
        a_lambda = extinction.calzetti00(x, av, rv)

        # Multiplicative factor applied to the model flux
        dust_factor = extinction.apply(
            a_lambda, np.ones_like(x, dtype=float))

        axdust.axhline(1.0, color='black', linestyle='--',
                       linewidth=0.7, antialiased=False)

        axdust.fill_between(x, dust_factor, 1.0,
                            color='mistyrose', alpha=0.35)

        axdust.plot(x, dust_factor, color='firebrick',
                    linewidth=0.8, antialiased=False,
                    label=r'Calzetti dust factor')

        dust_min = np.nanmin(dust_factor)

        #axdust.set_ylim(max(0.0, dust_min - 0.03), 1.03)
        axdust.set_ylim(0., 1.15)
    else:
        axdust.set_ylim(0.95, 1.05)

    axdust.set_ylabel('dust factor')
    axdust.set_xlabel('wavelength [Ang]')
    axdust.tick_params(direction='in', which='both')
    axdust.minorticks_on()
    axdust.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if EBV is not None:
        axdust.legend(loc='best', fontsize=8, frameon=False)

    # add print statements
    nmom = np.max(pp.moments)

    if nmom == 2:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}") + \
                   (f", S/N_res = {snrResid:.1f}")
        if EBV is not None:
            plotText += f", E(B-V) = {EBV:.2f}"

    if nmom == 4:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, "
                    f"h4 = {pp.sol[3]:.3f}") + \
                   (f", S/N_res = {snrResid:.1f}")
        if EBV is not None:
            plotText += f", E(B-V) = {EBV:.2f}"


    if nmom == 6:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, "
                    f"h4 = {pp.sol[3]:.3f}, h5 = {pp.sol[4]:.3f}, "
                    f"h6 = {pp.sol[5]:.3f}") + \
                   (f", S/N_res = {snrResid:.1f}")
        if EBV is not None:
            plotText += f", E(B-V) = {EBV:.2f}"


    ax2.text(0.01, 0.95, plotText, fontsize=9, ha='left', va='top',
             transform=ax2.transAxes, backgroundcolor='white')

    plt.savefig(outfig_ppxf, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def clip_outliers(galaxy, bestfit, mask):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    while True:
        scale = galaxy[mask] @ bestfit[mask]/np.sum(bestfit[mask]**2)
        resid = scale*bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3*err
        if np.array_equal(mask, ok_old):
            break
            
    return mask

def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    np.seterr(all='ignore') # to avoid getting a lot of warnings in zerodivide

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


def run_ppxf_firsttime(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    adeg,
    mdeg,
    velscale_ratio,
):
    """
    Call PPXF for first time to get optimal template
    """

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
        degree=adeg,
        vsyst=offset,
        mdegree=mdeg,
        velscale_ratio=velscale_ratio,
    )

    normalized_weights = pp.weights / np.sum( pp.weights )
    
    optimal_template   = np.zeros((templates.shape[0],1))
    nonzero_weights = np.shape(np.where(normalized_weights > 0)[0])[0]
    optimal_template_set = np.zeros( [templates.shape[0], nonzero_weights])
    printStatus.running('Number of Templates with non-zero weights ' +str(nonzero_weights))
    
    count_nonzero = 0
    for j in range(0, templates.shape[1]):
        optimal_template[:,0] = optimal_template[:,0] + templates[:,j]*normalized_weights[j]
        if normalized_weights[j] > 0:
            optimal_template_set[:,count_nonzero] = templates[:,j]
            count_nonzero += 1

    return optimal_template, optimal_template_set

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels_premask,
    goodPixels_dust,
    goodPixels,
    nmoments,
    adeg,
    mdeg,
    doclean,
    logLam,
    offset,
    velscale_ratio,
    ntemplates,
    nsims,
    nbins,
    i,
    optimal_template_in,
    EBV_init,
    config,
    doplot,    
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics.
    """
    
    try:
        if len(optimal_template_in) > 1:

            # normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

            #calculate the snr before the fit (may be used for bias)
            snr_prefit = np.nanmedian(log_bin_data/log_bin_error)
            
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels_premask,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

            goodPixels_preclip = goodPixels
            # Find a proper estimate of the noise
            noise_orig = biweight_location(log_bin_error[goodPixels_premask])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_premask] - pp_step1.bestfit[goodPixels_premask])

            # calculate SNR postfit step 1
            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_premask]/noise_est)
            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 2 ##################
            # Second step (formely done with pPXF CLEAN)
            # switch to mask instead of goodpixels
            mask0 = logLam > 0
            mask0[:] = False
            mask0[goodPixels] = True
            mask = mask0.copy()
            
            if doclean == True:
                # Now use new function to clip outliers
                mask = clip_outliers(log_bin_data, pp_step1.bestfit, mask)
                # Add clipped pixels to the original masked emission lines regions and repeat the fit
                mask &= mask0

            ################ 3 ##################
            # Third step - Only fit dust, no polynomials allowed
            #create a mask for dust specifically
            mask_dust = np.zeros_like(mask, dtype=bool)
            mask_dust[goodPixels_dust] = True
            mask_dust &= mask # Keep only pixels good in both masks

            # create the dust model
            Rv = 4.05
            Av_init = 4.05 * EBV_init            
            component_step3 = [0] *  np.prod(optimal_template_in.shape[1:])
            component_true_step3 = np.array(component_step3) == 0
            dust = [{"start": [Av_init], "bounds": [[0, 8]], "component": component_true_step3}]

            # fit only for dust
            pp_step3 = ppxf(
                optimal_template_in, 
                log_bin_data, 
                noise_new, 
                velscale, 
                lam=np.exp(logLam), 
                mask=mask_dust,
                degree=-1, 
                mdegree=-1,
                vsyst=offset, 
                velscale_ratio=velscale_ratio,
                moments=nmoments, 
                start=start, 
                plot=False, 
                dust = dust, 
                component = component_step3, 
                regul=0, 
                quiet=True,
            )

            # Save dust values
            Av = pp_step3.dust[0]["sol"][0]
            EBV = Av/Rv
            component_step4 = [0]*ntemplates
            component_true_step4 = np.array(component_step4) == 0

            # apply the dust correction if keyword is set:
            if config["KIN"]["DUST_CORR"] == True:
                dust_step4 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step4, 
                         "fixed":[True]}]
            else:
                dust_step4 = None
            
            ################ 4 ##################
            # Fourth step: Last Call PPXF - use all templates, get best-fit

            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                mask=mask,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                component=component_step4,
                dust=dust_step4,                
            )

        # update goodpixels again
        goodPixels = pp.goodpixels

        # make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # define goodPixels over SNR range for final SNR
        goodPixels_SNR_range = goodPixels[
            (np.exp(logLam[goodPixels]) >= config["READ_DATA"]["LMIN_SNR"])
            & (np.exp(logLam[goodPixels]) <= config["READ_DATA"]["LMAX_SNR"])]
        
        # Calculate the true S/N from the residual over the SNR MIN MAX range
        noise_est = robust_sigma(pp.galaxy[goodPixels_SNR_range] - pp.bestfit[goodPixels_SNR_range])
        snr_postfit = np.nanmedian(pp.galaxy[goodPixels_SNR_range]/noise_est)

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum(pp.weights)
        optimal_template = np.zeros(templates.shape[0])
        for j in range(0, templates.shape[1]):
            optimal_template = (
                optimal_template + templates[:, j] * normalized_weights[j]
            )

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        #plotting output
        if doplot == True:

            # check if figure  folder exists, otherwise
            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"],'FigFit_CONT')
            os.makedirs(outfigDir, exist_ok=True)
                                    
            outfigFile_step1 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_cont_bin_"+str(i)+"_step1.pdf"))
            outfigFile_step3 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_cont_bin_"+str(i)+"_step3.pdf"))
            outfigFile_step4 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_cont_bin_"+str(i)+"_step4.pdf"))

            #produce plots
            tmp_plot1 = plot_ppxf_cont(pp_step1,np.exp(logLam),i,outfigFile_step1,
                                      snrCubevar=snr_prefit,snrResid=snr_Resid1,
                                      poly_type='mpoly')
            tmp_plot3 = plot_ppxf_cont(pp_step3,np.exp(logLam),i,outfigFile_step3,
                                      snrCubevar=snr_prefit,snrResid=snr_Resid1, EBV=EBV,
                                      poly_type='mpoly')
            tmp_plot3 = plot_ppxf_cont(pp,np.exp(logLam),i,outfigFile_step4,
                                      snrCubevar=snr_prefit,snrResid=snr_postfit,
                                      goodpixelsPre=goodPixels_preclip, EBV=EBV,
                                      poly_type='mpoly')


        # Do MC-Simulations
        sol_MC = np.zeros((nsims, nmoments))
        mc_results = np.zeros(nmoments)
        for o in range(0, nsims):
            # Add noise to bestfit:
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit + np.random.normal(
                0, 1, len(log_bin_data)
            ) * np.std(log_bin_data[goodPixels] - pp.bestfit[goodPixels])

            mc = ppxf(
                templates,
                noisy_bestfit,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                bias=0.0,
            )
            sol_MC[o, :] = mc.sol[:]

        if nsims != 0:
            mc_results = np.nanstd(sol_MC, axis=0)

        # add normalisation factor back in main results
        pp.bestfit = pp.bestfit * median_log_bin_data

        return(
            pp.sol[:],
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
        printStatus.warning(f"An error occurred: {e}")
        return ( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def save_ppxf_hdf5(
    config,
    ppxf_bestfit,
    logLam,
    goodPixels,
    bin_data,
):
    """Saves all results to disk."""
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin_bestfit_cont.hdf5"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin_bestfit_cont.hdf5")

    with h5py.File(outfits_ppxf, "w") as f:
        # Save PPXF bestfit
        f.create_dataset("BESTFIT", data=ppxf_bestfit)

        # Save PPXF logLam
        f.create_dataset("LOGLAM", data=logLam)

        # Save PPXF goodpixels
        f.create_dataset("GOODPIX", data=goodPixels)

        # Save SPEC data
        f.create_dataset("SPEC", data=bin_data.T)

    printStatus.running("Writing complete: " + config["GENERAL"]["RUN_ID"] + "_kin_bestfit_cont.hdf5")
    logging.info("Wrote: " + outfits_ppxf)
    
    
def createContinuumCube(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the nGIST conventions.
    """
    # Read data from file
    infile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5"
    printStatus.running("Reading: " + config["GENERAL"]["RUN_ID"] + "_bin_spectra.hdf5")
    
    # Open the HDF5 file
    with h5py.File(infile, 'r') as f:
        
        # Read the data from the file
        logLam = f["LOGLAM"][:]
        idx_lam = np.where(
        np.logical_and(
            np.exp(logLam) > config["CONT"]["LMIN"],
            np.exp(logLam) < config["CONT"]["LMAX"],
        )
        )[0]

        bin_data = f["SPEC"][:][idx_lam, :]
        bin_err = f["ESPEC"][:][idx_lam, :]
        velscale = f.attrs["VELSCALE"]
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

    # Read LSF information

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "CONT")  # added input of module

    # Prepare templates
    velscale_ratio = 2
    logging.info("Using full spectral library for PPXF")
    (
        templates,
        lamRange_spmod,
        logLam_template,
        ntemplates,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config["CONT"]["LMIN"],
        config["CONT"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        "CONT",
    )[
        :4
    ]
    templates = templates.reshape((templates.shape[0], ntemplates))

    # check that template wavelength is larger than requested fit range otherwise stop
    if (lamRange_spmod[0] >= config["CONT"]["LMIN"]) or (lamRange_spmod[1] <= config["CONT"]["LMAX"]):
        logging.info("Template wavelength range needs to be larger than fitting range, exiting")
        printStatus.warning(
            "Template wavelength range needs to be larger than fitting range, exiting"
        )
        return

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0]) * C

    #check what type of noise should be passed on:
    if config["CONT"]["NOISE"] == 'variance': # use noise from cube 
        noise = bin_err  # already converted to noise, i.e. sqrt(variance)
    elif config["CONT"]["NOISE"] == 'constant': # use constant noise
        noise  = np.ones((npix,nbins))
        # while constant, the noise does need to be scaled to match the bin_err
        med_bin_err = np.nanmedian(bin_err, axis=0)
        noise *= med_bin_err

    # Initial guesses
    start = np.zeros((nbins, 2))
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin_guess.fits"
        )
        == True
    ):
        printStatus.done(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin_guess.fits' as initial guesses"
        )
        logging.info(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin_guess.fits' as initial guesses"
        )
        guess = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin_guess.fits"
        )[1].data
        start[:, 0] = guess.V
        start[:, 1] = guess.SIGMA
    else:
        # Use the same initial guess for all bins, as stated in MasterConfig
        printStatus.done(
            "Using V and SIGMA from the MasterConfig file as initial guesses"
        )
        logging.info("Using V and SIGMA from the MasterConfig file as initial guesses")
        start[:, 0] = 0.0
        start[:, 1] = config["CONT"]["SIGMA"]

    # Define goodpixels
    goodPixels_cont = _auxiliary.spectralMasking(config, config["KIN"]["SPEC_MASK"], logLam)

    # Check if a premask for the first step has been defined
    if 'SPEC_PREMASK' in config["KIN"]:
        goodPixels_premask_cont = _auxiliary.spectralMasking(config, config["KIN"]["SPEC_PREMASK"], logLam)
    else:
        goodPixels_premask_cont = _auxiliary.spectralMasking(config, config["KIN"]["SPEC_MASK"], logLam)
    
    if 'SPEC_DUSTMASK' in config["KIN"]:
        goodPixels_dust_cont = _auxiliary.spectralMasking(config, config["KIN"]["SPEC_DUSTMASK"], logLam)
    else:
        goodPixels_dust_cont = _auxiliary.spectralMasking(config, config["KIN"]["SPEC_MASK"], logLam)

    # Check if plot keyword is set:
    doplot = config["CONT"].get("PLOT", False)

    # Array to store results of ppxf
    ppxf_result = np.zeros((nbins, 6))
    ppxf_bestfit = np.zeros((nbins, npix))
    optimal_template = np.zeros((nbins, templates.shape[0]))
    mc_results = np.zeros((nbins, 6))
    formal_error = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)
    EBV = np.zeros(nbins)
    
    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    if (config["CONT"]["OPT_TEMP"] == "galaxy_single") or (config["CONT"]["OPT_TEMP"] == "galaxy_set"):
        comb_spec = np.nanmean(bin_data[:,:],axis=1)
        comb_espec = np.nanmean(bin_err[:,:],axis=1)

        optimal_template_out, optimal_template_set = run_ppxf_firsttime(
            templates,
            comb_spec,
            comb_espec,
            velscale,
            start[0,:],
            goodPixels_premask_cont,
            config["CONT"]["MOM"],
            offset,
            config["CONT"]["ADEG"],
            config["CONT"]["MDEG"],
            velscale_ratio,
        )

        # now define the optimal template that we'll use throughout
        if config["CONT"]["OPT_TEMP"] == 'galaxy_single':
            optimal_template_comb = optimal_template_out # single template
        if config["CONT"]["OPT_TEMP"] == 'galaxy_set':
            optimal_template_comb = optimal_template_set # selected set  from total galaxy fit
    else:
        optimal_template_comb = templates # all templates

     # ====================
    EBV_init = 0.1 # PHANGS value initial guess
    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        # Create a unique temporary directory for this run's memmaps
        memmap_parent = ("/scratch"
            if os.access("/scratch", os.W_OK)
            else config["GENERAL"]["OUTPUT"])

        memmap_folder = tempfile.mkdtemp(
            prefix=f"{config['GENERAL']['RUN_ID']}_cont_",
            dir=memmap_parent)

        # Dump the arrays and reload them as read-only memmaps
        templates_filename_memmap = os.path.join(memmap_folder, "templates_memmap.tmp")
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode="r")

        if config["CONT"]["OPT_TEMP"] == "default":
            optimal_template_comb = templates
        else:
            opt_temp_file = os.path.join(
                memmap_folder, "optimal_template_memmap.tmp"
            )
            dump(optimal_template_comb, opt_temp_file)
            optimal_template_comb = load(opt_temp_file, mmap_mode="r")

        bin_data_filename_memmap = os.path.join(memmap_folder, "bin_data_memmap.tmp")
        dump(bin_data, bin_data_filename_memmap)
        bin_data = load(bin_data_filename_memmap, mmap_mode="r")

        noise_filename_memmap = os.path.join(memmap_folder, "noise_memmap.tmp")
        dump(noise, noise_filename_memmap)
        noise = load(noise_filename_memmap, mmap_mode="r")

        # Define a function to encapsulate the work done in the loop
        def worker(chunk, templates):
            results = []
            for i in chunk:
                result = run_ppxf(
                    templates,
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    goodPixels_premask_cont,
                    goodPixels_dust_cont,
                    goodPixels_cont,
                    config["CONT"]["MOM"],
                    config["CONT"]["ADEG"],
                    config["CONT"]["MDEG"],
                    config["CONT"]["DOCLEAN"],
                    logLam,
                    offset,
                    velscale_ratio,
                    ntemplates,
                    0,
                    nbins,
                    i,
                    optimal_template_comb,
                    EBV_init,
                    config,
                    doplot,
                )
                results.append(result)
            return results
        
        # Use joblib to parallelize the work
        max_nbytes = "1M" # max array size before memory mapping is triggered
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)]
        parallel_configs = {
            "n_jobs": config["GENERAL"]["NCPU"],
            "max_nbytes": None,
            "return_as": "generator",
        }

        #ppxf_tmp = list(tqdm(Parallel(**parallel_configs)(delayed(worker)(chunk, templates) for chunk in chunks),
        #                total=len(chunks), desc="Processing chunks", ascii=" #", unit="chunk"))

        with Parallel(**parallel_configs) as parallel:
            ppxf_tmp = list(tqdm(
                parallel(delayed(worker)(chunk, templates) for chunk in chunks),
                total=len(chunks), desc="Processing chunks",
                ascii=" #", unit="chunk"
            ))
        
        # Flatten the results
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        for i in range(0, nbins):
            ppxf_result[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][0]
            ppxf_bestfit[i, :] = ppxf_tmp[i][1]
            optimal_template[i, :] = ppxf_tmp[i][2]
            mc_results[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][3]
            formal_error[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][4]
            spectral_mask[i, :] = ppxf_tmp[i][5]
            snr_postfit[i] = ppxf_tmp[i][6]
            EBV[i] = ppxf_tmp[i][7]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=False)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")

        # check if we need to run all bins or only a subset
        if 'DEBUG_BIN' in config["CONT"]:
            runbin = np.array(config["CONT"]["DEBUG_BIN"])
            # replace config keyword with string to save it in header later
            config["CONT"]["DEBUG_BIN"] = str(runbin)
        else:
            runbin = np.arange(0, nbins)
        
        for i in runbin:
            (
                ppxf_result[i, : config["CONT"]["MOM"]],
                ppxf_bestfit[i, :],
                optimal_template[i, :],
                mc_results[i, : config["CONT"]["MOM"]],
                formal_error[i, : config["CONT"]["MOM"]],
                spectral_mask[i, :],
                snr_postfit[i],
                EBV[i],
            ) = run_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                goodPixels_premask_cont,
                goodPixels_dust_cont,
                goodPixels_cont,
                config["CONT"]["MOM"],
                config["CONT"]["ADEG"],
                config["CONT"]["MDEG"],
                config["CONT"]["DOCLEAN"],
                logLam,
                offset,
                velscale_ratio,
                ntemplates,
                0,
                nbins,
                i,
                optimal_template_comb,
                EBV_init,
                config,
                doplot,                
            )

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

    # Save to file
    if "DEBUG_BIN" in config["CONT"]:
        # replace config keyword with string to save it in header later
        config["CONT"]["DEBUG_BIN"] = str(config["CONT"]["DEBUG_BIN"])

    save_ppxf_hdf5(
        config,
        ppxf_bestfit,
        logLam,
        goodPixels_cont,
        bin_data,
    )

    if config["GENERAL"]["PARALLEL"] == True:
        templates._mmap.close()
        bin_data._mmap.close()
        noise._mmap.close()

        if config["CONT"]["OPT_TEMP"] != "default":
            optimal_template_comb._mmap.close()

        shutil.rmtree(memmap_folder)
    
    # Return
    return None