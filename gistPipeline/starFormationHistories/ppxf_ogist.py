import glob
import logging
import os
import time
import extinction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def plot_ppxf(pp ,x, i,outfig_ppxf, snrCubevar=-99, snrResid=-99, goodpixelsPre=[], norm=True,mean_results=''):
    #foutine to plot first and final pPXF fit
    fig = plt.figure(i, figsize=(13, 3.0))

    #plot second figure
    ax2 = plt.subplot(111)


    if norm == True:
        median_norm = np.nanmedian(pp.galaxy[pp.goodpixels])
    else:
        median_norm = 1

    stars_bestfit = pp.bestfit / median_norm
    bestfit_shown = pp.bestfit / median_norm
    galaxy = pp.galaxy / median_norm
    resid = galaxy - stars_bestfit
    goodpixels = pp.goodpixels
    
    ll, rr = np.min(x), np.max(x)
    
    sig3 = np.percentile(abs(resid[goodpixels]), 99.73)
    bestfit_shown = bestfit_shown[goodpixels[0] : goodpixels[-1] + 1]
    mx = 2.49
    mn = -0.49
    plt.plot(x, galaxy, 'black', linewidth=0.5)
    plt.plot(x[goodpixels], resid[goodpixels], 'd',
                color='LimeGreen', mec='LimeGreen', ms=1)
    
    if len(goodpixelsPre) > 0:
        w = np.flatnonzero(np.diff(goodpixels) > 1)
        for wj in w:
            a, b = goodpixels[wj : wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightpink')
            plt.plot(x[a : b + 1], resid[a : b + 1], 'green', linewidth=0.5,alpha=0.5)
        for k in goodpixels[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightpink', linewidth=0.5)

        #repeat square lines with  pp_step1
            w = np.flatnonzero(np.diff(goodpixelsPre) > 1)
        for wj in w:
            a, b = goodpixelsPre[wj : wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightgray')
        for k in goodpixelsPre[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightgray', linewidth=0.5)
    else:
        w = np.flatnonzero(np.diff(goodpixels) > 1)
        for wj in w:
            a, b = goodpixels[wj : wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightgray')
            plt.plot(x[a : b + 1], resid[a : b + 1], 'green', linewidth=0.5, alpha=0.5)
        for k in goodpixels[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightgray', linewidth=0.5)
    
    plt.plot(x[goodpixels], goodpixels*0, '.k', ms=1)
    plt.plot(x, stars_bestfit, 'red', linewidth=0.5)
    ax2.set(xlabel='wavelength [Ang]', ylabel='Flux [normalised]')
    ax2.set(ylim=(mn,mx))
    ax2.tick_params(direction='in', which='both') 
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    #plot output

    nmom = np.max(pp.moments)

    if nmom == 2:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}")+\
        (f", S/N Residual = {snrResid:.1f}")
    if nmom == 4:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}")+\
        (f", S/N Residual = {snrResid:.1f}")        
    if nmom == 6:            
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}, ")+\
        (f"h5 = {pp.sol[4]:.3f}, h6 = {pp.sol[5]:.3f}")+\
        (f", S/N Residual = {snrResid:.1f}")   
    
    if len(mean_results) > 0:
        plotText += (f", Age [] = {mean_results[0][0]:.2f}, [M/H] = {mean_results[0][1]:.2f}")+\
        (", [alpha/Fe] = ")+(f"{mean_results[0][2]:.2f}")

    plt.text(0.01,0.95, plotText, fontsize=10, ha='left', va='top',transform=ax2.transAxes, backgroundcolor='white')
    plt.savefig(outfig_ppxf, bbox_inches='tight', pad_inches=0.3)
    plt.close()


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
        nsims,
        logAge_grid,
        metal_grid,
        alpha_grid,
        config,
        doplot,
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
            nsims,
            logAge_grid,
            metal_grid,
            alpha_grid,
            config,
            doplot,
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
    alpha_grid,
    config,
    doplot,
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    printStatus.progressBar(i, nbins, barLength=50)

    do_try = True
    if do_try == True:
    #try:

        if do_try == True:
            
            snr_prefit = np.nanmedian(log_bin_data/log_bin_error)
        
            ################ 3 ##################
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
        w_row   = np.array([np.reshape(weights, ncomb)])

        #plotting output
        if doplot == True:

            # check if figure  folder exists, otherwise
            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"],'oGIST_figFit_SFH')
            if os.path.exists(outfigDir) == False:
                printStatus.running('Creating directory for pPXF figures:' + outfigDir)
                os.mkdir(outfigDir)
            
            outfigFile_step3 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_"+str(i)+"_ogist.pdf"))

            # Calculate mean age, metallicity and alpha
            #mean_results_step1 = mean_agemetalalpha(w_row_step1, 10**logAge_grid, metal_grid, alpha_grid, 1)
            #calculate mean age, metallicity, and alpha step 1
            mean_results_step3 = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, 1)

            if fixed[0] == True:
                pp.sol[0:4] = start
            #produce plots
            tmp3 = plot_ppxf(pp,np.exp(logLam),i,outfigFile_step3,snrCubevar=snr_prefit,snrResid=snr_postfit,\
                             goodpixelsPre=pp.goodpixels,mean_results=mean_results_step3)

        mc_results = {
            "w_row_MC_iter": np.nan,
            "w_row_MC_mean": np.nan,
            "w_row_MC_err": np.nan,
            "mean_results_MC_iter": np.nan,
            "mean_results_MC_mean":  np.nan,
            "mean_results_MC_err":  np.nan
        }

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            1,
        )

 

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
    if config['SFH']['MC_PPXF'] > 0:
        cols.append(fits.Column(name='AGE_MC',   format='D', array=mean_result_MC_mean[:,0]))
        cols.append(fits.Column(name='METAL_MC', format='D', array=mean_result_MC_mean[:,1]))
        cols.append(fits.Column(name='ALPHA_MC', format='D', array=mean_result_MC_mean[:,2]))
        cols.append(fits.Column(name='ERR_AGE_MC',  format='D', array=mean_result_MC_err[:,0]))
        cols.append(fits.Column(name='ERR_METAL_MC',format='D', array=mean_result_MC_err[:,1]))
        cols.append(fits.Column(name='ERR_ALPHA_MC',format='D', array=mean_result_MC_err[:,2]))

    if config["SFH"]["FIXED"] == False:
        cols.append(fits.Column(name="V", format="D", array=ppxf_result[:, 0]))
        cols.append(fits.Column(name="SIGMA", format="D", array=ppxf_result[:, 1]))
        if np.any(ppxf_result[:, 2]) != 0:
            cols.append(fits.Column(name="H3", format="D", array=ppxf_result[:, 2]))
        if np.any(ppxf_result[:, 3]) != 0:
            cols.append(fits.Column(name="H4", format="D", array=ppxf_result[:, 3]))
        if np.any(ppxf_result[:, 4]) != 0:
            cols.append(fits.Column(name="H5", format="D", array=ppxf_result[:, 4]))
        if np.any(ppxf_result[:, 5]) != 0:
            cols.append(fits.Column(name="H6", format="D", array=ppxf_result[:, 5]))

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
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh-weights.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh-weights.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols = []
    cols.append(fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row))
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
    noise = np.ones((npix,nbins))
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
                    1,
                    EBV_init,
                    logLam,
                    config['SFH']['MC_PPXF'],
                    logAge_grid,
                    metal_grid,
                    alpha_grid,
                    config,
                    True,
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
            w_row_MC_iter[i,:,:] = ppxf_tmp[i][5]["w_row_MC_iter"]
            w_row_MC_mean[i,:] = ppxf_tmp[i][5]["w_row_MC_mean"]
            w_row_MC_err[i,:] = ppxf_tmp[i][5]["w_row_MC_err"]
            mean_results_MC_iter[i,:,:] = ppxf_tmp[i][5]["mean_results_MC_iter"]
            mean_results_MC_mean[i,:]  = ppxf_tmp[i][5]["mean_results_MC_mean"]
            mean_results_MC_err[i,:]  = ppxf_tmp[i][5]["mean_results_MC_err"]
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
        w_row_MC_iter = w_row_MC_iter[argidx,:,:]
        w_row_MC_mean = w_row_MC_mean[argidx,:]
        w_row_MC_err  = w_row_MC_err[argidx,:]
        mean_results_MC_iter = mean_results_MC_iter[argidx,:,:]
        mean_results_MC_mean = mean_results_MC_mean[argidx,:]
        mean_results_MC_err  = mean_results_MC_err[argidx,:]
        formal_error = formal_error[argidx,:]
        spectral_mask = spectral_mask[argidx,:]
        snr_postfit = snr_postfit[argidx]
        EBV = EBV[argidx]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        #for i in range(0, nbins):
        #runbin = [1,882,1873,1949]  
        runbin = [441, 457, 471, 476, 501, 521, 522, 545, 551,1325,1417,1458,1479,1494,1499]
        for i in runbin:                        
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
                1,
                EBV_init,
                logLam,
                config['SFH']['MC_PPXF'],
                logAge_grid,
                metal_grid,
                alpha_grid,
                config,
                True,
            )
            w_row_MC_iter[i,:,:] = mc_results_i["w_row_MC_iter"]
            w_row_MC_mean[i,:] = mc_results_i["w_row_MC_mean"]
            w_row_MC_err[i,:] = mc_results_i["w_row_MC_err"]
            mean_results_MC_iter[i,:,:] = mc_results_i["mean_results_MC_iter"]
            mean_results_MC_mean[i,:] = mc_results_i["mean_results_MC_mean"]
            mean_results_MC_err[i,:] = mc_results_i["mean_results_MC_err"]
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
        1,
        snr_postfit,
        EBV,
    )

    # Return
    return None
