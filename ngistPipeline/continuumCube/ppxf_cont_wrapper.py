import code
import logging
import os
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from joblib import Parallel, delayed, dump, load
from ppxf.ppxf import ppxf
from printStatus import printStatus
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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



def plot_ppxf_kin(pp ,x, i,outfig_ppxf, snrCubevar=-99, snrResid=-99, goodpixelsPre=[], norm=False):
    
    #routine to plot first and final pPXF fit
    fig = plt.figure(i, figsize=(13, 3.0))

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

    nmom = np.max(pp.moments)

    if nmom == 2:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}")+\
        (f", S/N Residual = {snrResid:.1f}")
    if nmom == 4:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, \
                    h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}")+\
                    (f", S/N Residual = {snrResid:.1f}")        
    if nmom == 6:            
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, \
                    h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}, ")+\
                    (f"h5 = {pp.sol[4]:.3f}, h6 = {pp.sol[5]:.3f}")+\
                    (f", S/N Residual = {snrResid:.1f}")   
            
    plt.text(0.01,0.95, plotText, fontsize=10, ha='left', va='top',transform=ax2.transAxes, backgroundcolor='white')
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
    goodPixels_step0,
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
            
            # Here add in the extra, 0th step to estimate the dust and print out the E(B-V) map
            # Call PPXF, using an extinction law, no polynomials.
            # First define the dust law (from cappellari 2023):
            component_step0 = [0] * np.prod(optimal_template_in.shape[1:])
            component_true_step0 = np.array(component_step0) == 0
            dust = [{"start": [EBV_init], "bounds": [[0, 8]], "component": component_true_step0}]

            pp_step0 = ppxf(optimal_template_in, log_bin_data, log_bin_error, velscale, lam=np.exp(logLam), 
                            goodpixels=goodPixels_step0,degree=-1, mdegree=-1, vsyst=offset, 
                            velscale_ratio=velscale_ratio,moments=nmoments, start=start, plot=False, 
                            dust = dust, component = component_step0, regul=0,quiet=True)

            # check which optimal template method is preferred. If default rederive optimal set from step 0
            if config["CONT"]["OPT_TEMP"] == 'default':

                # find non zero weights from step 0
                normalized_weights_step0 = pp_step0.weights / np.sum( pp_step0.weights )
                wNonzero_weights_step0 = np.where(normalized_weights_step0 > 0)[0] # where Nonzero
                nNonzero_weights_step0 = np.shape(wNonzero_weights_step0)[0] # number of Nonzero templates

                # prepare optimal template set
                optimal_template_set_step0 = np.zeros( [templates.shape[0], nNonzero_weights_step0])
                # combine non-zero templates into set to pass on step 1 and step 2
                for j in range(0, nNonzero_weights_step0):
                        optimal_template_set_step0[:,j] = templates[:,wNonzero_weights_step0[j]]

                # replace optimal template with set from step zero
                optimal_template_in = optimal_template_set_step0
                
            # Save dust values
            Rv = 4.05
            Av = pp_step0.dust[0]["sol"][0]
            EBV = Av/Rv

            # Define the components to be fit (True for all templates)
            component_step12 = [0]*(np.shape(optimal_template_in)[1])
            component_true_step12 = np.array(component_step12) == 0
            component_step3 = [0]*ntemplates
            component_true_step3 = np.array(component_step3) == 0

            # apply the dust correction if keyword is set:
            if config["CONT"]["DUST_CORR"] == True:
                # old approach --> remove extinction from spectra
                #log_bin_data_save = log_bin_data
                #log_bin_data_tmp = extinction.remove(extinction.calzetti00(np.exp(logLam), Av, Rv), log_bin_data)
                #median_log_bin_data_tmp = np.median(log_bin_data_tmp) # save number for later
                #log_bin_data = (log_bin_data_tmp/median_log_bin_data_tmp)
                #log_bin_error_tmp = extinction.remove(extinction.calzetti00(np.exp(logLam), Av, Rv), log_bin_error)
                #log_bin_error = (log_bin_error_tmp/np.median(log_bin_error_tmp))
                
                # new approach, fix dust in pPXF instead of normalising spectrum
                # fix dust in pPXF to best fit from step 0
                dust_step12 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step12, 
                                          "fixed": [True]}]

                dust_step3 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step3, 
                         "fixed":[True]}]
            else:
                dust_step12 = None
                dust_step3 = None

            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels_step0,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
                mdegree=mdeg,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                component=component_step12,
                dust=dust_step12,
            )

            goodPixels_preclip = goodPixels
            # Find a proper estimate of the noise
            noise_orig = biweight_location(log_bin_error[goodPixels_step0])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_step0] - pp_step1.bestfit[goodPixels_step0])

            # calculate SNR postfit step 1
            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_step0]/noise_est)
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
            # Third Call PPXF - use all templates, get best-fit

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
                component=component_step3,
                dust=dust_step3,                
            )

        # update goodpixels again
        goodPixels = pp.goodpixels

        # make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0
        
        # Calculate the true S/N from the residual the short version
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmedian(pp.galaxy[goodPixels]/noise_est)

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
            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"],'FigFit_CON')
            if os.path.exists(outfigDir) == False:
                printStatus.running('Creating directory for pPXF figures:' + outfigDir)
                os.mkdir(outfigDir)
                        
            outfigFile_step1 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_con_bin_"+str(i)+"_step1.pdf"))
            outfigFile_step3 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_con_bin_"+str(i)+"_step3.pdf"))

            #produce plots
            tmp_plot1 = plot_ppxf_kin(pp_step1,np.exp(logLam),i,outfigFile_step1,
                                      snrCubevar=snr_prefit,snrResid=snr_Resid1)
            tmp_plot3 = plot_ppxf_kin(pp,np.exp(logLam),i,outfigFile_step3,\
                                      snrCubevar=snr_prefit,snrResid=snr_postfit,
                                      goodpixelsPre=goodPixels_preclip)


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

        # apply the dust vector to bestfit if keyword set:
        #if config["CONT"]["DUST_CORR"] == True:
        #    pp.bestfit = extinction.apply(extinction.calzetti00(np.exp(logLam), Av, Rv), pp.bestfit) * \
        #                 median_log_bin_data_tmp

        # add normalisation factor back in main results
        pp.bestfit = pp.bestfit * median_log_bin_data
        if pp.reddening is not None:
            pp.reddening = pp.reddening * median_log_bin_data

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
    # we shouldn't use a bare except clause
    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)



def save_ppxf(
    config,
    ppxf_result,
    mc_results,
    formal_error,
    ppxf_bestfit,
    logLam,
    goodPixels,
    optimal_template,
    logLam_template,
    npix,
    spectral_mask,
    optimal_template_comb,
    bin_data,
):
    """Saves all results to disk."""
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-bestfit-cont.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits")

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

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["CONT"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["CONT"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["CONT"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["CONT"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["CONT"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, specHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    

def createContinuumCube(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the nGIST conventions.
    """
    # Read data from file
    infile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5"
    printStatus.running("Reading: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.hdf5")
    
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
    #check if a premask for step zero has been defined
    if 'SPEC_PREMASK' in config["CONT"]:
        #yes, load this premask file
        goodPixels_step0_con = _auxiliary.spectralMasking(config, config["CONT"]["SPEC_PREMASK"], logLam)
    else:
        #no, load this normal file
        goodPixels_step0_con = _auxiliary.spectralMasking(config, config["CONT"]["SPEC_MASK"], logLam)
    
    goodPixels_con = _auxiliary.spectralMasking(config, config["CONT"]["SPEC_MASK"], logLam)

    # Check if plot keyword is set:
    if 'PLOT' in config["CONT"]:
        doplot = True
    else:
        doplot = False

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
            goodPixels_step0_con,
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
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    goodPixels_step0_con,
                    goodPixels_con,
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
        parallel_configs = {"n_jobs": config["GENERAL"]["NCPU"], "max_nbytes": max_nbytes, "temp_folder": memmap_folder, "mmap_mode": "c", "return_as":"generator"}
        ppxf_tmp = list(tqdm(Parallel(**parallel_configs)(delayed(worker)(chunk, templates) for chunk in chunks),
                        total=len(chunks), desc="Processing chunks", ascii=" #", unit="chunk"))

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

        # Remove the memory-mapped files
        os.remove(templates_filename_memmap)
        os.remove(bin_data_filename_memmap)
        os.remove(noise_filename_memmap)

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
                goodPixels_step0_con,
                goodPixels_con,
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
    if 'DEBUG_BIN' in config["CONT"]:
        # replace config keyword with string to save it in header later
        config["CONT"]["DEBUG_BIN"] = str(config["CONT"]["DEBUG_BIN"])

    save_ppxf(
        config,
        ppxf_result,
        mc_results,
        formal_error,
        ppxf_bestfit,
        logLam,
        goodPixels_con,
        optimal_template,
        logLam_template,
        npix,
        spectral_mask,
        optimal_template_comb,
        bin_data,
    )

    # Return

    return None