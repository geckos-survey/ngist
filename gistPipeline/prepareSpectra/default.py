import logging
import os

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed, dump, load, wrap_non_picklable_objects
from ppxf.ppxf_util import log_rebin
from printStatus import printStatus


def prepSpectra(config, cube):
    """
    This function performs the following tasks:
     * Apply spatial bins to linear spectra; Save these spectra to disk
     * Log-rebin all spectra, regardless of whether the spaxels are masked or not; Save all spectra to disk
     * Apply spatial bins to log-rebinned spectra; Save these spectra to disk
    """

    # Read maskfile
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    mask = fits.open(maskfile)[1].data.MASK
    idxUnmasked = np.where(mask == 0)[0]
    idxMasked = np.where(mask == 1)[0]

    # Read binning pattern
    tablefile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    binNum = fits.open(tablefile)[1].data.BIN_ID[idxUnmasked]

    # Apply spatial bins to linear spectra
    bin_data, bin_error, bin_flux = applySpatialBins(
        binNum,
        cube["spec"][:, idxUnmasked],
        cube["error"][:, idxUnmasked],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        "lin",
    )
    # Save spatially binned spectra
    saveBinSpectra(
        config,
        bin_data,
        bin_error,
        config["PREPARE_SPECTRA"]["VELSCALE"],
        cube["wave"],
        "lin",
    )

    # Log-rebin spectra
    log_spec, log_error, logLam = log_rebinning(config, cube)
    # Save all log-rebinned spectra
    saveAllSpectra(
        config, log_spec, log_error, config["PREPARE_SPECTRA"]["VELSCALE"], logLam
    )

    # Apply bins to log spectra
    bin_data, bin_error, bin_flux = applySpatialBins(
        binNum,
        log_spec[:, idxUnmasked],
        log_error[:, idxUnmasked],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        "log",
    )
    # Save spatially binned spectra
    saveBinSpectra(
        config,
        bin_data,
        bin_error,
        config["PREPARE_SPECTRA"]["VELSCALE"],
        logLam,
        "log",
    )

    return None


def log_rebinning(config, cube):
    """
    Logarithmically rebin spectra and error spectra.
    """
    # Log-rebin the spectra
    printStatus.running("Log-rebinning the spectra")
    log_spec, logLam = run_log_rebinning(
        cube["spec"],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        len(cube["x"]),
        cube["wave"],
    )
    printStatus.updateDone("Log-rebinning the spectra", progressbar=True)
    logging.info("Log-rebinned the spectra")

    # Log-rebin the error spectra
    printStatus.running("Log-rebinning the error spectra")
    log_error, _ = run_log_rebinning(
        cube["error"],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        len(cube["x"]),
        cube["wave"],
    )
    printStatus.updateDone("Log-rebinning the error spectra", progressbar=True)
    logging.info("Log-rebinned the error spectra")

    return (log_spec, log_error, logLam)


def run_log_rebinning(
    binned_data, velocity_scale, num_bins, wavelength, chunk_size=1000
):
    """
    Perform log-rebinning on the given binned_data.

    Args:
    - binned_data (ndarray): 2D array of shape (num_pixels, num_bins), representing the binned spectra
    - velocity_scale (float): Velocity scale for the log-rebinning
    - num_bins (int): Number of bins
    - wavelength (ndarray): 1D array of shape (num_pixels), representing the wavelength array
    - chunk_size (int, optional): Size of the chunks for processing. Defaults to 1000.

    Returns:
    - log_binned_data (ndarray): 2D array of shape (len(log_lam), num_bins), representing the log-rebinned data
    - log_lam (ndarray): 1D array representing the log-rebinned wavelength array
    """
    # Setup arrays
    wavelength_range = np.array([np.amin(wavelength), np.amax(wavelength)])

    # Perform log-rebinning for the first bin and initialize the log-rebinned data array
    ssp_new, log_lam, _ = log_rebin(
        wavelength_range, binned_data[:, 0], velscale=velocity_scale
    )
    log_binned_data = np.zeros([len(log_lam), num_bins])

    # Do log-rebinning for each chunk of bins
    for i in range(0, num_bins, chunk_size):
        for j in range(i, min(i + chunk_size, num_bins)):
            try:
                # Perform log-rebinning for the current bin
                ssp_new, _, _ = log_rebin(
                    wavelength_range, binned_data[:, j], velscale=velocity_scale
                )
                log_binned_data[:, j] = ssp_new
            except:
                # If an error occurs, set the log-rebinned data for the current bin to NaN
                log_binned_data[:, j] = np.zeros(len(log_lam))
                log_binned_data[:, j][:] = np.nan

    return (log_binned_data, log_lam)


def saveAllSpectra(config, log_spec, log_error, velscale, logLam):
    """Save all logarithmically rebinned spectra to file."""
    outfits_spectra = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_AllSpectra.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append(
        fits.Column(name="SPEC", format=str(len(log_spec)) + "D", array=log_spec.T)
    )
    cols.append(
        fits.Column(name="ESPEC", format=str(len(log_spec)) + "D", array=log_error.T)
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "SPECTRA"

    # Table HDU for LOGLAM
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = "LOGLAM"

    # Create HDU List and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header keywords
    fits.setval(outfits_spectra, "VELSCALE", value=velscale)
    fits.setval(outfits_spectra, "CRPIX1", value=1.0)
    fits.setval(outfits_spectra, "CRVAL1", value=logLam[0])
    fits.setval(outfits_spectra, "CDELT1", value=logLam[1] - logLam[0])

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.fits"
    )
    logging.info("Wrote: " + outfits_spectra)


def saveBinSpectra(config, log_spec, log_error, velscale, logLam, flag):
    """Save spatially binned spectra and error spectra are saved to disk."""
    outfile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])

    if flag == "log":
        outfits_spectra = outfile + "_BinSpectra.fits"
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.fits"
        )
    elif flag == "lin":
        outfits_spectra = outfile + "_BinSpectra_linear.fits"
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra_linear.fits"
        )

    npix = len(log_spec)

    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=log_spec.T))
    cols.append(fits.Column(name="ESPEC", format=str(npix) + "D", array=log_error.T))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BIN_SPECTRA"

    # Table HDU for LOGLAM
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = "LOGLAM"

    # Create HDU list and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    fits.setval(outfits_spectra, "VELSCALE", value=velscale)
    fits.setval(outfits_spectra, "CRPIX1", value=1.0)
    fits.setval(outfits_spectra, "CRVAL1", value=logLam[0])
    fits.setval(outfits_spectra, "CDELT1", value=logLam[1] - logLam[0])

    if flag == "log":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.fits"
        )
    elif flag == "lin":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra_linear.fits"
        )
    logging.info("Wrote: " + outfits_spectra)


def applySpatialBins(binNum, spec, espec, velscale, flag):
    """
    The constructed spatial binning scheme is applied to the spectra.
    """
    printStatus.running("Applying the spatial bins to " + flag + "-data")
    bin_data, bin_error, bin_flux = spatialBinning(binNum, spec, espec)
    printStatus.updateDone(
        "Applying the spatial bins to " + flag + "-data", progressbar=True
    )
    logging.info("Applied spatial bins to " + flag + "-data")

    return (bin_data, bin_error, bin_flux)


def spatialBinning(binNum, spec, error):
    """Spectra belonging to the same spatial bin are added."""
    ubins = np.unique(binNum)
    nbins = len(ubins)
    npix = spec.shape[0]
    bin_data = np.zeros([npix, nbins])
    bin_error = np.zeros([npix, nbins])
    bin_flux = np.zeros(nbins)

    for i in range(nbins):
        k = np.where(binNum == ubins[i])[0]
        valbin = len(k)
        if valbin == 1:
            av_spec = spec[:, k]
            av_err_spec = np.sqrt(error[:, k])
        else:
            av_spec = np.nansum(spec[:, k], axis=1)
            av_err_spec = np.sqrt(np.sum(error[:, k], axis=1))

        bin_data[:, i] = np.ravel(av_spec)
        bin_error[:, i] = np.ravel(av_err_spec)
        bin_flux[i] = np.mean(av_spec, axis=0)
        printStatus.progressBar(i + 1, nbins, barLength=50)

    return (bin_data, bin_error, bin_flux)


if __name__ == "__main__":

    import sys
    import time
    from importlib import reload

    from gistPipeline.initialise import _initialise
    from gistPipeline.readData import _readData
    from gistPipeline.spatialBinning import _spatialBinning
    from gistPipeline.spatialMasking import _spatialMasking

    _readData = reload(_readData)

    class dirPath(object):
        pass

    # set config and default dir file paths
    dirPath = dirPath()

    dirPath.configFile = (
        "/arc/home/thbrown/mauve/productTesting/IC3392/IC3392_test.yaml"
    )
    dirPath.defaultDir = (
        "/arc/home/thbrown/mauve/productTesting/IC3392/IC3392_test.defaultDir"
    )

    # dirPath.configFile = "/arc/home/thbrown/mauve/dev/gist-geckos-supplementary/gistTutorial/configFiles/MasterConfig.yaml"
    # dirPath.defaultDir = "/arc/home/thbrown/mauve/dev/gist-geckos-supplementary/gistTutorial/configFiles/defaultDir"

    # Read config
    config = _initialise.readMasterConfig(dirPath.configFile, 1)
    config = _initialise.addPathsToConfig(config, dirPath)

    # Check output directory
    _initialise.checkOutputDirectory(config)

    # Setup logfile
    _initialise.setupLogfile(config)

    # Check output directory
    _initialise.checkOutputDirectory(config)

    # Setup logfile
    _initialise.setupLogfile(config)
    sys.excepthook = _initialise.handleUncaughtException

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  P R E P A R A T I O N   M O D U L E S  - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - READ_DATA MODULE - - - -
    cube = _readData.readData_Module(config)

    # - - - - - SPATIAL MASKING MODULE - - - - -
    # _ = _spatialMasking.spatialMasking_Module(config, cube)

    # - - - - - SPATIAL BINNING MODULE - - - - -
    # _ = _spatialBinning.spatialBinning_Module(config, cube)

    # - - - - - PREPARE SPECTRA MODULE - - - - -

    prepSpectra(config, cube)
