import logging
import os

import extinction
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ngistPipeline.readData import der_snr as der_snr
from printStatus import printStatus


# ======================================
# Routine to set DEBUG mode
# ======================================
def set_debug(cube, xext, yext):
    logging.info(
        "DEBUG mode is activated. Instead of the entire cube, only one line of spaxels is used."
    )
    cube["x"] = cube["x"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["y"] = cube["y"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["snr"] = cube["snr"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["signal"] = cube["signal"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["noise"] = cube["noise"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]

    cube["spec"] = cube["spec"][:, int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["error"] = cube["error"][:, int(yext / 2) * xext : (int(yext / 2) + 1) * xext]

    return cube


# ======================================
# Helper routine from PHANGS DAP
# ======================================
def reshape_extintion_curve(extinction_curve, cube):
    extra_dims = cube.ndim - extinction_curve.ndim
    new_shape = extinction_curve.shape + (1,) * extra_dims
    reshaped_extinction_curve = extinction_curve.reshape(new_shape)
    return reshaped_extinction_curve


# ======================================
# Routine to load MUSE-cubes
# ======================================
def readCube(config):
    loggingBlanks = (len(os.path.splitext(os.path.basename(__file__))[0]) + 33) * " "

    # Read MUSE-cube
    printStatus.running("Reading the MUSE-WFM cube")
    logging.info("Reading the MUSE-WFM cube: " + config["GENERAL"]["INPUT"])

    # Get shape from header to trim wavelength before loading full cube (saves memory)
    with fits.open(config["GENERAL"]["INPUT"], memmap=True, lazy_load_hdus=True) as hdu:
        if len(hdu) == 1:
            ihdu = 0
            printStatus.running("data in first HDU")
        else:
            ihdu = 1

        hdr = hdu[ihdu].header
        # Shape (nwave, ny, nx) from FITS NAXIS
        s = (hdr["NAXIS3"], hdr["NAXIS2"], hdr["NAXIS1"])
        wcshdr = WCS(hdr).to_header()

    # Compute wavelength and trim index before loading data
    if "CD3_3" not in hdr.keys():
        cdelt2 = hdr["CDELT3"]
        cdelt3 = hdr["CDELT3"]
    else:
        cdelt2 = hdr["CD2_2"]
        cdelt3 = hdr["CD3_3"]
    wave_full = hdr["CRVAL3"] + (np.arange(s[0])) * cdelt3
    wave_full = wave_full / (1 + config["GENERAL"]["REDSHIFT"])
    lmin = config["READ_DATA"]["LMIN_TOT"]
    lmax = config["READ_DATA"]["LMAX_TOT"]
    idx = np.where(np.logical_and(wave_full >= lmin, wave_full <= lmax))[0]

    # Read only the wavelength slice to avoid holding full cube in memory
    with fits.open(config["GENERAL"]["INPUT"], memmap=True, lazy_load_hdus=True) as hdu:
        data = hdu[ihdu].data
        data_slice = np.asarray(data[idx, :, :], dtype=np.float64)
        spec = np.reshape(data_slice, [len(idx), s[1] * s[2]])

        # Read the variance spectra if available. Otherwise estimate with der_snr
        if len(hdu) >= 3:
            logging.info("Reading the error (variance) spectra from the cube")
            stat = hdu[2].data
            stat_slice = np.asarray(stat[idx, :, :], dtype=np.float64)
            espec = np.reshape(stat_slice, [len(idx), s[1] * s[2]])
        else:
            logging.info(
                "No error (variance) extension found. Estimating the variance spectra with the der_snr algorithm"
            )
            noise_per_spaxel = der_snr.der_snr_2d(spec)
            espec = np.broadcast_to(
                noise_per_spaxel.reshape(1, -1), spec.shape
            ).copy()

    wave = wave_full[idx]

    # Correct spectra for Galactic extinction (taken from PHANGS DAP)
    if config["READ_DATA"]["EBmV"] is not None:
        Rv = 3.1
        Av = Rv * config["READ_DATA"]["EBmV"]
        ones = np.ones_like(wave)
        extinction_curve = extinction.apply(extinction.ccm89(wave, Av, Rv), ones)
        reshaped_extinction_curve = reshape_extintion_curve(
            extinction_curve, spec
        )
        np.divide(spec, reshaped_extinction_curve, out=spec)
        np.divide(espec, reshaped_extinction_curve, out=espec)
    # else: spec and espec unchanged

    # Getting the spatial coordinates
    origin = [
        float(config["READ_DATA"]["ORIGIN"].split(",")[0].strip()),
        float(config["READ_DATA"]["ORIGIN"].split(",")[1].strip()),
    ]
    xaxis = (np.arange(s[2]) - origin[0]) * cdelt2 * 3600.0
    yaxis = (np.arange(s[1]) - origin[1]) * cdelt2 * 3600.0
    x, y = np.meshgrid(xaxis, yaxis)
    x = np.reshape(x, [s[1] * s[2]])
    y = np.reshape(y, [s[1] * s[2]])
    pixelsize = cdelt2 * 3600.0
    logging.info(
        "Extracting spatial information:\n"
        + loggingBlanks
        + "* Spatial coordinates are centred to "
        + str(origin)
        + "\n"
        + loggingBlanks
        + "* Spatial pixelsize is "
        + str(pixelsize)
    )

    logging.info(
        "Shortening spectra to the wavelength range from "
        + str(config["READ_DATA"]["LMIN_TOT"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_TOT"])
        + "A."
    )

    # Computing the SNR per spaxel
    idx_snr = np.where(
        np.logical_and(
            wave >= config["READ_DATA"]["LMIN_SNR"],
            wave <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]
    signal = np.nanmedian(spec[idx_snr, :], axis=0)
    noise = np.sqrt(np.nanmedian(espec[idx_snr, :], axis=0))
    snr = np.nanmedian(spec[idx_snr, :] / np.sqrt(espec[idx_snr, :]), axis=0)
    logging.info(
        "Computing the signal-to-noise ratio in the wavelength range from "
        + str(config["READ_DATA"]["LMIN_SNR"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_SNR"])
        + "A."
    )

    # Propagate data-unit metadata from input (BUNIT) for downstream FITS/HDF5 products
    bunit = hdr.get("BUNIT")
    if bunit is not None:
        bunit = str(bunit).strip()

    # Storing everything into a structure
    cube = {
        "x": x,
        "y": y,
        "wave": wave,
        "spec": spec,
        "error": espec,
        "snr": snr,
        "signal": signal,
        "noise": noise,
        "pixelsize": pixelsize,
        "wcshdr": wcshdr,
        "bunit": bunit,
    }

    # Constrain cube to one central row if switch DEBUG is set
    if config["READ_DATA"]["DEBUG"] == True:
        cube = set_debug(cube, s[2], s[1])

    printStatus.updateDone(
        "Done reading " + str(len(cube["x"])) + " spectra from the MUSE-WFM cube"
    )

    logging.info(
        "Finished reading the MUSE cube! Read a total of "
        + str(len(cube["x"]))
        + " spectra!"
    )

    return cube
