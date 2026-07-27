import logging
import os

import numpy as np
from astropy.io import fits
from printStatus import printStatus
from scipy.ndimage import gaussian_filter, label

def generateSpatialMask(config, cube):
    """
    Default implementation of the spatialMasking module.

    This function masks defunct spaxels, rejects spaxels with a signal-to-noise ration below a given threshold, and
    masks spaxels according to a provided mask file. Finally, all masks are combined and saved.
    """

    # Mask defunct spaxels if needed
    maskedDefunct = maskDefunctSpaxels(
        cube,
        mask_nan=config["SPATIAL_MASKING"].get("MASK_NAN", True),
        mask_negative_median=config["SPATIAL_MASKING"].get("MASK_NEGATIVE_MEDIAN", True),
    )

    # Mask spaxels with SNR below threshold
    maskedSNR = applySNRThreshold(
        cube,
        cube["snr"], 
        cube["signal"], 
        config["SPATIAL_MASKING"]["MIN_SNR"], 
        config["SPATIAL_MASKING"].get("THRESHOLD_METHOD", "isophote"), 
        config["SPATIAL_MASKING"].get("SMOOTH_SIGMA", 5.0),
    )

    # Mask spaxels according to spatial mask file
    maskedMask = applyMaskFile(config, cube)

    # Create combined mask
    combinedMaskIdx = np.where(
        np.logical_or.reduce(
            (maskedDefunct == True, maskedSNR == True, maskedMask == True)
        )
    )[0]
    combinedMask = np.zeros(len(cube["snr"]), dtype=bool)
    combinedMask[combinedMaskIdx] = True
    logging.info(
        "Combined mask: " + str(len(combinedMaskIdx)) + " spaxels are rejected."
    )

    # Save mask to file
    saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config)

    # Return
    return None

def maskDefunctSpaxels(cube, mask_nan=True,mask_negative_median=True):
    """
    Mask spaxels containing NaNs and/or spaxels with non-positive median flux.
    """
    spec = cube["spec"]

    bad_nan = np.any(np.isnan(spec), axis=0) if mask_nan else np.zeros(spec.shape[1], dtype=bool)
    bad_median = np.nanmedian(spec, axis=0) <= 0.0 if mask_negative_median else np.zeros(spec.shape[1], dtype=bool)

    logging.info(f"NaN spaxels: {np.sum(bad_nan)}")
    logging.info(f"Negative-median spaxels: {np.sum(bad_median)}")

    bad = np.logical_or(bad_nan, bad_median)

    idx_bad = np.where(bad)[0]
    idx_good = np.where(~bad)[0]

    logging.info(
        "Masking defunct spaxels: " + str(len(idx_bad)) + " spaxels are rejected."
    )

    masked = np.ones(len(cube["snr"]), dtype=bool)
    masked[idx_good] = False

    return masked

def build_spatial_grid(cube, decimals=10):
    """
    Build the 2D spatial indexing from flattened spaxel coordinates.
    Returns x/y coordinate grids and the 1D->2D index mapping.
    """
    x = np.round(cube["x"], decimals)
    y = np.round(cube["y"], decimals)

    x_vals = np.unique(x)
    y_vals = np.unique(y)

    ix = np.searchsorted(x_vals, x)
    iy = np.searchsorted(y_vals, y)

    return x_vals, y_vals, ix, iy


def values_to_2d(values, x_vals, y_vals, ix, iy, fill_value=np.nan):
    """
    Place a flattened spaxel vector onto its 2D spatial grid.
    """
    arr_2d = np.full((y_vals.size, x_vals.size), fill_value, dtype=float)
    arr_2d[iy, ix] = values
    return arr_2d

def applySNRThreshold(cube, snr, signal, min_snr, threshold_method, smooth_sigma):
    """
    Mask those spaxels that are above the isophote level with a mean
    signal-to-noise ratio of MIN_SNR.
    """
    if threshold_method == "isophote_gist":
        # this is the old gist way of creating a mask, but does not work with low S/N
        idx_snr = np.where(np.abs(snr - min_snr) < 2.0)[0]
        meanmin_signal = np.mean(signal[idx_snr])
        idx_inside = np.where(signal >= meanmin_signal)[0]
        idx_outside = np.where(signal < meanmin_signal)[0]

    if threshold_method == "isophote":
        if min_snr == 0:
            # No spatial masking requested.
            inside_mask = np.ones(signal.shape, dtype=bool)
            outside_mask = np.zeros(signal.shape, dtype=bool)
        else:

            # Use only valid pixels to estimate the flux corresponding to the requested S/N.
            valid = np.isfinite(signal) & np.isfinite(snr)
            snr_valid = snr[valid]
            signal_valid = signal[valid]

            # Estimate the isophote from the 100 spaxels with S/N closest to the requested value.
            N_SNR_SAMPLES = min(100, snr_valid.size)
            idx = np.argsort(np.abs(snr_valid - min_snr))[:N_SNR_SAMPLES]
            meanmin_signal = np.mean(signal_valid[idx])

            # Create a mask from the corresponding flux isophote.
            inside_mask = signal >= meanmin_signal
            outside_mask = signal < meanmin_signal

        idx_inside = np.where(inside_mask)
        idx_outside = np.where(outside_mask)

    if threshold_method == "isophote_smooth":

        if min_snr == 0:
            # No spatial masking requested.
            inside_mask = np.ones(signal.shape, dtype=bool)
            outside_mask = np.zeros(signal.shape, dtype=bool)

        else:
            # Use only valid spaxels to estimate the flux corresponding to the requested S/N.
            valid_1d = np.isfinite(signal) & np.isfinite(snr)
            snr_valid = snr[valid_1d]
            signal_valid = signal[valid_1d]

            # Estimate the isophote from the 100 spaxels with S/N closest to the requested value.
            N_SNR_SAMPLES = min(100, snr_valid.size)
            idx = np.argsort(np.abs(snr_valid - min_snr))[:N_SNR_SAMPLES]
            meanmin_signal = np.mean(signal_valid[idx])

            # Reconstruct the 2D signal map.
            x_vals = np.unique(np.round(cube["x"], 10))
            y_vals = np.unique(np.round(cube["y"], 10))
            ix = np.searchsorted(x_vals, np.round(cube["x"], 10))
            iy = np.searchsorted(y_vals, np.round(cube["y"], 10))

            signal_2d = np.full((y_vals.size, x_vals.size), np.nan)
            signal_2d[iy, ix] = signal

            # Smooth the 2D flux map.
            mask_2d = np.isfinite(signal_2d)
            signal_filled = np.where(mask_2d, signal_2d, 0.0)

            smoothed_signal = gaussian_filter(signal_filled, sigma=smooth_sigma)
            smoothed_mask = gaussian_filter(mask_2d.astype(float), sigma=smooth_sigma)

            with np.errstate(divide="ignore", invalid="ignore"):
                smoothed_signal = smoothed_signal / smoothed_mask
            smoothed_signal[smoothed_mask == 0] = np.nan

            # Threshold the smoothed map.
            inside_mask_2d = smoothed_signal >= meanmin_signal

            # Keep only the largest connected region above the threshold.
            labels, nlab = label(inside_mask_2d)
            if nlab > 0:
                counts = np.bincount(labels.ravel())
                counts[0] = 0  # ignore background
                main_label = np.argmax(counts)
                inside_mask_2d = labels == main_label
            else:
                inside_mask_2d = np.zeros_like(inside_mask_2d, dtype=bool)

            # Map back to the original 1D spaxel order.
            inside_mask = inside_mask_2d[iy, ix]
            outside_mask = ~inside_mask

        idx_inside = np.where(inside_mask)[0]
        idx_outside = np.where(outside_mask)[0]

    if threshold_method == "actual":
        idx_inside = np.where(snr >= min_snr)[0]
        idx_outside = np.where(snr < min_snr)[0]

    if threshold_method == "actual_smooth":

        x_vals, y_vals, ix, iy = build_spatial_grid(cube)

        # Reconstruct the 2D S/N map and smooth it.
        snr_2d = values_to_2d(snr, x_vals, y_vals, ix, iy)
        mask_2d = np.isfinite(snr_2d)
        snr_filled = np.where(mask_2d, snr_2d, 0.0)

        smoothed_snr = gaussian_filter(snr_filled, sigma=smooth_sigma)
        smoothed_mask = gaussian_filter(mask_2d.astype(float), sigma=smooth_sigma)

        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed_snr = smoothed_snr / smoothed_mask
        smoothed_snr[smoothed_mask == 0] = np.nan

        # Threshold the smoothed map.
        inside_mask_2d = smoothed_snr >= min_snr

        # Keep only the largest connected region above the threshold.
        labels, nlab = label(inside_mask_2d)
        if nlab > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0  # ignore background
            main_label = np.argmax(counts)
            inside_mask_2d = labels == main_label
        else:
            inside_mask_2d = np.zeros_like(inside_mask_2d, dtype=bool)

        # Map back to the original 1D spaxel ordering.
        inside_mask = inside_mask_2d[iy, ix]
        outside_mask = ~inside_mask

        idx_inside = np.where(inside_mask)[0]
        idx_outside = np.where(outside_mask)[0]  

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange(len(snr))
        idx_outside = np.array([], dtype=np.int64)

    logging.info(
        "Masking low signal-to-noise spaxels: "
        + str(len(idx_outside))
        + " spaxels are rejected."
    )

    masked = np.zeros(len(snr), dtype=bool)
    masked[idx_inside] = False
    masked[idx_outside] = True

    return masked


def applyMaskFile(config, cube):
    """
    Select those spaxels that are unmasked in the input masking file.
    """

    if (
        config["SPATIAL_MASKING"]["MASK"] == False
        or config["SPATIAL_MASKING"]["MASK"] == None
    ):
        logging.info("No maskfile specified.")
        idxGood = np.arange(len(cube["snr"]))
        idxBad = np.array([], dtype=np.int64)

    else:
        maskfile = os.path.join(
            os.path.dirname(config["GENERAL"]["INPUT"]),
            config["SPATIAL_MASKING"]["MASK"],
        )

        if os.path.isfile(maskfile) == True:
            hdu = fits.open(maskfile)
            if len(hdu) == 1:
                mask = hdu[0].data
            else:
                mask = hdu[1].data
            s = np.shape(mask)
            mask = np.reshape(mask, s[0] * s[1])

            idxGood = np.where(mask == 0)[0]
            idxBad = np.where(mask == 1)[0]

            logging.info(
                "Masking spaxels according to maskfile: "
                + str(len(idxBad))
                + " spaxels are rejected."
            )

        elif os.path.isfile(maskfile) == False:
            logging.info("No maskfile found at " + maskfile)
            idxGood = np.arange(len(cube["snr"]))
            idxBad = np.array([], dtype=np.int64)

    masked = np.zeros(len(cube["snr"]), dtype=bool)
    masked[idxGood] = False
    masked[idxBad] = True

    return masked


def saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config):
    """Save the mask to disk."""
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with output data
    # This is an integer array! 0 means unmasked, 1 means masked!
    cols = []
    cols.append(
        fits.Column(
            name="MASK", format="I", array=np.array(combinedMask, dtype=np.int32)
        )
    )
    cols.append(
        fits.Column(
            name="MASK_DEFUNCT",
            format="I",
            array=np.array(maskedDefunct, dtype=np.int32),
        )
    )
    cols.append(
        fits.Column(
            name="MASK_SNR", format="I", array=np.array(maskedSNR, dtype=np.int32)
        )
    )
    cols.append(
        fits.Column(
            name="MASK_FILE", format="I", array=np.array(maskedMask, dtype=np.int32)
        )
    )
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "MASKFILE"

    # Create HDU list and write to file
    tbhdu.header["COMMENT"] = "Value 0  -->  unmasked"
    tbhdu.header["COMMENT"] = "Value 1  -->  masked"
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")
    logging.info("Wrote mask file: " + outfits)

    return None
