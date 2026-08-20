import functools
import logging
import os

import numpy as np
import scipy.spatial.distance as dist
from astropy.io import fits
from printStatus import printStatus
from scipy.spatial import cKDTree
from powerbin import PowerBin

"""
PURPOSE:
  This file contains a collection of functions necessary to PowerBin the data.
  The PowerBinning makes use of the algorithm from Cappellari 2025
  (https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.1432C/abstract)
"""


# Power binning following Cappellari (2025)
def fun_capacity(index, signal=None, noise=None, covar=0.00):
        """Calculates (S/N)^2 for a bin from its pixel indices."""
        # Standard S/N formula for uncorrelated noise
        sn = np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))
        # Example for correlated noise (see full example file for details):
        sn /= 1 + covar * np.log10(len(index))
        return sn**2


def generateSpatialBins(config, cube):
    """
    This function applies the PowerBinning algorithm of Cappellari (2025) to the data. 
    Spaxels which were masked are excluded
    from the PowerBinning, but are assigned a negative BIN_ID, with the
    absolute value of the BIN_ID corresponding to the nearest PowerBin that
    satisfies the minimum SNR threshold.  All results are saved in a dedicated
    table to provide easy means of matching spaxels and bins.
    """
    # Generate the bins
    printStatus.running("Defining the Power bins")
    logging.info("Defining the Power bins")

    # Read maskfile
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    mask = fits.open(maskfile)[1].data.MASK
    idxUnmasked = np.where(mask == 0)[0]
    idxMasked = np.where(mask == 1)[0]

    xy = np.column_stack([cube["x"][idxUnmasked], cube["y"][idxUnmasked]])

    # Pass a function for the SNR calculation to the PowerBinning algorithm,
    # in order to account for spatial correlations in the noise
    capacity_spec_fun = functools.partial(
        fun_capacity, covar=config["SPATIAL_BINNING"]["COVARIANCE"],
        signal=cube["signal"][idxUnmasked], noise=cube["noise"][idxUnmasked]
    )

    try:
        # Do the binning
        pow = PowerBin(xy, capacity_spec_fun, config["SPATIAL_BINNING"]["TARGET_SNR"]**2)
        binNum = pow.bin_num
        xNode = pow.xybin[:, 0]
        yNode = pow.xybin[:,1]
        sn = np.sqrt(pow.bin_capacity)
        nPixels = pow.npix

        printStatus.updateDone("Defining the bins")
        print("             " + str(np.max(binNum) + 1) + " bins generated!")
        logging.info(str(np.max(binNum) + 1) + " Power bins generated!")

    # Handle common exceptions
    except ValueError as e:
        # Sufficient SNR and no binning needed
        if str(e) == "All pixels have enough S/N and binning is not needed":
            printStatus.updateWarning("Defining the Power bins")
            print(
                "             "
                + "The PowerBinning routine of Cappellari (2025) returned the following error:"
            )
            print("             " + str(e))
            printStatus.warning("Analysis will continue without PowerBinning!")
            print(
                "             "
                + str(len(idxUnmasked))
                + " spaxels will be treated as PowerBins."
            )

            logging.warning(
                "Defining the Power bins failed. The PowerBinning routine of Cappellari  "
                + "(2025) returned the following error: \n"
                + str(e)
            )
            logging.info(
                "Analysis will continue without PowerBinning! "
                + str(len(idxUnmasked))
                + " spaxels will be treated as PowerBins."
            )

            binNum, xNode, yNode, sn, nPixels = noBinning(
                cube["x"], cube["y"], cube["snr"], idxUnmasked
            )

        # Any uncaught exceptions, causing the galaxy to be skipped
        else:
            printStatus.updateFailed("Defining the Power bins")
            print(
                "The PowerBinning routine of Cappellari(2025) returned the following error: \n"
                + str(e)
            )
            logging.error(
                "Defining the Power bins failed. The PowerBinning routine of Cappellari "
                + "(2025) returned the following error: \n"
                + str(e)
            )
            return "SKIP"

    # Find the nearest Power-bin for the pixels outside the Voronoi region
    binNum_outside = find_nearest_powerbin(
        cube["x"], cube["y"], idxMasked, xNode, yNode
    )

    # Generate extended binNum-list:
    #   Positive binNum (including zero) indicate the Power-bin of the spaxel (for unmasked spaxels)
    #   Negative binNum indicate the nearest Power-bin of the spaxel (for masked spaxels)
    ubins = np.unique(binNum)
    nbins = len(ubins)
    binNum_long = np.zeros(len(cube["x"]))
    binNum_long[:] = np.nan
    binNum_long[idxUnmasked] = binNum
    binNum_long[idxMasked] = -1 * binNum_outside

    # Save bintable: data for *ALL* spectra inside and outside of the PowerBinned region!
    save_table(
        config,
        cube["x"],
        cube["y"],
        cube["signal"],
        cube["snr"],
        binNum_long,
        ubins,
        xNode,
        yNode,
        sn,
        nPixels,
        cube["pixelsize"],
        cube["wcshdr"],
    )

    return None


def noBinning(x, y, snr, idx_inside):
    """
    In case no PowerBinning is required/possible, treat spaxels in the input
    data as Power bins, in order to continue the analysis.
    """
    binNum = np.arange(0, len(idx_inside))
    xNode = x[idx_inside]
    yNode = y[idx_inside]
    sn = snr[idx_inside]
    nPixels = np.ones(len(idx_inside))

    return (binNum, xNode, yNode, sn, nPixels)


def find_nearest_powerbin(x, y, idx_outside, xNode, yNode):
    """
    Find the nearest PowerBin for each spaxel that does not satisfy the minimum SNR threshold.
    
    Args:
    - x (array): x-coordinates of all spaxels
    - y (array): y-coordinates of all spaxels
    - idx_outside (array): indices of spaxels which do not satisfy the minimum SNR threshold
    - xNode (array): x-coordinates of the Power bins
    - yNode (array): y-coordinates of the Power bins
    
    Returns:
    - closest (array): array of indices representing the nearest PowerBin for each spaxel
    """
    # Create an array of pixel coordinates
    pix_coords = np.column_stack((x[idx_outside], y[idx_outside]))
    # Create an array of bin coordinates
    bin_coords = np.column_stack((xNode, yNode))

    # Build a KDTree from the bin coordinates
    tree = cKDTree(bin_coords)
    # Query the nearest bin for each pixel
    closest = tree.query(pix_coords, k=1, workers=-1)[1]

    return closest

def save_table(
    config,
    x,
    y,
    signal,
    snr,
    binNum_new,
    ubins,
    xNode,
    yNode,
    sn,
    nPixels,
    pixelsize,
    wcshdr,
):
    """
    Save all relevant information about the Power-binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins.

    Args:
        config (dict): Configuration settings.
        x (ndarray): X-coordinates of the spaxels.
        y (ndarray): Y-coordinates of the spaxels.
        signal (ndarray): Flux values of the spaxels.
        snr (ndarray): Signal-to-noise ratio values of the spaxels.
        binNum_new (ndarray): Array of bin IDs for each spaxel.
        ubins (ndarray): Unique bin IDs.
        xNode (ndarray): X-coordinates of the bin nodes.
        yNode (ndarray): Y-coordinates of the bin nodes.
        sn (ndarray): Signal-to-noise ratio values of the bins.
        nPixels (ndarray): Number of spaxels in each bin.
        pixelsize (float): Size of each pixel.
        wcshdr (Header): WCS header information.

    Returns:
        None
    """
    outfits_table = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    xshape = x.shape
    yshape = y.shape
    printStatus.running("size of the input array=" + str(xshape) + str(yshape))
    # Expand data to spaxel level
    xNode_new = np.zeros(len(x))
    yNode_new = np.zeros(len(x))
    sn_new = np.zeros(len(x))
    nPixels_new = np.zeros(len(x))
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_new))[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        sn_new[idx] = sn[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode_new))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode_new))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn_new))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels_new))

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "TABLE"

    # create empty imageHDU with wcs header info
    imghdu = fits.ImageHDU(data=None, header=wcshdr)

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, tbhdu, imghdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    logging.info("Wrote PowerBin table: " + outfits_table)
