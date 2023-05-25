#!/usr/bin/env python

import optparse
import os
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")

def savefitsmaps(module_id, outdir=""):
    """
    savefitsmaps _summary_

    Parameters
    ----------
    module_id : _type_
        _description_
    outdir : str, optional
        _description_, by default ""
    """

    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir, rootname) + "_table.fits")
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X) * -1
    Y = np.array(table_hdu[1].data.Y)
    FLUX = np.array(table_hdu[1].data.FLUX)
    binNum_long = np.array(table_hdu[1].data.BIN_ID)
    ubins = np.unique(np.abs(np.array(table_hdu[1].data.BIN_ID)))
    pixelsize = table_hdu[0].header["PIXSIZE"]
    wcshdr = table_hdu[2].header

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    # Read Results
    if module_id == "KIN":
        
        # read results
        hdu = fits.open(os.path.join(outdir, rootname) + "_kin.fits")
        names = list(hdu[1].data.dtype.names)
        
        result = np.zeros((len(ubins), len(names)))
        result[:, 0] = np.array(hdu[1].data.V)
        result[:, 1] = np.array(hdu[1].data.SIGMA)
                
        for i, name in enumerate(names):
            result[:, i] = np.array(hdu[1].data[name])

    elif module_id == "SFH":
        # Read results
        sfh_hdu = fits.open(os.path.join(outdir, rootname) + "_sfh.fits")
        result = np.zeros((len(ubins), 3))
        result[:, 0] = np.array(sfh_hdu[1].data.AGE)
        result[:, 1] = np.array(sfh_hdu[1].data.METAL)
        result[:, 2] = np.array(sfh_hdu[1].data.ALPHA)
        if len(np.unique(result[:, 2])) == 1: # propose change to simply names = list(hdu[1].data.dtype.names) but not yet tested
            labellist = ["AGE", "METAL"]
        else:
            labellist = ["AGE", "METAL", "ALPHA"]
            names = labellist
    
    # Convert results to long version
    result_long = np.zeros((len(binNum_long), result.shape[1]))
    result_long[:, :] = np.nan
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_long))[0]
        result_long[idx, :] = result[i, :]
    result = result_long
    
    # result[:, 0] = result[:, 0] - np.nanmedian(result[:, 0]) [median subtraction on products]
    
    ####### Adding the ability to output maps as fits files
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])

    for iterate in range(0, len(names)):
        # Prepare main plot
        val = result[:, iterate]

        # Create image in pixels
        xmin = np.min(X)
        xmax = np.max(X)
        ymin = np.min(Y)
        ymax = np.max(Y)
        npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
        npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
        i = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
        j = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
        image = np.full((npixels_x, npixels_y), np.nan)
        # Reverse the i index to each row of the image
        # because ra increases West-East (right-left in image plane)
        image[i[::-1][idx_inside], j[idx_inside]] = val[idx_inside]
        # Transpose x and y because numpy uses arr[row, col] and FITS uses
        # im[ra, dec] = arr[col, row]
        image = image.T

        # make HDU
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(
        os.path.join(outdir, rootname) + "_" + module_id + "_maps.fits", overwrite=True
    )
    hdu1.close()


def savefitsmaps_GASmodule(module_id="GAS", outdir="", LEVEL="", AoNThreshold=4):
    """
    savefitsmaps_GASmodule _summary_

    Parameters
    ----------
    module_id : str, optional
        _description_, by default "GAS"
    outdir : str, optional
        _description_, by default ""
    LEVEL : str, optional
        _description_, by default ""
    AoNThreshold : int, optional
        _description_, by default 4
    """

    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Construct a mask for defunct spaxels
    mask = fits.open(os.path.join(outdir, rootname) + "_mask.fits")[1].data.MASK_DEFUNCT
    maskedSpaxel = np.array(mask, dtype=bool)

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir, rootname) + "_table.fits")
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X) * -1
    Y = np.array(table_hdu[1].data.Y)
    FLUX = np.array(table_hdu[1].data.FLUX)
    binNum_long = np.array(table_hdu[1].data.BIN_ID)
    ubins = np.unique(np.abs(binNum_long))
    pixelsize = table_hdu[0].header["PIXSIZE"]
    wcshdr = table_hdu[2].header

    maskedSpaxel = maskedSpaxel[idx_inside]

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    # Read Gandalf results
    if LEVEL == "SPAXEL":
        results = fits.open(os.path.join(outdir, rootname) + "_gas_SPAXEL.fits")[
            1
        ].data[~maskedSpaxel]
    elif LEVEL == "BIN":
        results = fits.open(os.path.join(outdir, rootname) + "_gas_BIN.fits")[1].data
    elif LEVEL == None:
        print("LEVEL keyword not set!")

    # Convert results to long version
    if LEVEL == "BIN":
        _, idxConvert = np.unique(np.abs(binNum_long), return_inverse=True)
        results = results[idxConvert]

    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])

    # Iterate over all lines
    for line in results.names:
        if line[-3:] == "AON":
            continue
        if line in ["EBmV_0", "EBmV_1"]:
            continue

        data = results[line]
        data_aon = results[line[:-2] + "_AON"]

        data[np.where(data_aon < AoNThreshold)[0]] = np.nan
        data[np.where(data == -1)[0]] = np.nan

        # [median subtraction on products]
        # if line.split("_")[-1] == "V":
        #     data = data - np.nanmedian(data)

        # Create image in pixels
        xmin = np.min(X)
        xmax = np.max(X)
        ymin = np.min(Y)
        ymax = np.max(Y)
        npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
        npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
        col = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
        row = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
        image = np.full((npixels_x, npixels_y), np.nan)

        # reverse the index to flip vertically
        # since WCS transformations - like FITS files - assume
        # that the origin is the lower left pixel of the image
        # (origin is in top left for numpy arrays)
        image[
            col[::-1][idx_inside][~maskedSpaxel], row[idx_inside][~maskedSpaxel]
        ] = data[idx_inside][~maskedSpaxel]

        # Transpose x and y because numpy uses arr[row, col] and FITS uses im[ra, dec] = arr[col, row]
        image = image.T

        image_hdu = fits.ImageHDU(image, header=wcshdr, name=line)
        # Append fits image
        hdu1.append(image_hdu)

    hdu1.writeto(
        os.path.join(outdir, rootname) + "_" + module_id + "_" + LEVEL + "_maps.fits",
        overwrite=True,
    )
    hdu1.close()


def savefitsmaps_LSmodule(module_id="LS", outdir="", RESOLUTION=""):
    """
    savefitsmaps_LSmodule _summary_

    Parameters
    ----------
    module_id : str, optional
        _description_, by default "LS"
    outdir : str, optional
        _description_, by default ""
    RESOLUTION : str, optional
        _description_, by default ""
    """
    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir, rootname) + "_table.fits")
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X) * -1
    Y = np.array(table_hdu[1].data.Y)
    FLUX = np.array(table_hdu[1].data.FLUX)
    binNum_long = np.array(table_hdu[1].data.BIN_ID)
    ubins = np.unique(np.abs(np.array(table_hdu[1].data.BIN_ID)))
    pixelsize = table_hdu[0].header["PIXSIZE"]
    wcshdr = table_hdu[2].header

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    # Read results
    if RESOLUTION == "ORIGINAL":
        hdu = fits.open(os.path.join(outdir, rootname) + "_ls_OrigRes.fits")
    elif RESOLUTION == "ADAPTED":
        hdu = fits.open(os.path.join(outdir, rootname) + "_ls_AdapRes.fits")
    result = np.zeros((len(ubins), 3))
    result[:, 0] = np.array(hdu[1].data.Hbeta_o)
    result[:, 1] = np.array(hdu[1].data.Fe5015)
    result[:, 2] = np.array(hdu[1].data.Mgb)

    labellist = ["Hbeta_o", "Fe5015", "Mgb"]

    # Convert results to long version
    result_long = np.zeros((len(binNum_long), result.shape[1]))
    result_long[:, :] = np.nan
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_long))[0]
        result_long[idx, :] = result[i, :]
    result = result_long
    
    # result[:, 0] = result[:, 0] - np.nanmedian(result[:, 0]) [median subtraction on products]
    
    ####### Adding the ability to output maps as fits files
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])
    names = labellist

    for iterate in range(0, len(names)):
        # Prepare main plot
        val = result[:, iterate]

        # Create image in pixels
        xmin = np.min(X)
        xmax = np.max(X)
        ymin = np.min(Y)
        ymax = np.max(Y)
        npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
        npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
        i = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
        j = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
        image = np.full((npixels_x, npixels_y), np.nan)

        # Reverse the i index to each row of the image
        # because ra increases West-East (right-left in image plane)
        image[i[::-1], j] = val

        # Transpose x and y to reorient the image correctly
        # im[ra, dec] = arr[col, row]
        image = image.T

        # make HDU
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(
        os.path.join(outdir, rootname) + "_" + module_id + "_" + RESOLUTION + "_maps.fits",
        overwrite=True,
    )
    hdu1.close()
