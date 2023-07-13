#!/usr/bin/env python

import logging
import optparse
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import CubicSpline
from spectral_cube import SpectralCube
from astropy import units as u
from astropy.wcs import WCS

from gistPipeline.readData.MUSE_WFM import readCube
from gistPipeline.utils.wcs_utils import (diagonal_wcs_to_cdelt,
                                          strip_wcs_from_header)

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
    oldwcshdr = table_hdu[2].header.copy()

    # update WCS
    wcs = WCS(oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

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
        for i, name in enumerate(names):
            result[:, i] = np.array(hdu[1].data[name])

    elif module_id == "SFH":
        # Read results
        sfh_hdu = fits.open(os.path.join(outdir, rootname) + "_sfh.fits")
        result = np.zeros((len(ubins), 3))

        result[:, 0] = np.array(sfh_hdu[1].data.AGE)
        result[:, 1] = np.array(sfh_hdu[1].data.METAL)
        result[:, 2] = np.array(sfh_hdu[1].data.ALPHA)
        if (
            len(np.unique(result[:, 2])) == 1
        ):  # propose change to simply names = list(hdu[1].data.dtype.names) but not yet tested
            names = ["AGE", "METAL"]
        else:
            names = ["AGE", "METAL", "ALPHA"]

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
        image_hdu = fits.ImageHDU(image, header=newwcshdr, name=names[iterate])
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
    oldwcshdr = table_hdu[2].header.copy()

    # update WCS
    wcs = WCS(oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

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

        image_hdu = fits.ImageHDU(image, header=newwcshdr, name=line)
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
    oldwcshdr = table_hdu[2].header.copy()

    # update WCS
    wcs = WCS(oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

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

    names = list(hdu[1].data.dtype.names)

    result = np.zeros((len(ubins), len(names)))
    for i, name in enumerate(names):
        result[:, i] = np.array(hdu[1].data[name])

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
        image[i[::-1], j] = val

        # Transpose x and y to reorient the image correctly
        # im[ra, dec] = arr[col, row]
        image = image.T

        # make HDU
        image_hdu = fits.ImageHDU(image, header=newwcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(
        os.path.join(outdir, rootname)
        + "_"
        + module_id
        + "_"
        + RESOLUTION
        + "_maps.fits",
        overwrite=True,
    )
    hdu1.close()


def saveContLineCube(config):
    """
    saveContLineCubes _summary_

    Write continuum-only and line-only cubes to FITS files.

    Parameters
    ----------
    config : str, optional
        gistPipeline config
    """

    # read cube header - check extension contains WCS
    cubehdr = fits.getheader(config["GENERAL"]["INPUT"], ext=0)
    if "NAXIS1" not in cubehdr:
        cubehdr = fits.getheader(config["GENERAL"]["INPUT"], ext=1)
    elif "NAXIS1" not in cubehdr:
        cubehdr = fits.getheader(config["GENERAL"]["INPUT"], ext=2)

    NX = cubehdr["NAXIS1"]
    NY = cubehdr["NAXIS2"]

    inputCube = readCube(config)
    spectra_all = inputCube["spec"]
    linLam = inputCube["wave"]
    
    idx_lam = np.where(
        np.logical_and(linLam > config["KIN"]["LMIN"], linLam < config["KIN"]["LMAX"])
    )[0]
    spectra_all = spectra_all[idx_lam, :]
    linLam = linLam[idx_lam]

    # get PPXF best fit continuum from kinematics module
    ppxf_bestfit = fits.open(
        os.path.join(
            config["GENERAL"]["OUTPUT"],
            config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits",
        )
    )[1].data.BESTFIT

    # ABW get logLam from best fit (continuum/kinematics) module outputs ##:OLD:get logLam from Bin Spectra HDU
    logLam = fits.open(
        os.path.join(
            config["GENERAL"]["OUTPUT"],
            config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits",
        )
    )[2].data.LOGLAM

    # table HDU
    tablehdu = fits.open(
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )

    spaxID = np.array(tablehdu[1].data.ID)
    binID = np.array(tablehdu[1].data.BIN_ID)

    contCube = np.full([len(linLam), NY * NX], np.nan)
    lineCube = np.full([len(linLam), NY * NX], np.nan)

    idx_snr = np.where(
        np.logical_and(
            linLam >= config["READ_DATA"]["LMIN_SNR"],
            linLam <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]

    # loop over spaxels
    for s in spaxID:
        # bin ID of spaxel s
        binID_spax = binID[s]
        obsSpec_lin = spectra_all[:, s]
        obsSignal = np.nanmedian(obsSpec_lin[idx_snr])

        if binID_spax < 0:
            fitSpec_lin = np.zeros(len(obsSpec_lin))
        elif binID_spax >= 0:
            fitSpec = ppxf_bestfit[binID_spax, :]

            fitSpec_func = CubicSpline(np.exp(logLam), fitSpec, extrapolate=False)

            fitSpec_lin = fitSpec_func(linLam)
            fitSignal = np.nanmedian(fitSpec_lin[idx_snr])

            fitSpec_lin *= obsSignal / fitSignal

        # assign continuum fits and emission lines (obs - cont) to cube
        contCube[:, s] = fitSpec_lin
        lineCube[:, s] = obsSpec_lin - fitSpec_lin

    # spectral axes in observed wavelength frame
    # (cube is de-redshifted during read in by MUSE_WFM.py)
    cubehdr["NAXIS3"] = len(linLam)
    cubehdr["CRVAL3"] = linLam[0] * (1 + config["GENERAL"]["REDSHIFT"]) # 
    cubehdr["CRPIX3"] = 1
    cubehdr["CTYPE3"] = "AWAV"
    cubehdr["CUNIT3"] = "angstrom"
    
    # set the WCS keywords to CDELT standard format
    cdi_j_wcs = WCS(cubehdr)
    newcubehdr = strip_wcs_from_header(cubehdr)  # remove all WCS keys from header
    newcubehdr.update(
        diagonal_wcs_to_cdelt(cdi_j_wcs).to_header()
    )  # replace with CDELT standard keys
    
    # set the correct CDELT *after* CD3_3 has been removed, adjust to observed wavelength frame
    # as cube is de-redshifted during read in by MUSE_WFM.py
    newcubehdr["CDELT3"] = np.abs(np.diff(linLam * (1 + config["GENERAL"]["REDSHIFT"])))[0] * 1e-10 # A -> m

    # save line and continuum cubes
    # float32 preferred over float64 to save size and allow for conversion to hdf5 
    fn_suffix = ["CONT", "LINE"]
    for cube, name in zip([contCube, lineCube], fn_suffix):
        
        outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_KIN_{}cube.fits".format(name)
        )

        # make sure the pixel unit is saved in the new cube
        specCube = SpectralCube(data=np.float32(cube.reshape((len(linLam), NY, NX))) * u.Unit(cubehdr["BUNIT"]),
                                wcs=WCS(newcubehdr),
                                header=newcubehdr)
        
        specCube.write(outfits, overwrite=True)
        
