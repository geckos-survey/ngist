import glob
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from ngistPipeline._version import __version__

"""
This file contains a selection of functions that are needed at multiple locations in the framework. This includes
functions to print the status of the GIST to stdout, read the line-spread-function from file, and create spectral masks.

When developing user-defined modules, you can take advantage of these functions or simply include your own tools in the
module.
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


def getLSF(config, module_used):
    """
    Read the given LSF from file for the specified module.
    module_used: 'KIN', 'CONT', 'GAS', 'SFH', 'LS', or 'UMOD'
    to account for differing template sets for the same run.
    """
    # Read LSF of observation and templates and construct an interpolation function
    lsfDataFile = os.path.join(
        config["GENERAL"]["CONFIG_DIR"], config["GENERAL"]["LSF_DATA"]
    )
    if module_used == "KIN":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["KIN"]["LSF_TEMP"]
        )        
    elif module_used == "CONT":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["CONT"]["LSF_TEMP"]
        )
    elif module_used == "GAS":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["GAS"]["LSF_TEMP"]
        )
    elif module_used == "SFH":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["SFH"]["LSF_TEMP"]
        )
    elif module_used == "LS":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["LS"]["LSF_TEMP"]
        )
    elif module_used == "UMOD":
        lsfTempFile = os.path.join(
            config["GENERAL"]["CONFIG_DIR"], config["UMOD"]["LSF_TEMP"]
        )
    LSF = np.genfromtxt(lsfDataFile, comments="#")
    LSF[:, 0] = LSF[:, 0] / (1 + config["GENERAL"]["REDSHIFT"])
    LSF[:, 1] = LSF[:, 1] / (1 + config["GENERAL"]["REDSHIFT"])
    LSF[LSF[:, 1] / (1 + config["GENERAL"]["REDSHIFT"]) < 2.51, 1] = 2.54 * (
        1 + config["GENERAL"]["REDSHIFT"]
    )
    LSF_Data = interp1d(LSF[:, 0], LSF[:, 1], "linear", fill_value="extrapolate")
    LSF = np.genfromtxt(lsfTempFile, comments="#")
    LSF_Templates = interp1d(LSF[:, 0], LSF[:, 1], "linear", fill_value="extrapolate")
    return (LSF_Data, LSF_Templates)


def spectralMasking(config, file, logLam):
    """Mask spectral region in the fit."""
    # Read file
    mask = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], file), usecols=(0, 1)
    )
    maskComment = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], file), usecols=(2), dtype=str
    )
    goodPixels = np.arange(len(logLam))

    # In case there is only one mask
    if len(mask.shape) == 1 and mask.shape[0] != 0:
        mask = mask.reshape(1, 2)
        maskComment = maskComment.reshape(1)

    for i in range(mask.shape[0]):
        # Check for sky-lines
        if (
            maskComment[i] == "sky"
            or maskComment[i] == "SKY"
            or maskComment[i] == "Sky"
        ):
            mask[i, 0] = mask[i, 0] / (1 + config["GENERAL"]["REDSHIFT"])

        # Define masked pixel range
        minimumPixel = int(
            np.round(
                (np.log(mask[i, 0] - mask[i, 1] / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )
        maximumPixel = int(
            np.round(
                (np.log(mask[i, 0] + mask[i, 1] / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )

        # Handle border of wavelength range
        if minimumPixel < 0:
            minimumPixel = 0
        if maximumPixel < 0:
            maximumPixel = 0
        if minimumPixel >= len(logLam):
            minimumPixel = len(logLam) - 1
        if maximumPixel >= len(logLam):
            maximumPixel = len(logLam) - 1

        # Mark masked spectral pixels
        goodPixels[minimumPixel : maximumPixel + 1] = -1

    goodPixels = goodPixels[np.where(goodPixels != -1)[0]]

    return goodPixels


def addGISTHeaderComment(config):
    """
    Add a GIST header comment in all fits output files.
    """
    filelist = glob.glob(os.path.join(config["GENERAL"]["OUTPUT"], "*.fits"))

    for file in filelist:
        if "Generated with the nGIST pipeline" not in str(fits.getheader(file)):
            fits.setval(file, "COMMENT", value="", ext=0)
            fits.setval(
                file,
                "COMMENT",
                value="                Generated with the nGIST pipeline , V"
                + __version__
                + "                  ",
                ext=0,
            )
            fits.setval(
                file,
                "COMMENT",
                value="------------------------------------------------------------------------",
                ext=0,
            )
            fits.setval(
                file,
                "COMMENT",
                value=" Based on the GIST pipeline of Bittner et al.  ",
                ext=0,
            )
            fits.setval(
                file,
                "COMMENT",
                value="       analysis       ",
                ext=0,
            )
            fits.setval(file, "COMMENT", value="", ext=0)
            fits.setval(
                file,
                "COMMENT",
                value="         For a thorough documentation of this software package,         ",
                ext=0,
            )
            fits.setval(
                file,
                "COMMENT",
                value="         please see https://geckos-survey.github.io/gist-documentation/          ",
                ext=0,
            )
            fits.setval(
                file,
                "COMMENT",
                value="------------------------------------------------------------------------",
                ext=0,
            )
            fits.setval(file, "COMMENT", value="", ext=0)

    return None


def saveConfigToHeader(hdu, config):
    """
    Write each key-value pair of the given config subsection into the FITS header.
    config should be a dict of header-compatible (scalar/string) values.
    """
    for i in config.keys():
        hdu.header[i] = config[i]
    return hdu
