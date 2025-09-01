#!/usr/bin/env python

# ==================================================================================================================== #
#                                                                                                                      #
#                                          T H E   nG I S T   P I P E L I N E                                          #
#                                                                                                                      #
# ==================================================================================================================== #


import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

import numpy as np
from astropy.io import ascii, fits
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
import importlib.util
import logging
import optparse
import sys
import time

import matplotlib

matplotlib.use("pdf")

from printStatus import printStatus

from ngistPipeline._version import __version__
from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.continuumCube import _continuumCube
from ngistPipeline.emissionLines import _emissionLines
from ngistPipeline.initialise import _initialise
from ngistPipeline.lineStrengths import _lineStrengths
from ngistPipeline.prepareSpectra import _prepareSpectra
from ngistPipeline.readData import _readData
from ngistPipeline.spatialBinning import _spatialBinning
from ngistPipeline.spatialMasking import _spatialMasking
from ngistPipeline.starFormationHistories import _starFormationHistories
from ngistPipeline.stellarKinematics import _stellarKinematics
from ngistPipeline.userModules import _userModules
from ngistPipeline.writeHTML import _writeHTML


def skipGalaxy(config):
    # _auxiliary.addGISTHeaderComment(config)
    printStatus.module("The nGIST pipeline")
    printStatus.failed("Galaxy is skipped!")
    logging.critical("Galaxy is skipped!")


def numberOfGalaxies(filename):
    """
    Returns the number of galaxies to be analysed, as stated in the config file.
    """
    i = 0
    for line in open(filename):
        if not line.startswith("#"):
            i = i + 1
    return i


def runGIST(dirPath, galindex):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - -  I N I T I A L I S E   T H E   G I S T  - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - INITIALISE MODULE - - - - -
    # Read config
    config = _initialise.readMasterConfig(dirPath.configFile, galindex)
    config = _initialise.addPathsToConfig(config, dirPath)

    # Print configurations
    _initialise.printConfig(config)

    # Check output directory
    _initialise.checkOutputDirectory(config)

    # Setup logfile
    _initialise.setupLogfile(config)
    sys.excepthook = _initialise.handleUncaughtException

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  P R E P A R A T I O N   M O D U L E S  - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - READ_DATA MODULE - - - - -

    cube = _readData.readData_Module(config)
    if cube == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - SPATIAL MASKING MODULE - - - - -

    _ = _spatialMasking.spatialMasking_Module(config, cube)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - SPATIAL BINNING MODULE - - - - -

    _ = _spatialBinning.spatialBinning_Module(config, cube)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - PREPARE SPECTRA MODULE - - - - -

    _ = _prepareSpectra.prepareSpectra_Module(config, cube)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    del cube

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - -   A N A L Y S I S   M O D U L E S   - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - STELLAR KINEMATICS MODULE - - - - -

    _ = _stellarKinematics.stellarKinematics_Module(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - CONTINUUM CUBE MODULE - - - - -

    _ = _continuumCube.continuumCube_Module(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - EMISSION LINES MODULE - - - - -

    _ = _emissionLines.emissionLines_Module(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -

    _ = _starFormationHistories.starFormationHistories_Module(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - LINE STRENGTHS MODULE - - - - -

    _ = _lineStrengths.lineStrengths_Module(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - USERS  MODULE - - - - -

    _ = _userModules.user_Modules(config)
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - CREATE HTML - - - - -

    if config["GENERAL"].get("CREATE_HTML", False):
        _ = _writeHTML.writeHTML(config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  F I N A L I S E   T H E   A N A L Y S I S  - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Branding
    # _auxiliary.addGISTHeaderComment(config)

    # Goodbye
    printStatus.module("nGIST pipeline")
    printStatus.done("nGIST completed successfully.")
    logging.info("nGIST completed successfully.")


# ============================================================================ #
#                           M A I N   F U N C T I O N                          #
# ============================================================================ #
def main(args=None):
    # Capture command-line arguments
    parser = optparse.OptionParser(usage="%ngistPipeline [options] arg")
    jls_extract_var = "configFile"
    parser.add_option(
        "--config",
        dest=jls_extract_var,
        type="string",
        help="State the path of the config file.",
    )
    parser.add_option(
        "--default-dir",
        dest="defaultDir",
        type="string",
        help="File defining default directories for input, output, configuration files, and spectral templates.",
    )
    (dirPath, args) = parser.parse_args()

    # Check if required command-line argument is given
    if dirPath.configFile == None:
        printStatus.failed(
            "Please specify the path of the config file to be used. Exit!"
        )
        exit(1)

    # Check if Config-file exists
    if os.path.isfile(dirPath.configFile) == False:
        printStatus.failed("Config file at " + dirPath.configFile + " not found. Exit!")
        exit(1)

    # Iterate over galaxies in Config-file
    ngalaxies = 1  # numberOfGalaxies(dirPath.configFile) - 2 # Amelia changed this because she changed the format of the config file. We can revisit should we ever need to run more than one galaxy per config file
    if ngalaxies <= 0:
        message = "The number of runs defined in the config file seems to be 0. Exit."
        printStatus.failed(message)
        exit(1)
    for galindex in range(ngalaxies):
        runGIST(dirPath, galindex)
        print("\n")


if __name__ == "__main__":
    # Call the main function
    main()
