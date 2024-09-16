import importlib.util
import logging
import os

from printStatus import printStatus


def prepareSpectra_Module(config, cube):
    """
    This function calls the prepareSpectra routine specified by the user.
    """
    printStatus.module("prepareSpectra module")

    # Check if module is turned off in MasterConfig
    if config["PREPARE_SPECTRA"]["METHOD"] == False:
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return None

    # Check if outputs are already available
    # Note that _AllSpectra.fits only needs to exists if config["GAS"]["LEVEL"] == 'SPAXEL'
    outputPrefix = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + ""
    )
    
    # Check if the 'OW_OUTPUT' flag in the config is set to False
    if config["GENERAL"]["OW_OUTPUT"] == False:
        # List of FITS files to check
        fits_files = [
            outputPrefix + "_AllSpectra.fits",
            outputPrefix + "_BinSpectra.fits",
            outputPrefix + "_BinSpectra_linear.fits"
        ]
        # List of HDF5 files to check
        hdf5_files = [
            outputPrefix + "_AllSpectra.hdf5",
            outputPrefix + "_BinSpectra.hdf5",
            outputPrefix + "_BinSpectra_linear.hdf5"
        ]

        # Combine both lists and check for missing files
        missing_files = [f for f in fits_files + hdf5_files if not os.path.isfile(f)]

        # Raise an error if any files are missing
        if missing_files:
            raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

        # Log and print status if all files are present
        logging.info("Results of the module are already in the output directory. Module is skipped.")
        printStatus.done("Results are already available. Module is skipped.")
        return None


    # Import the chosen prepareSpectra routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config["PREPARE_SPECTRA"]["METHOD"]
            + ".py",
        )
        logging.info("Using the routine for " + config["PREPARE_SPECTRA"]["METHOD"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the routine '"
            + config["PREPARE_SPECTRA"]["METHOD"]
            + ".py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen spatialBinning routine
    try:
        module.prepSpectra(config, cube)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Routine " + config["PREPARE_SPECTRA"]["METHOD"] + " failed."
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Return
    return None
