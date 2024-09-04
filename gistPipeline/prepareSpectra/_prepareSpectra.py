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
    if (
        config["GENERAL"]["OW_OUTPUT"] == False
        and ((os.path.isfile(outputPrefix + "_AllSpectra.fits") == True) or
             (config["GAS"]["LEVEL"] != "SPAXEL")) 
        and os.path.isfile(outputPrefix + "_BinSpectra.fits") == True
        and os.path.isfile(outputPrefix + "_BinSpectra_linear.fits") == True
    ):
        logging.info(
            "Results of the module are already in the output directory. Module is skipped."
        )
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
