import importlib.util
import logging
import os

from printStatus import printStatus

def _preparation_products_exist(config):

    outputPrefix = os.path.join(config["GENERAL"]["OUTPUT"],
                                config["GENERAL"]["RUN_ID"])

    masking_done = (
        config["SPATIAL_MASKING"]["METHOD"] == False
        or os.path.isfile(outputPrefix + "_mask.fits"))

    binning_done = (
        config["SPATIAL_BINNING"]["METHOD"] == False
        or os.path.isfile(outputPrefix + "_table.fits"))

    spectra_done = (
        config["PREPARE_SPECTRA"]["METHOD"] == False
        or (
            os.path.isfile(outputPrefix + "_bin_spectra.hdf5")
            and os.path.isfile(outputPrefix + "_bin_spectra_linear.hdf5")
            and (
                config["GAS"]["LEVEL"] != "SPAXEL"
                or os.path.isfile(outputPrefix + "_all_spectra.hdf5"))))

    return (
        config["GENERAL"]["OW_OUTPUT"] == False
        and masking_done
        and binning_done
        and spectra_done
    )

def readData_Module(config):
    """
    This function calls the readData routine specified by the user.
    """
    printStatus.module("readData module")

    # Check if module is turned off in MasterConfig
    if config["READ_DATA"]["METHOD"] == False:
        message = "Read data module was turned off. Module is skipped but beware this can cause issues with the preparation modules."
        logging.warning(message)
        printStatus.warning(message)
        return None

    # Check whether all preparation products already exist
    if _preparation_products_exist(config):
        message = (
            "Preparation products are already available. "
            "The readData module is skipped.")
        logging.info(message)
        printStatus.done("Preparation products already available. "
                         "Module is skipped.")
        return None

    # Import the chosen readData routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config["READ_DATA"]["METHOD"]
            + ".py",
        )
        logging.info("Using the read-in routine for " + config["READ_DATA"]["METHOD"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the read-in routine "
            + config["READ_DATA"]["METHOD"]
            + "."
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen readData routine
    try:
        cube = module.readCube(config)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Read-in routine "
            + config["READ_DATA"]["METHOD"]
            + " failed to read "
            + config["GENERAL"]["INPUT"]
            + "."
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Return the results
    return cube
