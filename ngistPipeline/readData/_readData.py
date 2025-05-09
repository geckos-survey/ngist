import importlib.util
import logging
import os

from printStatus import printStatus


def readData_Module(config):
    """
    This function calls the readData routine specified by the user.
    """
    printStatus.module("readData module")

    # Check if module is turned off in MasterConfig
    if config["READ_DATA"]["METHOD"] == False:
        # message = "The module was turned off. The nGIST cannot be executed without running the readData module."
        # printStatus.failed(message)
        # return "SKIP"
        message = "Read data module was turned off. Module is skipped but beware this can cause issues with the preparation modules."
        logging.warning(message)
        printStatus.warning(message)
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
