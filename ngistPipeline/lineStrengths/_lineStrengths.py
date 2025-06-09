import importlib.util
import logging
import os

from printStatus import printStatus

from ngistPipeline.writeFITS import _writeFITS


def lineStrengths_Module(config):
    """
    This function calls the lineStrengths routine specified by the user.
    """
    printStatus.module("lineStrengths module")

    # Check if module is turned off in MasterConfig
    if config["LS"]["METHOD"] == False:
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return None

    # Check if outputs are already available
    outPrefix = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
    if (
        config["GENERAL"]["OW_OUTPUT"] == False
        and os.path.isfile(outPrefix + "_ls_orig_res.fits") == True
        and os.path.isfile(outPrefix + "_ls_adap_res.fits") == True
        and os.path.isfile(outPrefix + "_ls_cleaned_linear.fits") == True
    ):
        logging.info(
            "Results of the module are already in the output directory. Module is skipped."
        )
        printStatus.done("Results are already available. Module is skipped.")
        return None

    # Import the chosen lineStrengths routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config["LS"]["METHOD"]
            + ".py",
        )
        logging.info(
            "Using the lineStrengths routine '" + config["LS"]["METHOD"] + ".py'"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the lineStrengths routine '"
            + config["LS"]["METHOD"]
            + ".py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen lineStrengths routine
    try:
        module.measureLineStrengths(config)
        _writeFITS.generateFITS(config, "LS")
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "lineStrengths routine '" + config["LS"]["METHOD"] + ".py' failed."
        printStatus.failed(message + " See LOGFILE for further information.")
        logging.critical(message)
        return "SKIP"

    # Return
    return None
