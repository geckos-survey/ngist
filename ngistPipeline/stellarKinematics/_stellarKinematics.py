import importlib.util
import logging
import os

import yaml
from printStatus import printStatus

from ngistPipeline.writeFITS import _writeFITS


def stellarKinematics_Module(config):
    """
    This function calls the stellarKinematics routine specified by the user.
    """
    printStatus.module("stellarKinematics module")
        
    config_use = config['KIN']["METHOD"]

    # Check if module is turned off in MasterConfig
    if config_use == False:
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return None

    # Check if outputs are already available
    outPrefix = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])        
    if (
        config["GENERAL"]["OW_OUTPUT"] == False
        and os.path.isfile(outPrefix + "_kin.fits") == True
        and os.path.isfile(outPrefix + "_kin_bestfit.fits") == True
        and os.path.isfile(outPrefix + "_kin_optimal_templates.fits") == True
    ):
        logging.info(
            "Results of the module are already in the output directory. Module is skipped."
        )
        printStatus.done("Results are already available. Module is skipped.")
        return None

    # Import the chosen stellarKinematics routine
    try:

        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config_use
            + "_kin_wrapper.py",
        )
        logging.info(
            "Using the stellarKinematics routine '" + config_use + "_kin_wrapper.py'"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the stellarKinematics routine '"
            + config_use
            + "_kin_wrapper.py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen stellarKinematics routine
    try:
        module.extractStellarKinematics(config)
        _writeFITS.generateFITS(config, 'KIN')
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "stellarKinematics routine '" + config_use + "_kin_wrapper.py' failed."
        )
        printStatus.failed(message + " See LOGFILE for further information.")
        logging.critical(message)
        return "SKIP"

    # Return
    return None
