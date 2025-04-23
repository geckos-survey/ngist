import importlib.util
import logging
import os

from printStatus import printStatus

from ngistPipeline.writeFITS import _writeFITS

import yaml

def user_Modules(config):
    """
    This function calls user defined routin.
    """
    printStatus.module("user module")
        
    if config['UMOD']["METHOD"] == "twocomp_ppxf":

        config_use = config['UMOD']["METHOD"]

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
            and os.path.isfile(outPrefix + "_twocomp_kin.fits") == True
            and os.path.isfile(outPrefix + "_twocomp_kin-bestfit.fits") == True
            and os.path.isfile(outPrefix + "_twocomp_kin-optimalTemplates.fits") == True
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
                + ".py",
            )
            logging.info(
                "Using the twocomp stellarKinematics routine '" + config_use + ".py'"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logging.critical(e, exc_info=True)
            message = (
                "Failed to import the twocomp stellarKinematics routine '"
                + config_use
                + ".py'"
            )
            printStatus.failed(message)
            logging.critical(message)
            return "SKIP"

        # Execute the twocomponent stellarKinematics routine
        try:
            module.extractStellarKinematics(config)
            _writeFITS.generateFITS(config, 'UMOD')
        except Exception as e:
            logging.critical(e, exc_info=True)
            message = (
                "userModule routine '" + config_use + ".py' failed."
            )
            printStatus.failed(message + " See LOGFILE for further information.")
            logging.critical(message)
            return "SKIP"

    # Return
    return None
