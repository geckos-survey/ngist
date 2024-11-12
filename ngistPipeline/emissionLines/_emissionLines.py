import importlib.util
import logging
import os

from printStatus import printStatus

from ngistPipeline.writeFITS import _writeFITS


def emissionLines_Module(config):
    """
    This function calls the emissionLines routine specified by the user.
    """
    printStatus.module("emissionLines module")

    # Check if module is turned off in MasterConfig
    if config["GAS"]["METHOD"] == False:
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return None

    # Check if outputs are already available
    outPrefix = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
    if config["GAS"]["LEVEL"] == "BIN":
        if (
            config["GENERAL"]["OW_OUTPUT"] == False
            and os.path.isfile(outPrefix + "_gas_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas-bestfit_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas-cleaned_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas_BIN_maps.fits") == True
        ):
            logging.info(
                "Results of the module are already in the output directory. Module is skipped."
            )
            printStatus.done("Results are already available. Module is skipped.")
            return None
    elif config["GAS"]["LEVEL"] == "SPAXEL":
        if (
            config["GENERAL"]["OW_OUTPUT"] == False
            and os.path.isfile(outPrefix + "_gas_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas-bestfit_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas-cleaned_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas_SPAXEL_maps.fits") == True
        ):
            logging.info(
                "Results of the module are already in the output directory. Module is skipped."
            )
            printStatus.done("Results are already available. Module is skipped.")

            #check if AllSpectra.fits exists and if so remove it.
            AllSpectra_file = (os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) 
            + "_AllSpectra.fits")
            if os.path.exists(AllSpectra_file):    
                printStatus.warning(
                    "Removing the AllSpectra.fits file to save space"
                )
                os.remove(
                    AllSpectra_file
                    )        
    
            return None
    elif config["GAS"]["LEVEL"] == "BOTH":
        if (
            config["GENERAL"]["OW_OUTPUT"] == False
            and os.path.isfile(outPrefix + "_gas_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas-bestfit_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas-bestfit_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas-cleaned_BIN.fits") == True
            and os.path.isfile(outPrefix + "_gas-cleaned_SPAXEL.fits") == True
            and os.path.isfile(outPrefix + "_gas_BIN_maps.fits") == True
            and os.path.isfile(outPrefix + "_gas_SPAXEL_maps.fits") == True
        ):
            logging.info(
                "Results of the module are already in the output directory. Module is skipped."
            )
            printStatus.done("Results are already available. Module is skipped.")
            return None

    # Import the chosen emissionLines routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config["GAS"]["METHOD"]
            + "_gas_wrapper.py",
        )
        logging.info(
            "Using the emissionLines routine '" + config["GAS"]["METHOD"] + "_gas_wrapper.py'"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the emissionLines routine '"
            + config["GAS"]["METHOD"]
            + ".py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen emissionLines routine
    try:
        module.performEmissionLineAnalysis(config)
        if config["GAS"]["LEVEL"] == "BOTH": # rerun emission line module for the spaxel products
            module.performEmissionLineAnalysis(config)
        _writeFITS.generateFITS(config, "GAS") #Then move on to saving results as usual
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "emissionLine routine '" + config["GAS"]["METHOD"] + "_gas_wrapper.py' failed."
        printStatus.failed(message + " See LOGFILE for further information.")
        logging.critical(message)
        return "SKIP"

     # Remove AllSpectra file which is now no longer needed and takes up a lot of space
    # if config["GAS"]["LEVEL"] == "SPAXEL":
    #     printStatus.warning(
    #         "Removing the AllSpectra.hdf5 file to save space"
    #     )
    #     os.remove(
    #         os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
    #         + "_AllSpectra.hdf5"
    #         )        


    # Return
    return None
