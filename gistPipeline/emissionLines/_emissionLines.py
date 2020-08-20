import os
import importlib.util
import logging

from printStatus import printStatus

from gistPipeline.plotting import _plotting



def emissionLines_Module(config):
    """
    This function calls the emissionLines routine specified by the user.
    """
    printStatus.module("emissionLines module")

    # Check if module is turned off in MasterConfig
    if config['GAS']['METHOD'] == False: 
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return(None)

    # Check if outputs are already available
    outPrefix = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])
    if config['GAS']['LEVEL'] == 'BIN':
        if config['GENERAL']['OW_OUTPUT'] == False and \
                os.path.isfile(outPrefix+"_gas_BIN.fits") == True and \
                os.path.isfile(outPrefix+"_gas-bestfit_BIN.fits") == True and \
                os.path.isfile(outPrefix+"_gas-cleaned_BIN.fits") == True and \
                os.path.isfile(outPrefix+"_gas-emission_BIN.fits") == True:
            logging.info("Results of the module are already in the output directory. Module is skipped.") 
            printStatus.done("Results are already available. Module is skipped.")
            return(None)
    elif config['GAS']['LEVEL'] == 'SPAXEL':
        if config['GENERAL']['OW_OUTPUT'] == False and \
                os.path.isfile(outPrefix+"_gas_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-bestfit_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-cleaned_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-emission_SPAXEL.fits") == True:
            logging.info("Results of the module are already in the output directory. Module is skipped.") 
            printStatus.done("Results are already available. Module is skipped.")
            return(None)
    elif config['GAS']['LEVEL'] == 'BOTH':
        if config['GENERAL']['OW_OUTPUT'] == False and \
                os.path.isfile(outPrefix+"_gas_BIN.fits") == True and os.path.isfile(outPrefix+"_gas_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-bestfit_BIN.fits") == True and os.path.isfile(outPrefix+"_gas-bestfit_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-cleaned_BIN.fits") == True and os.path.isfile(outPrefix+"_gas-cleaned_SPAXEL.fits") == True and \
                os.path.isfile(outPrefix+"_gas-emission_BIN.fits") == True and os.path.isfile(outPrefix+"_gas-emission_SPAXEL.fits") == True:
            logging.info("Results of the module are already in the output directory. Module is skipped.") 
            printStatus.done("Results are already available. Module is skipped.")
            return(None)

    # Import the chosen emissionLines routine
    try:
        spec = importlib.util.spec_from_file_location("",os.path.dirname(os.path.realpath(__file__))+"/"+config['GAS']['METHOD']+'.py')
        logging.info("Using the emissionLines routine '"+config['GAS']['METHOD']+".py'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Failed to import the emissionLines routine '"+config['GAS']['METHOD']+".py'"
        printStatus.failed(message)
        logging.critical(message)
        return("SKIP")

    # Execute the chosen emissionLines routine
    try:
        module.performEmissionLineAnalysis(config)
        _plotting.generatePlots(config, 'GAS')
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "emissionLine routine '"+config['GAS']['METHOD']+".py' failed."
        printStatus.failed(message+" See LOGFILE for further information.")
        logging.critical(message)
        return("SKIP")
 
    # Return
    return(None)


