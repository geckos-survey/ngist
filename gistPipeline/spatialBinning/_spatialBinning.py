import os
import importlib.util
import logging

from printStatus import printStatus



def spatialBinning_Module(config, cube):
    """
    This functions calls the spatialMasking routine specified by the user.
    """
    printStatus.module("spatialBinning module")

    # Check if module is turned off in MasterConfig
    if config['SPATIAL_BINNING']['METHOD'] == False: 
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return(None)

    # Check if outputs are already available
    outputFile = os.path.join( config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'] ) + "_table.fits"
    if os.path.isfile( outputFile ) == True  and  config['GENERAL']['OW_OUTPUT'] == False: 
        logging.info("Results of the module are already in the output directory. Module is skipped.") 
        printStatus.done("Results are already available. Module is skipped.")
        return(None)

    # Import the chosen spatialBinning routine
    try:
        spec = importlib.util.spec_from_file_location("",os.path.dirname(os.path.realpath(__file__))+"/"+config['SPATIAL_BINNING']['METHOD']+'.py')
        logging.info("Using the spatial binning routine for "+config['SPATIAL_BINNING']['METHOD'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Failed to import the spatial binning routine "+config['SPATIAL_BINNING']['METHOD']+"."
        printStatus.failed(message)
        logging.critical(message)
        return("SKIP")

    # Execute the chosen spatialBinning routine
    try:
        module.generateSpatialBins(config, cube)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Spatial binning routine "+config['SPATIAL_BINNING']['METHOD']+" failed."
        printStatus.failed(message)
        logging.critical(message)
        return("SKIP")
 
    # Return
    return(None)


