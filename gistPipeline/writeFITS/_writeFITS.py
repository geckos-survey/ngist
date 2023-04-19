import logging
import os

from gistPipeline.writeFITS import (
    save_maps_fits,
)

from printStatus import printStatus


def generateFITS(config, module):
    outputPrefix = os.path.join(
        config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]
    )

    # - - - - - STELLAR KINEMATICS MODULE - - - - -
    if module == "KIN":
        try:
            printStatus.running("Producing stellar kinematics maps in FITS format")
            save_maps_fits.savefitsmaps("kin", config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing stellar kinematics maps in FITS format")
            logging.info("Produced stellar kinematics maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing stellar kinematics maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce stellar kinematics maps.")

    # - - - - - EMISSION LINES MODULE - - - - -
    if module == "GAS":
        try:
            printStatus.running("Producing FITS maps from the emission-line analysis")
            if os.path.isfile(outputPrefix + "_gas_BIN.fits") == True:
                # gistPlot_gas.plotMaps(config['GENERAL']['OUTPUT'], 'BIN', True)
                save_maps_fits.savefitsmaps_GASmodule(
                    "gas",
                    config["GENERAL"]["OUTPUT"],
                    LEVEL=config["GAS"]["LEVEL"],
                    AoNThreshold=4,
                )
            if os.path.isfile(outputPrefix + "_gas_SPAXEL.fits") == True:
                save_maps_fits.savefitsmaps_GASmodule(
                    "gas",
                    config["GENERAL"]["OUTPUT"],
                    LEVEL=config["GAS"]["LEVEL"],
                    AoNThreshold=4,
                )
            printStatus.updateDone("Producing FITS maps from the emission-line analysis")
            logging.info("Producing FITS maps from the emission-line analysis")
        except Exception as e:
            printStatus.updateFailed("Producing FITS maps from the emission-line analysis")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce maps from the emission-line analysis.")

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -
    if module == "SFH":
        try:
            printStatus.running("Producing SFH maps in FITS format")
            # gistPlot_sfh.plotMaps('SFH', config['GENERAL']['OUTPUT'])
            save_maps_fits.savefitsmaps("sfh", config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing SFH maps in FITS format")
            logging.info("Produced SFH maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing SFH maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce SFH maps.")

    # - - - - - LINE STRENGTHS MODULE - - - - -
    if module == "LS":
        try:
            printStatus.running("Producing line strength maps in FITS format")
            # gistPlot_ls.plotMaps(config['GENERAL']['OUTPUT'], 'ORIGINAL')
            # gistPlot_ls.plotMaps(config['GENERAL']['OUTPUT'], 'ADAPTED')
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ORIGINAL"
            )
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ADAPTED"
            )
            # if config['LS']['TYPE'] == 'SPP':
            # gistPlot_sfh.plotMaps('LS', config['GENERAL']['OUTPUT'])
            printStatus.updateDone("Producing line strength maps in FITS format")
            logging.info("Produced line strength maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing line strength maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce line strength maps.")

    # Return
    return None
