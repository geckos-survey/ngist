import logging
import os

from printStatus import printStatus

from ngistPipeline.plotting import (ngistPlot_gas, ngistPlot_kin,
                                   ngistPlot_lambdar, ngistPlot_ls, ngistPlot_sfh,
                                   save_maps_fits)


def generatePlots(config, module):
    outputPrefix = os.path.join(
        config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]
    )

    # - - - - - STELLAR KINEMATICS MODULE - - - - -
    if module == "KIN":
        try:
            printStatus.running("Producing stellar kinematics maps")
            # gistPlot_kin.plotMaps('KIN', config['GENERAL']['OUTPUT'])
            save_maps_fits.savefitsmaps("kin", config["GENERAL"]["OUTPUT"])
            # gistPlot_lambdar.plotMaps(config['GENERAL']['OUTPUT']) # Don't want to plot lambda
            printStatus.updateDone("Producing stellar kinematics maps")
            logging.info("Produced stellar kinematics maps")
        except Exception as e:
            printStatus.updateFailed("Producing stellar kinematics maps")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce stellar kinematics maps.")

    # - - - - - EMISSION LINES MODULE - - - - -
    if module == "GAS":
        try:
            printStatus.running("Producing maps from the emission-line analysis")
            if os.path.isfile(outputPrefix + "_gas_bin.fits") == True:
                # gistPlot_gas.plotMaps(config['GENERAL']['OUTPUT'], 'BIN', True)
                save_maps_fits.savefitsmaps_GASmodule(
                    "gas",
                    config["GENERAL"]["OUTPUT"],
                    LEVEL=config["GAS"]["LEVEL"],
                    AoNThreshold=4,
                )
            if os.path.isfile(outputPrefix + "_gas_spaxel.fits") == True:
                save_maps_fits.savefitsmaps_GASmodule(
                    "gas",
                    config["GENERAL"]["OUTPUT"],
                    LEVEL=config["GAS"]["LEVEL"],
                    AoNThreshold=4,
                )
            printStatus.updateDone("Producing maps from the emission-line analysis")
            logging.info("Producing maps from the emission-line analysis")
        except Exception as e:
            printStatus.updateFailed("Producing maps from the emission-line analysis")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce maps from the emission-line analysis.")

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -
    if module == "SFH":
        try:
            printStatus.running("Producing SFH maps")
            # gistPlot_sfh.plotMaps('SFH', config['GENERAL']['OUTPUT'])
            save_maps_fits.savefitsmaps("sfh", config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing SFH maps")
            logging.info("Produced SFH maps")
        except Exception as e:
            printStatus.updateFailed("Producing SFH maps")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce SFH maps.")

    # - - - - - LINE STRENGTHS MODULE - - - - -
    if module == "LS":
        try:
            printStatus.running("Producing line strength maps")
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
            printStatus.updateDone("Producing line strength maps")
            logging.info("Produced line strength maps")
        except Exception as e:
            printStatus.updateFailed("Producing line strength maps")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce line strength maps.")

    # Return
    return None
