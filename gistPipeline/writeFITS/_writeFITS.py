import logging
import os

from printStatus import printStatus

from gistPipeline.writeFITS import save_maps_fits


def generateFITS(config, module):
    """
    generateFITS _summary_

    Args:
        config (_type_): _description_
        module (_type_): _description_

    Returns:
        _type_: _description_
    """

    outputPrefix = os.path.join(
        config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]
    )


    # - - - - - TABLES MODULE - - - - -
    if module == "SPATIAL_BINNING":
        try:
            printStatus.running("Producing table binned maps in FITS format")
            save_maps_fits.savefitsmaps("TABLE", config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing table binned maps in FITS format")
            logging.info("Produced table binned maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing table binned maps maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce table binned maps.")

    # - - - - - STELLAR KINEMATICS MODULE - - - - -
    if module == "KIN":
        try:
            printStatus.running("Producing stellar kinematics maps in FITS format")
            save_maps_fits.savefitsmaps("KIN", config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing stellar kinematics maps in FITS format")
            logging.info("Produced stellar kinematics maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing stellar kinematics maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce stellar kinematics maps.")

    # - - - - - CONTINUUM CUBE MODULE - - - - -
    if module == "CONT":
        try:
            printStatus.running(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            save_maps_fits.saveContLineCube(config)
            printStatus.updateDone(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            logging.info("Produced continuum-only and line-only cubes in FITS format")
        except Exception as e:
            printStatus.updateFailed(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            logging.error(e, exc_info=True)
            logging.error(
                "Failed to produce continuum-only and line-only cubes in FITS format"
            )

    # - - - - - EMISSION LINES MODULE - - - - -
    if module == "GAS":
        try:
            printStatus.running("Producing FITS maps from the emission-line analysis")
            if os.path.isfile(outputPrefix + "_gas_BIN.fits") == True:
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
            printStatus.updateDone(
                "Producing FITS maps from the emission-line analysis"
            )
            logging.info("Producing FITS maps from the emission-line analysis")
        except Exception as e:
            printStatus.updateFailed(
                "Producing FITS maps from the emission-line analysis"
            )
            logging.error(e, exc_info=True)
            logging.error("Failed to produce maps from the emission-line analysis.")

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -
    if module == "SFH":
        try:
            printStatus.running("Producing SFH maps in FITS format")
            save_maps_fits.savefitsmaps("SFH", config["GENERAL"]["OUTPUT"])
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
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ORIGINAL"
            )
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ADAPTED"
            )
            printStatus.updateDone("Producing line strength maps in FITS format")
            logging.info("Produced line strength maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing line strength maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce line strength maps.")

    return None
