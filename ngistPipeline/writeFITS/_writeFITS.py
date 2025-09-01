import logging
import os

from printStatus import printStatus

from ngistPipeline.writeFITS import save_maps_fits


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
            save_maps_fits.savefitsmaps("SPATIAL_BINNING", config["SPATIAL_BINNING"]["METHOD"],
                                        config["GENERAL"]["OUTPUT"]
                                        )
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
            save_maps_fits.savefitsmaps("KIN", config["KIN"]["METHOD"],
                                        config["GENERAL"]["OUTPUT"]
                                        )
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
            if os.path.isfile(outputPrefix + "_gas_bin.fits") == True:
                if config["GAS"]["LEVEL"] == "BIN": #And we aren't running in BOTH mode
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL=config["GAS"]["LEVEL"],
                        AoNThreshold=4,
                    )
            if os.path.isfile(outputPrefix + "_gas_spaxel.fits") == True:
                if config["GAS"]["LEVEL"] == 'SPAXEL': #And we aren't running in BOTH mode
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL=config["GAS"]["LEVEL"],
                        AoNThreshold=4,
                    )
            if os.path.isfile(outputPrefix + "_gas_bin.fits") == True and os.path.isfile(outputPrefix + "_gas_spaxel.fits") == True:
                if config["GAS"]["LEVEL"] == 'BOTH': # Special case for running in BOTH mode
                    # Run first to create the _BIN maps
                    printStatus.running('First run-through to save bin results')
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL='BIN',
                        AoNThreshold=4,
                    )
                    # Then run to create the SPAXEL maps
                    printStatus.running('second run through to save SPAXEL results')
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL='SPAXEL',
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
            save_maps_fits.savefitsmaps("SFH", config["SFH"]["METHOD"], config["GENERAL"]["OUTPUT"])
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

    # - - - - - USER MODULE - - - - -
    if module == "UMOD":
        try:
            printStatus.running("Producing User Module maps in FITS format")
            save_maps_fits.savefitsmaps("UMOD",  config["UMOD"]["METHOD"],
                                        config["GENERAL"]["OUTPUT"]
                                       )
            printStatus.updateDone("Producing User Module maps in FITS format")
            logging.info("Produced User Module maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing User Module maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce User Module component maps.")


    return None
