import logging
import os
import sys
import time

import yaml
from printStatus import printStatus

from ngistPipeline._version import __version__

"""
PURPOSE:
  This file contains a collection of functions necessary to initialise the
  pipeline. This includes the creation of the LOGFILE, functions to read, save,
  and check the MasterConfig file, and print the configurations to stdout.

  The functions in this file do not interfere with subsequent modules or their
  configuration parameters provided in MasterConfig.

  02 Mar 2023: AFM editing to change configFile to a .yaml to make more user friendly
"""


def setupLogfile(config):
    """Initialise the LOGFILE."""
    welcomeString = "\n\n# ============================================== #\n#{:^48}#\n#{:^48}#\n# ============================================== #\n".format(
        "THE GIST PIPELINE", "Version " + __version__
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(config["GENERAL"]["OUTPUT"], "LOGFILE"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)-8s - %(module)s: %(message)s",
        datefmt="%m/%d/%y %H:%M:%S",
    )
    logging.Formatter.converter = time.gmtime
    logging.info(welcomeString)


def handleUncaughtException(exceptionType, exceptionValue, exceptionTraceback):
    """Write error message of uncaught exceptions in the logfile."""
    logging.error(
        "Uncaught Exception",
        exc_info=(exceptionType, exceptionValue, exceptionTraceback),
    )
    sys.__excepthook__(exceptionType, exceptionValue, exceptionTraceback)
    print("\n")
    printStatus.failed("FATAL ERROR! The execution of the pipeline is terminated.")
    print("")


def convertConfigDataType(value):
    """
    Convert the configuration parameters from MasterConfig to the most suitable data type.
    """
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() == "none":
        return None
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    try:
        return str(value)
    except:
        pass


def readMasterConfig(filename, galindex):
    """
    Read the MasterConfig file and stores all parameters in the configs dictionary.
    """
    # Amelia edited this module to instead of reading in the old MasterConfig, to read in MasterConfig.yaml
    with open(filename, "r") as f:
        configs = yaml.safe_load(f)

    return configs


def addPathsToConfig(
    config, dirPath
):  # Amrlia - I *think* the input here is the dictionary...
    """
    Combine the configuration parameters from MasterConfig with the paths specified as command line arguments.

    Naturally, this function cannot account for paths in user-defined parameters and/or user-defined modules.
    """
    if os.path.isfile(dirPath.defaultDir) == True:
        for line in open(dirPath.defaultDir, "r"):

        #for line in open('configFiles/defaultDir', "r"): #Amelia uncomment for testing only. Need to be in gistTutorial folder
            if not line.startswith('#'):
                line = line.split('=')
                line = [x.strip() for x in line]
                if os.path.isdir(line[1]) == True:
                    if line[0] == "inputDir":
                        config["GENERAL"]["INPUT"] = os.path.join(
                            line[1], config["GENERAL"]["INPUT"]
                        )
                    elif line[0] == "outputDir":
                        config["GENERAL"]["OUTPUT"] = os.path.join(
                            line[1],
                            config["GENERAL"]["OUTPUT"],
                            config["GENERAL"]["RUN_ID"],
                        )
                    elif line[0] == "configDir":
                        config["GENERAL"]["CONFIG_DIR"] = line[1]
                    elif line[0] == "templateDir":
                        config["GENERAL"]["TEMPLATE_DIR"] = line[1]
                elif line[1] == "outputDir" and line[0] == "configDir":
                    config["GENERAL"]["CONFIG_DIR"] = config["GENERAL"]["OUTPUT"]
                else:
                    print(
                        "WARNING! "
                        + line[1]
                        + " specified as default "
                        + line[0]
                        + " is not a directory!"
                    )
    else:
        print("WARNING! " + dirPath.defaultDir + " is not a file!")

    return config


def checkOutputDirectory(config):
    """
    Create output directory if it does not exist yet.
    """
    if os.path.isdir(config["GENERAL"]["OUTPUT"]) == False:
        os.mkdir(config["GENERAL"]["OUTPUT"])
        saveConfig(config)
    else:
        if config["GENERAL"]["OW_CONFIG"] == True:
            saveConfig(config)
        else:
            loadedConfig = loadConfig(config["GENERAL"]["OUTPUT"])
            checkConfig(config, loadedConfig)

    return None


def saveConfig(config):
    """
    Save configurations from MasterConfig in the output directory of the current run.
    """
    with open(os.path.join(config["GENERAL"]["OUTPUT"], "CONFIG"), "w") as file:
        yaml.dump(config, file, sort_keys=False)
    return None


def loadConfig(outdir):
    """
    Load configurations from a saved CONFIG file in the output directory of the current run.
    """
    with open(os.path.join(outdir, "CONFIG"), "r") as file:
        loadedConfig = yaml.load(file, Loader=yaml.FullLoader)
    return loadedConfig


def checkConfig(config, loadedConfig):
    """
    Compare to config dictionaries.
    """
    if config != loadedConfig:
        message = "The configurations set in MasterConfig and those saved in the output directory are not identical. Please double-check your configurations. The analysis will continue with the configurations from MasterConfig, however, this does not imply that any previous results are compatible with these configurations."
        printStatus.warning(message)
    return None


def printConfig(config):
    """
    Print an overview of the configuration parameters to stdout and in the logfile.
    """
    os.system("clear")
    headerString = (
        "\n"
        "\033[0;37m"
        + "********************************************************************"
        + "\033[0;39m\n"

        "\033[0;37m"
        + "*     The      ____ ___ ____ _____                                 *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*       _ __  / ___|_ _/ ___|_   _|                                *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      | '_ \| |  _ | |\___ \ | |                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      | | | | |_| || | ___) || |                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      |_| |_|\____|___|____/ |_|  Pipeline                        *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "* .    _     *         |     .       .     <==X==>           +     *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    .' \\\`.     +    -*-     *   .                .   *           *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "* .  |__''_|  .        |   +         .    +       .                *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    |     | .                                        .     -*-    *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    |     |           `  .    '      *     . *   .    +    '      *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*  _.'-----'-._     *                  .                           *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*/             \__.__.--._______________                           *"
        + "\033[0;39m\n"

        "\033[0;37m"
        + "********************************************************************"
        + "\033[0;39m\n"
    )
    infoString = ""
    for mK in config.keys():
        infoString = infoString + "\n"
        infoString = infoString + mK + "\n"
        for pK in config[mK].keys():
            infoString = infoString + "    {:13}{}\n".format(
                str(pK) + ":", str(config[mK][pK])
            )

    print(headerString + infoString + "\n")
