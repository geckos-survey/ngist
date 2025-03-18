import os

import numpy as np
from astropy.table import Table

def loadSpecMask(config, file):
    """
    Returns mapping between emission line and properties,
    returns skylines separately as an array

    specMap:
    Key: [LineName][LineWavelength]
    Value: { wavelength: l, width: w, adaptive: True}

    skyLines:
    Value: { wavelength: l, width: w }
    """
    specMask = {}
    skyLines = []

    maskWavelength = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], str(file)), usecols=[0]
    )

    maskWidth = np.genfromtxt(
        os.path.join(str(config["GENERAL"]["CONFIG_DIR"]), str(file)), usecols=[1]
    )

    maskComment = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], str(file)), usecols=[2], dtype=str
    )

    for wavelength, width, comment in zip(maskWavelength, maskWidth, maskComment):
        if "sky" in comment.lower():
            skyLines.append({ "wavelength": wavelength, "width": width })
        else:
            comment = comment.split()
            # Remove brackets for forbidden lines
            lineName = comment[0].strip("[]")
            # limits wavelength to first 4 characters
            lineWavelength = str(int(float(wavelength)))
            # Checks if adaptive masking is turned on this line
            if len(comment) == 2 and "adaptive" == comment[1].lower():
                adaptive = True
            else:
                adaptive = False
            specMask[
                f"{lineName}{lineWavelength}"
            ] = { "wavelength": wavelength, "width": width, "adaptive": adaptive }

    return specMask, skyLines

def loadGasKinematics(config):
    """
    Loads in gas kinematics per bin from GAS module output if it exists, otherwise returns None
    """
    gas_kin_path = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_gas_BIN.fits"
    if not os.path.exists(gas_kin_path):
        return None

    gas_kin = Table.read(gas_kin_path)

    emldb = Table.read(
        config["GENERAL"]["CONFIG_DIR"] + "/" + config["GAS"]["EMI_FILE"],
        format="ascii",
    )
    w = emldb["action"] == "f"
    linesfitted = emldb[w]

    # Create columns to extract
    columns = ["BIN_ID", "V_STARS2"]
    LMAX, LMIN = config["GAS"]["LMAX"], config["GAS"]["LMIN"]
    for lineName, lineWavelength in zip(linesfitted["name"], linesfitted["lambda"]):
        if LMIN <= lineWavelength <= LMAX:
            # convert lineWavelength to take the first 4 characters
            lineWavelength = str(int(float(lineWavelength)))
            for measurement in ["VEL", "SIGMA"]:
                columns.append(f"{lineName}{lineWavelength}_{measurement}")

    return  gas_kin[columns]


def createAdaptiveSpectralMask(specMap, skyLines, gasKinematics, logLam, binId):
    """
    Creates an adaptive masking depending on gasKinematics measured in the gas module
    """

    # Create a copy such that we don't modify specMap outside the function
    specMap = dict(specMap)
    goodPixels = np.arange(len(logLam))
    # Loads the gas kinematics from bin with id binId
    pass
    # For each gas kinematics line, first check if the mask exists and modify if it exists
    # For now, only adaptive mask ones included in specMask
    pass
    # create goodPixels
    pass
    return goodPixels
