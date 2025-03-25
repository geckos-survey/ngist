import os

import numpy as np
from astropy.table import Table

# PHYSICAL CONSTANTS
C = 299792.458  # km/s

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
    emLines = {}
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
            # Comments should be in the format [line label as defined in emissionLines.config],adaptive
            comment = comment.split("-")
            # Remove brackets for forbidden lines
            lineName = comment[0].strip("[]")
            # limits wavelength to first 4 characters
            lineWavelength = str(int(float(wavelength)))
            # Checks if adaptive masking is turned on this line
            adaptive = len(comment) == 2 and "adaptive" in comment[1].lower()
            emLines[
                f"{lineName}{lineWavelength}"
            ] = { "wavelength": wavelength, "width": width, "adaptive": adaptive }

    return emLines, skyLines

def loadGasKinematics(config):
    """
    Loads in gas kinematics per bin from GAS module output if it exists, otherwise returns None
    """
    gasKinPath = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_gas_BIN.fits"
    if not os.path.exists(gasKinPath):
        return None

    gasKin = Table.read(gasKinPath)

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

    return  gasKin[columns]


def createAdaptiveSpectralMask(emLines, skyLines, gasKin, logLam, binId, config):
    """
    Creates an adaptive masking depending on gasKinematics measured in the gas module
    """

    # Create a copy such that we don't modify specMap outside the function
    emLines = emLines.copy()
    goodPixels = np.arange(len(logLam))
    binInfo = gasKin[gasKin["BIN_ID"] == binId]

    # if gas kinematics is defined, update emLines
    if gasKin:
        for lineLabel, lineInfo in emLines.items():
            wavelength, width, adaptive = lineInfo["wavelength"], lineInfo["width"], lineInfo["adaptive"]
            if adaptive:
                newWidth = 4 * binInfo[f"{lineLabel}_SIGMA"].item() / C * wavelength
                emLines[lineLabel]["width"] = newWidth

    # create goodPixels
    for lineInfo in emLines.values():
        wavelength, width = lineInfo["wavelength"], lineInfo["width"]
        minimumPixel = int(
            np.round(
                (np.log(wavelength - width / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )
        maximumPixel = int(
            np.round(
                (np.log(wavelength + width / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )

        # Handle border of wavelength range
        minimumPixel = min(0, minimumPixel)
        minimumPixel = max(len(logLam) - 1, minimumPixel)
        maximumPixel = min(0, maximumPixel)
        maximumPixel = max(len(logLam) - 1, maximumPixel)

        # Mark masked spectral pixels
        goodPixels[minimumPixel : maximumPixel + 1] = -1

    for skyLine in skyLines:
        wavelength, width = skyLine["wavelength"] / (1 + config["GENERAL"]["REDSHIFT"]), skyLine["width"]
        minimumPixel = int(
            np.round(
                (np.log(wavelength - width / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )
        maximumPixel = int(
            np.round(
                (np.log(wavelength + width / 2.0) - logLam[0])
                / (logLam[1] - logLam[0])
            )
        )

        # Handle border of wavelength range
        minimumPixel = min(0, minimumPixel)
        minimumPixel = max(len(logLam) - 1, minimumPixel)
        maximumPixel = min(0, maximumPixel)
        maximumPixel = max(len(logLam) - 1, maximumPixel)

        # Mark masked spectral pixels
        goodPixels[minimumPixel : maximumPixel + 1] = -1

    goodPixels = goodPixels[np.where(goodPixels != -1)[0]]

    return goodPixels
