import os

import numpy as np

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

def loadGasKinematics(file):
    """
    Loads in gas kinematics from GAS module if output, otherwise returns None
    """
    pass

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
