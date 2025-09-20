import logging
import os

import numpy as np
from astropy.table import Table

# PHYSICAL CONSTANTS
C = 299792.458  # km/s

def loadSpecMask(config, file, logLam):
    """
    Returns a map of all emission line masked and a
    base goodPixels that has skylines already masked
    emission_lines is a map
    e.g. emission_lines["Hb4861"] = {
        wavelength: 4861,
        width: 20,
        adaptive: False
    }
    """

    # Creates good pixels with only skylines
    base_goodPixels = np.arange(len(logLam))
    emission_lines = {}

    mask_wavelength = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], str(file)), usecols=[0]
    )

    mask_width = np.genfromtxt(
        os.path.join(str(config["GENERAL"]["CONFIG_DIR"]), str(file)), usecols=[1]
    )

    mask_comment = np.genfromtxt(
        os.path.join(config["GENERAL"]["CONFIG_DIR"], str(file)), usecols=[2], dtype=str
    )

    for wavelength, width, comment in zip(mask_wavelength, mask_width, mask_comment):
        if "sky" in comment.lower():
            wavelength = wavelength / (1 + config["GENERAL"]["REDSHIFT"])
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
            if minimumPixel < 0:
                minimumPixel = 0
            if maximumPixel < 0:
                maximumPixel = 0
            if minimumPixel >= len(logLam):
                minimumPixel = len(logLam) - 1
            if maximumPixel >= len(logLam):
                maximumPixel = len(logLam) - 1

            # Mark masked spectral pixels
            base_goodPixels[minimumPixel : maximumPixel + 1] = -1

        else:
            # Comments should be in the format [line label as defined in emissionLines.config],adaptive
            comment = comment.split("-")
            # Remove brackets for forbidden lines
            lineName = comment[0].strip("[]")
            # limits wavelength to first 4 characters
            lineWavelength = str(int(float(wavelength)))
            # Checks if adaptive masking is turned on this line
            adaptive = len(comment) == 2 and "adaptive" in comment[1].lower()
            emission_lines[
                f"{lineName}{lineWavelength}"
            ] = { "wavelength": wavelength, "width": width, "adaptive": adaptive }
    return emission_lines, base_goodPixels

def loadGasKinematics(config):
    """
    Loads in gas kinematics per bin from GAS module output if it exists, otherwise returns None
    """
    gas_kin_path \
        = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_gas_bin.fits"
    try:
        gas_kin = Table.read(gas_kin_path)
    except FileNotFoundError:
        logging.warning("Gas kinematics file not found. Returning None.")
        return None

    emldb = Table.read(
        config["GENERAL"]["CONFIG_DIR"] + "/" + config["GAS"]["EMI_FILE"],
        format="ascii",
        )
    w = emldb["action"] == "f"
    linesfitted = emldb[w]

    # Create columns to extract
    columns = ["BIN_ID"]
    LMAX, LMIN = config["GAS"]["LMAX"], config["GAS"]["LMIN"]

    for line_name, line_wavelength in zip(linesfitted["name"], linesfitted["lambda"]):
        if LMIN <= line_wavelength <= LMAX:
            # Round down to nearest angstrom to get the same label style as emissionLines.config
            line_wavelength = str(int(float(line_wavelength)))
            for measurement in ["VEL", "SIGMA"]:
                column_name = f"{line_name}{line_wavelength}_{measurement}"
                columns.append(column_name)

    return gas_kin[columns]

def createAdaptiveSpectralMask(
        emission_lines,
        base_goodpixels,
        gas_kin,
        logLam,
        bin_id,
        mask_width,
        lmin,
        lmax
):
    """
    Creates an adaptive masking depending on gasKinematics measured in the gas module
    """
    try:
        goodPixels = np.array(base_goodpixels)
        if emission_lines is None or gas_kin is None:
            goodPixels = goodPixels[np.where(goodPixels != -1)[0]]
            return goodPixels

        gas_kin_bin = gas_kin[gas_kin["BIN_ID"] == bin_id]

        # Mask Width as a function of the velocity dispersion
        for line_label, line_info in emission_lines.items():
            wavelength, width = line_info["wavelength"], line_info["width"]
            if not (lmin <= wavelength <= lmax):
                continue
            elif line_info["adaptive"]:
                print("Adaptive masking for line:", line_label)
                print(f"Old position: {wavelength:.4f} and old width {width:.4f}")
                velocity, velocity_dispersion = gas_kin_bin[f"{line_label}_VEL"].value[0], gas_kin_bin[f"{line_label}_SIGMA"].value[0]
                adjusted_width = 2 * wavelength * (mask_width * velocity_dispersion / C)
                width = np.clip(adjusted_width, None, width)
                wavelength = wavelength * (1 + velocity / C)
                print(f"New position: {wavelength:.4f} and new width {width:.4f}")

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
            if minimumPixel < 0:
                minimumPixel = 0
            if maximumPixel < 0:
                maximumPixel = 0
            if minimumPixel >= len(logLam):
                minimumPixel = len(logLam) - 1
            if maximumPixel >= len(logLam):
                maximumPixel = len(logLam) - 1

            # Mark masked spectral pixels
            goodPixels[minimumPixel : maximumPixel + 1] = -1

        goodPixels = goodPixels[np.where(goodPixels != -1)[0]]

        return goodPixels
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
        goodPixels = np.array(base_goodpixels)
        return goodPixels
