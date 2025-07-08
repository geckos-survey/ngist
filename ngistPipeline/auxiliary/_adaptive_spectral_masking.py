import os

import numpy as np
from astropy.table import Table

# PHYSICAL CONSTANTS
C = 299792.458  # km/s

def load_spec_mask(config, file, logLam):
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
    base_goodpixels = np.arange(len(logLam))
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
            base_goodpixels[minimumPixel : maximumPixel + 1] = -1

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

    return emission_lines, base_goodpixels

def load_gas_kinematics(config):
    """
    Loads in gas kinematics per bin from GAS module output if it exists, otherwise returns None
    """
    gas_kin_path \
        = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_gas_BIN.fits"
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

    for line_name, line_wavelength in zip(linesfitted["name"], linesfitted["lambda"]):
        if LMIN <= line_wavelength <= LMAX:
            # Round down to nearest angstrom to get the same label style as emissionLines.config
            line_wavelength = str(int(float(line_wavelength)))
            for measurement in ["VEL", "SIGMA"]:
                column_name = f"{line_name}{line_wavelength}_{measurement}"
                columns.append(column_name)

    return gas_kin[columns]


def create_adaptive_spectral_mask(emission_lines, base_goodpixels, gas_kin, logLam, bin_id):
    """
    Creates an adaptive masking depending on gasKinematics measured in the gas module
    """

    goodPixels = np.array(base_goodpixels)
    if emission_lines is None or gas_kin is None:
        goodPixels = goodPixels[np.where(goodPixels != -1)[0]]
        return goodPixels

    gas_kin_bin = gas_kin[gas_kin["BIN_ID"] == bin_id]

    for line_label, line_info in emission_lines.items():
        wavelength, width = line_info["wavelength"], line_info["width"]
        if line_info["adaptive"]:
            velocity, velocity_dispersion = gas_kin_bin[f"{line_label}_VEL"].value[0], gas_kin_bin[f"{line_label}_SIGMA"].value[0]
            # Keep width between 5 and 30 angstroms
            width = max(
                5,
                min(
                    6 * wavelength * velocity_dispersion / C,
                    30
                )
            )
            wavelength = wavelength * (1 + velocity / C)

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