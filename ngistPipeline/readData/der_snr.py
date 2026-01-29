# =====================================================================================
"""
DER_SNR: noise estimate per spectrum (ST-ECF / MAST / CADC definition).

This module returns the NOISE term (not SNR) used to fill variance/error arrays
when no variance extension is present. Callers use it as a constant noise per spaxel:
  noise = der_snr(flux_1d)  -> scalar
  noise_per_spaxel = der_snr_2d(spec_2d)  -> array of shape (n_spaxels,)
"""

import numpy as np

# Coefficient 1.482602/sqrt(6) for robust noise scale from median of differences
_DER_SNR_COEFF = 0.6052697


def der_snr(flux):
    """
    Compute the DER_SNR noise estimate for a single spectrum (1D).

    The algorithm uses:
      noise = 1.482602/sqrt(6) * median(|2*flux_i - flux_{i-2} - flux_{i+2}|)
    Values with padded zeros are skipped. Returns the NOISE (not SNR); callers
    use this to fill variance/error arrays when no variance extension exists.

    Parameters
    ----------
    flux : array-like, 1D
        Flux spectrum (unit independent).

    Returns
    -------
    float
        Noise estimate [same units as flux], or 0.0 if spectrum too short.
    """
    flux = np.asarray(flux, dtype=float)
    # Values that are exactly zero (padded) are skipped
    flux = flux[flux != 0.0]
    n = len(flux)

    if n <= 4:
        return 0.0

    noise = _DER_SNR_COEFF * np.nanmedian(
        np.abs(2.0 * flux[2 : n - 2] - flux[0 : n - 4] - flux[4:n])
    )
    return float(noise)


def der_snr_2d(flux_2d):
    """
    Compute DER_SNR noise estimate for all spectra in a 2D array (vectorized).

    Each column is treated as one spectrum. Zeros are treated as missing (NaN)
    in the median. Returns one noise value per column.

    Parameters
    ----------
    flux_2d : array-like, shape (n_wave, n_spaxels)
        Cube of spectra (e.g. spec from readCube).

    Returns
    -------
    np.ndarray, shape (n_spaxels,)
        Noise estimate per spaxel. Short or all-zero columns get 0.0.
    """
    flux = np.asarray(flux_2d, dtype=float)
    if flux.ndim == 1:
        return np.array([der_snr(flux)], dtype=float)

    # Mask zeros so they are skipped in nanmedian
    flux = np.where(flux == 0.0, np.nan, flux)
    n = flux.shape[0]

    if n <= 4:
        return np.zeros(flux.shape[1], dtype=float)

    diff = 2.0 * flux[2:-2, :] - flux[:-4, :] - flux[4:, :]
    noise = _DER_SNR_COEFF * np.nanmedian(np.abs(diff), axis=0)
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    return noise


# end DER_SNR -------------------------------------------------------------------------
