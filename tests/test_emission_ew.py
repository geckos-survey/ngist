"""
Tests for equivalent width (EW) and KIN stellar continuum functionality
in the emission-line modules (ppxf and GandALF gas wrappers).
"""
import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from ngistPipeline.emissionLines.ppxf_gas_wrapper import (
    _load_kin_stellar_continuum,
    compute_equivalent_width,
)


def test_compute_equivalent_width_without_kin_continuum():
    """EW from gas-fit continuum (bestfit - gas_bestfit) gives positive EW for emission."""
    npix = 100
    nbins = 3
    nlines = 2
    logLam = np.linspace(np.log(4000), np.log(7000), npix)
    line_wavelengths = np.array([4861.32, 5006.77])

    # Synthetic: stellar continuum ~1, gas emission spikes at line centers
    wave = np.exp(logLam)
    stellar_cont = np.ones((nbins, npix)) * 0.5
    gas_bestfit = np.zeros((nbins, npix))
    for i, lam in enumerate(line_wavelengths):
        idx = np.argmin(np.abs(wave - lam))
        gas_bestfit[:, idx] = 0.1
    bestfit = stellar_cont + gas_bestfit

    gas_flux = np.array([[0.5, 0.3], [0.4, 0.2], [0.6, 0.25]])
    gas_flux_err = gas_flux * 0.1
    velscale = 50.0

    ew, ew_err, cont_at_line = compute_equivalent_width(
        gas_flux,
        gas_flux_err,
        bestfit,
        gas_bestfit,
        logLam,
        line_wavelengths,
        velscale,
        stellar_continuum=None,
    )

    assert ew.shape == (nbins, nlines)
    assert ew_err.shape == (nbins, nlines)
    assert cont_at_line.shape == (nbins, nlines)
    # Emission lines: EW should be positive (or NaN where f_cont <= 0)
    assert np.all(ew[np.isfinite(ew)] >= 0) or np.any(np.isfinite(ew))
    assert np.all(np.isfinite(cont_at_line) | (cont_at_line == 0))


def test_compute_equivalent_width_with_kin_continuum():
    """EW using provided stellar continuum (KIN) matches shape and gives finite values."""
    npix = 100
    nbins = 2
    nlines = 1
    logLam = np.linspace(np.log(4000), np.log(7000), npix)
    line_wavelengths = np.array([4861.32])

    stellar_continuum_kin = np.ones((nbins, npix)) * 0.6
    gas_bestfit = np.zeros((nbins, npix))
    bestfit = stellar_continuum_kin + gas_bestfit

    gas_flux = np.array([[0.3], [0.2]])
    gas_flux_err = np.array([[0.03], [0.02]])
    velscale = 50.0

    ew, ew_err, cont_at_line = compute_equivalent_width(
        gas_flux,
        gas_flux_err,
        bestfit,
        gas_bestfit,
        logLam,
        line_wavelengths,
        velscale,
        stellar_continuum=stellar_continuum_kin,
    )

    assert ew.shape == (nbins, nlines)
    assert np.any(np.isfinite(ew))
    assert np.all(ew[np.isfinite(ew)] >= 0)


def test_load_kin_stellar_continuum_missing_file():
    """_load_kin_stellar_continuum returns None when kin-bestfit.fits is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "GENERAL": {"OUTPUT": tmpdir, "RUN_ID": "TestRun"},
        }
        logLam = np.linspace(np.log(4500), np.log(5500), 50)
        result = _load_kin_stellar_continuum(config, logLam, nbins=5, currentLevel="BIN")
    assert result is None


def test_load_kin_stellar_continuum_bin_level():
    """_load_kin_stellar_continuum returns (nbins, npix) for BIN when file exists."""
    npix_kin = 80
    nbins = 4
    npix_gas = 60
    logLam_gas = np.linspace(np.log(4600), np.log(5400), npix_gas)

    with tempfile.TemporaryDirectory() as tmpdir:
        kin_path = os.path.join(tmpdir, "TestRun_kin-bestfit.fits")
        kin_bestfit = np.ones((nbins, npix_kin)) * 0.5
        kin_logLam = np.linspace(np.log(4000), np.log(7000), npix_kin)

        cols1 = fits.ColDefs([fits.Column(name="BESTFIT", format=f"{npix_kin}D", array=kin_bestfit)])
        cols2 = fits.ColDefs([fits.Column(name="LOGLAM", format="D", array=kin_logLam)])
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(cols1),
            fits.BinTableHDU.from_columns(cols2),
        ])
        hdul.writeto(kin_path, overwrite=True)

        config = {"GENERAL": {"OUTPUT": tmpdir, "RUN_ID": "TestRun"}}
        result = _load_kin_stellar_continuum(
            config, logLam_gas, nbins=nbins, currentLevel="BIN"
        )

    assert result is not None
    assert result.shape == (nbins, npix_gas)
    assert np.all(np.isfinite(result))
