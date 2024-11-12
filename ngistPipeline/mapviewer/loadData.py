import os

import h5py
import numpy as np
from astropy.io import fits


def loadData(self):
    """Load all available data to make it accessible for plotting."""

    # Clean all figures
    try:
        for i in range(len(self.axes)):
            self.axes[i].cla()
        self.cax.cla()

        self.cax3.cla()
        self.cax4.cla()
    except:
        pass

    try:
        self.canvas.draw()
    except:
        pass

    # Check if *_table.fits is available. This file is required!
    if os.path.isfile(self.dirprefix + "_table.fits") == False:
        print("You did not select a valid GIST output directory.")
        self.dialogRunSelection()

    # Determine which data is available
    self.MASK = os.path.isfile(self.dirprefix + "_mask.fits")

    self.KIN = os.path.isfile(self.dirprefix + "_kin.fits")

    if (
        os.path.isfile(self.dirprefix + "_gas_BIN.fits") == False
        and os.path.isfile(self.dirprefix + "_gas_SPAXEL.fits") == False
    ):
        self.GAS = False
        self.gasLevelAvailable = []
    if (
        os.path.isfile(self.dirprefix + "_gas_BIN.fits") == True
        and os.path.isfile(self.dirprefix + "_gas_SPAXEL.fits") == False
    ):
        self.GAS = True
        self.gasLevelAvailable = ["BIN"]
    if (
        os.path.isfile(self.dirprefix + "_gas_BIN.fits") == False
        and os.path.isfile(self.dirprefix + "_gas_SPAXEL.fits") == True
    ):
        self.GAS = True
        self.gasLevelAvailable = ["SPAXEL"]
    if (
        os.path.isfile(self.dirprefix + "_gas_BIN.fits") == True
        and os.path.isfile(self.dirprefix + "_gas_SPAXEL.fits") == True
    ):
        self.GAS = True
        self.gasLevelAvailable = ["BIN", "SPAXEL"]

    if len(self.gasLevelAvailable) == 1:
        self.gasLevel = self.gasLevelAvailable[0]
    else:
        self.gasLevel = self.gasLevelSelected

    self.SFH = os.path.isfile(self.dirprefix + "_sfh.fits")

    if (
        os.path.isfile(self.dirprefix + "_ls_OrigRes.fits") == False
        and os.path.isfile(self.dirprefix + "_ls_AdapRes.fits") == False
    ):
        self.LINE_STRENGTH = False
        self.LsLevelAvailable = []
    if (
        os.path.isfile(self.dirprefix + "_ls_OrigRes.fits") == True
        and os.path.isfile(self.dirprefix + "_ls_AdapRes.fits") == False
    ):
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ["ORIGINAL"]
    if (
        os.path.isfile(self.dirprefix + "_ls_OrigRes.fits") == False
        and os.path.isfile(self.dirprefix + "_ls_AdapRes.fits") == True
    ):
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ["ADAPTED"]
    if (
        os.path.isfile(self.dirprefix + "_ls_OrigRes.fits") == True
        and os.path.isfile(self.dirprefix + "_ls_AdapRes.fits") == True
    ):
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ["ORIGINAL", "ADAPTED"]

    if len(self.LsLevelAvailable) == 1:
        self.LsLevel = self.LsLevelAvailable[0]
    else:
        self.LsLevel = self.LsLevelSelected

    # ======================================================== #
    #                    R E A D   D A T A                     #
    # ======================================================== #

    # Read table and get transformation array
    self.table = fits.open(self.dirprefix + "_table.fits")[1].data
    self.pixelsize = fits.open(self.dirprefix + "_table.fits")[0].header["PIXSIZE"]
    _, idxConvertShortToLong = np.unique(np.abs(self.table.BIN_ID), return_inverse=True)

    # Read spectra
    hdf5_file = self.dirprefix + "_BinSpectra.hdf5"
    fits_file = self.dirprefix + "_BinSpectra.fits"

    if os.path.isfile(hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            self.Spectra = f['SPEC'][:].T
            self.Lambda = f['LOGLAM'][:]
    else:
        print(hdf5_file + " does not exist. Trying " + fits_file)
        self.Spectra = fits.open(fits_file)[1].data.SPEC
        self.Lambda = fits.open(fits_file)[2].data.LOGLAM

    nbins = self.Spectra.shape[0]

    if self.gasLevel == "SPAXEL":
        hdf5_file = self.dirprefix + "_AllSpectra.hdf5"
        fits_file = self.dirprefix + "_AllSpectra.fits"

        if os.path.isfile(hdf5_file):
            print('Loading All Spectra for GAS SPX mode. This could take some time')
            with h5py.File(hdf5_file, 'r') as f:
                self.AllSpectra = f['SPEC'][:].T
        else:
            print(hdf5_file + " does not exist. Trying " + fits_file)
            self.AllSpectra = fits.open(fits_file)[1].data.SPEC

    # Read mask
    if self.MASK == True:
        self.Mask = fits.open(self.dirprefix + "_mask.fits")[1].data

    # Read stellar kinematics
    if self.KIN:
        self.kinResults = fits.open(self.dirprefix + "_kin.fits")[1].data[
            idxConvertShortToLong
        ]
        self.kinBestfit = fits.open(self.dirprefix + "_kin-bestfit.fits")[
            1
        ].data.BESTFIT
        self.kinLambda = fits.open(self.dirprefix + "_kin-bestfit.fits")[2].data.LOGLAM
        self.kinGoodpix = fits.open(self.dirprefix + "_kin-bestfit.fits")[
            3
        ].data.GOODPIX

        # following line does not work if your data is not symetric around centre
        #median_V_stellar = np.nanmedian(
        #    self.kinResults.V[np.where(self.table.BIN_ID >= 0)[0]]
        #)
        #self.kinResults.V = self.kinResults.V - median_V_stellar
    else:
        self.kinResults = None
        self.kinBestfit = None
        self.kinLambda = None
        self.kinGoodpix = None

    # Read emissionLines results
    if self.GAS:
        if os.path.isfile(self.dirprefix + "_gas-cleaned_BIN.fits") == True:
            self.EmissionSubtractedSpectraBIN = np.array(
                fits.open(self.dirprefix + "_gas-cleaned_BIN.fits")[1].data.SPEC
            )
        if os.path.isfile(self.dirprefix + "_gas-cleaned_SPAXEL.fits") == True:
            self.EmissionSubtractedSpectraSPAXEL = np.array(
                fits.open(self.dirprefix + "_gas-cleaned_SPAXEL.fits")[1].data.SPEC
            )

        gas = fits.open(self.dirprefix + "_gas_" + self.gasLevel + ".fits")[1].data
        self.gasBestfit = fits.open(
            self.dirprefix + "_gas-bestfit_" + self.gasLevel + ".fits"
        )[1].data.BESTFIT
        self.gasLambda = fits.open(
            self.dirprefix + "_gas-bestfit_" + self.gasLevel + ".fits"
        )[2].data.LOGLAM
        self.gasGoodpix = fits.open(
            self.dirprefix + "_gas-bestfit_" + self.gasLevel + ".fits"
        )[3].data.GOODPIX

        if self.gasLevel == "BIN":
            self.gasResults = gas[idxConvertShortToLong]
        if self.gasLevel == "SPAXEL":
            self.gasResults = gas

        # following line does not work if your data is not symetric around centre
        #for name in self.gasResults.names:
        #    if name.split("_")[-1] == "V":
        #        self.gasResults[name] = self.gasResults[name] - median_V_stellar
    else:
        self.gasResults = None
        self.gasBestfit = None
        self.gasLambda = None
        self.gasGoodpix = None

    # Read starFormatioHistories results
    if self.SFH:
        self.sfhResults = fits.open(self.dirprefix + "_sfh.fits")[1].data[
            idxConvertShortToLong
        ]
        self.sfhBestfit = fits.open(self.dirprefix + "_sfh-bestfit.fits")[
            1
        ].data.BESTFIT
        self.sfhLambda = fits.open(self.dirprefix + "_sfh-bestfit.fits")[2].data.LOGLAM
        self.sfhGoodpix = fits.open(self.dirprefix + "_sfh-bestfit.fits")[
            3
        ].data.GOODPIX

        # following line does not work if your data is not symetric around centre
        #if "V" in self.sfhResults.names:
        #    self.sfhResults.V = self.sfhResults.V - median_V_stellar

        # Read the age, metallicity and [Mg/Fe] grid
        grid = fits.open(self.dirprefix + "_sfh-weights.fits")[2].data
        self.metals = np.unique(grid.METAL)
        self.age = np.power(10, np.unique(grid.LOGAGE))

        # Read weights
        hdu_weights = fits.open(self.dirprefix + "_sfh-weights.fits")
        nAges = hdu_weights[0].header["NAGES"]
        nMetal = hdu_weights[0].header["NMETAL"]
        nAlpha = hdu_weights[0].header["NALPHA"]
        self.Weights = np.reshape(
            np.array(hdu_weights[1].data.WEIGHTS), (nbins, nAges, nMetal, nAlpha)
        )
        self.Weights = np.transpose(self.Weights, (0, 2, 1, 3))
    else:
        self.sfhResults = None
        self.sfhBestfit = None
        self.sfhLambda = None
        self.sfhGoodpix = None
        self.metals = None
        self.age = None
        self.Weights = None

    # Read lineStrengths results
    if self.LINE_STRENGTH == True:
        if self.LsLevel == "ORIGINAL":
            ls = fits.open(self.dirprefix + "_ls_OrigRes.fits")[1].data
        elif self.LsLevel == "ADAPTED":
            ls = fits.open(self.dirprefix + "_ls_AdapRes.fits")[1].data
        self.lsResults = ls[idxConvertShortToLong]
    else:
        self.lsResults = None
