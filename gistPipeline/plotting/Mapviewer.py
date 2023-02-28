#!/usr/bin/env pythonw

from PyQt5 import QtCore, QtGui, QtWidgets as pyqt

import sys
import warnings
warnings.filterwarnings("ignore")

from gistPipeline.mapviewer import createWindow    as _createWindow
from gistPipeline.mapviewer import createFigure    as _createFigure
from gistPipeline.mapviewer import loadData        as _loadData
from gistPipeline.mapviewer import plotData        as _plotData
from gistPipeline.mapviewer import helperFunctions as _helperFunctions
from gistPipeline._version import __version__


class Mapviewer(pyqt.QMainWindow):

    # Initialize the class
    def __init__(self, parent=None):

        # Setup window
        super(Mapviewer, self).__init__(parent)
        self.main_widget = pyqt.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Some default settings
        self.restrict2voronoi = 2
        self.markercolor      = 'r'
        self.gasLevelSelected = 'BIN'
        self.LsLevelSelected  = 'ADAPTED'
        self.AoNThreshold     = 3
        self.forAleksandra    = False

        # Setup the rest
        self.createFigure()
        self.dialogRunSelection()


    # ========================
    # Function definitions

    # Create the GUI
    def loadData(self):
        _loadData.loadData(self)
    def createFigure(self):
        _createFigure.createFigure(self)
    def createWindow(self):
        _createWindow.createWindow(self)

    # Plot the data
    def plotMap(self, module, maptype):
        _plotData.plotMap(self, module, maptype)
    def plotData(self):
        _plotData.plotData(self)
    def plotSpectraKIN(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraKIN(self, spectra, bestfit, goodpix, panel)
    def plotSpectraGAS(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraGAS(self, spectra, bestfit, goodpix, panel)
    def plotSpectraSFH(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraSFH(self, spectra, bestfit, goodpix, panel)
    def plotPlainSpectrum(self, spectra, snr, panel):
        _plotData.plotPlainSpectrum(self, spectra, snr, panel)
    def plotSSPGrid(self, alpha_idx, panel):
        _plotData.plotSSPGrid(self, alpha_idx, panel)
    def plotSFH(self, panel):
        _plotData.plotSFH(self, panel)
    
    # Relevant helper functions
    def onpick(self, event):
        _helperFunctions.onpick(self,event)
    def defineGoodPixels(self):
        _loadData.defineGoodPixels(self)
    def getVoronoiBin(self):
        _helperFunctions.getVoronoiBin(self)
    def getSettings(self):
        _helperFunctions.getSettings(self)
    def selectBinIDfromDialog(self):
        _helperFunctions.selectBinIDfromDialog(self)
    def selectSpaxelIDfromDialog(self):
        _helperFunctions.selectSpaxelIDfromDialog(self)

    # Dialogs 
    def dialogAbout(self):
        _helperFunctions.dialogAbout(self)
    def dialogNotAvailable(self, which):
        _helperFunctions.dialogNotAvailable(self, which)
    def dialogRunSelection(self):
        _helperFunctions.dialogRunSelection(self)
    def dialogSettings(self):
        _helperFunctions.dialogSettings(self)
    def dialogInfo(self):
        _helperFunctions.dialogInfo(self)
    def dialogBinID(self):
        _helperFunctions.dialogBinID(self)



# ============================================================================ #
#                                    M A I N                                   #
# ============================================================================ #
def main(args=None):
    # Start GUI application
    application = pyqt.QApplication(sys.argv)
    program = Mapviewer()
    program.showMaximized()
    program.setWindowTitle("Mapviewer  ---  GIST V"+__version__)
    sys.exit( application.exec_() )


if __name__ == '__main__':
    # Call the main function
    main()


