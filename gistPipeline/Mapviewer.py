#!/usr/bin/env pythonw

from PyQt5 import QtCore, QtGui, QtWidgets as pyqt

import sys
import optparse
import warnings
warnings.filterwarnings("ignore")

from gistPipeline.mapviewer import createWindow    as _createWindow
from gistPipeline.mapviewer import createFigure    as _createFigure
from gistPipeline.mapviewer import loadData        as _loadData
from gistPipeline.mapviewer import plotData        as _plotData
from gistPipeline.mapviewer import helperFunctions as _helperFunctions


class Mapviewer(pyqt.QMainWindow):

    # Initialize the class
    def __init__(self, MODE, parent=None):

        # Setup window
        super(Mapviewer, self).__init__(parent)
        self.main_widget = pyqt.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Some default settings
        self.MODE                 = MODE
        self.restrict2voronoi     = 2
        self.markercolor          = 'r'
        self.GandalfLevelSelected = 'BIN'
        self.LsLevelSelected      = 'ADAPTED'
        self.AoNThreshold         = 3
        self.forAleksandra        = False

        # Setup the rest
        self.createFigure()
        self.createWindow()
        self.dialogRunSelection()


    # ========================
    # Function definitions

    # Create the GUI
    def createFigure(self):
        _createFigure.createFigure(self)
    def createWindow(self):
        _createWindow.createWindow(self)
    def loadData(self):
        _loadData.loadData(self)

    # Plot the data
    def plotMap(self, maptype):
        _plotData.plotMap(self, maptype)
    def plotData(self):
        _plotData.plotData(self)
    def plotSpectraPPXF(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraPPXF(self, spectra, bestfit, goodpix, panel)
    def plotSpectraGANDALF(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraGANDALF(self, spectra, bestfit, goodpix, panel)
    def plotSpectraSFH(self, spectra, bestfit, goodpix, panel):
        _plotData.plotSpectraSFH(self, spectra, bestfit, goodpix, panel)
    def plotPlainSpectrum(self, spectra, snr, panel):
        _plotData.plotPlainSpectrum(self, spectra, snr, panel)
    def plotSSPGrid(self, alpha_idx, panel):
        _plotData.plotSSPGrid(self, alpha_idx, panel)
    def plotSFH(self, panel):
        _plotData.plotSFH(self, panel)
    def plotPercentiles(self, panel, labels):
        _plotData.plotPercentiles(self, panel, labels)
    def plotLSSpectrum(self, panel):
        _plotData.plotLSSpectrum(self, panel)
    
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
    def dialogWrongDirectory(self):
        _helperFunctions.dialogWrongDirectory(self)
    def dialogSettings(self):
        _helperFunctions.dialogSettings(self)
    def dialogInfo(self):
        _helperFunctions.dialogInfo(self)
    def dialogBinID(self):
        _helperFunctions.dialogBinID(self)



# ==============================================================================
#                                    M A I N 
# ==============================================================================
def main(args=None):
    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="")
    parser.add_option("-m", "--mode", dest="MODE", type="string", help="Set 'LS' in order to highlight results from the LS module.")
    (options, args) = parser.parse_args()
    MODE = options.MODE
    if   MODE == None: MODE = 'ALL'
    elif MODE == 'LS'      or MODE == 'ls'      or MODE == 'L' or MODE == 'l': MODE = 'LS'

    # Start GUI application
    application = pyqt.QApplication(sys.argv)
    program = Mapviewer(MODE)
    program.showMaximized()
    program.setWindowTitle("Mapviewer")
    sys.exit( application.exec_() )


if __name__ == '__main__':
    # Call the main function
    main()
