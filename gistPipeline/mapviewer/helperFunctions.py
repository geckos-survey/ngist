import numpy as np
import os
import yaml

from PyQt5 import QtCore, QtGui, QtWidgets as pyqt

from gistPipeline._version import __version__



def onpick(self, event):
    """ Capture the event of performing a left-click on the map """
    self.mouse = event.mouseevent
    if self.mouse.button == 1:
        self.getVoronoiBin()


def getVoronoiBin(self):
    """
    Identify the Voronoi-bin/spaxel closest to the clicked location on the map
    """
    idx = np.where( np.logical_and( np.abs(self.table.X-self.mouse.xdata) < self.pixelsize, np.abs(self.table.Y-self.mouse.ydata) < self.pixelsize ) )[0]

    if len(idx) == 1:
        final_idx = idx[0]
    elif len(idx) == 4:
        xmini = np.argsort( np.abs( self.table.X[idx]        - self.mouse.xdata ) )[:2]
        ymini = np.argmin(  np.abs( self.table.Y[idx[xmini]] - self.mouse.ydata ) )
        final_idx = idx[xmini[ymini]]
    else:
        return(None)

    # Save index of chosen Voronoi-bin
    self.idxBinLong  = final_idx                            # In Spaxel arrays
    self.idxBinShort = np.abs(self.table.BIN_ID[final_idx]) # In Bin arrays

    # Plot data
    self.plotData() 


def dialogAbout(self):
    message = "{:10}\n{:10}\n\n{:10}".format("The GIST Pipeline", "Version "+__version__, "For a thorough documentation of this software package, please see https://abittner.gitlab.io/thegistpipeline")
    pyqt.QMessageBox.about(self, "About", message)


def dialogNotAvailable(self, which):
    """ Show information if any selected data is not available """
    if which == "directory":
        pyqt.QMessageBox.information(self, "Not available", "There is no data available! Please choose another directory! \n:( ")
    else:
        pyqt.QMessageBox.information(self, "Not available", which+" data is not available! :( ")


def dialogRunSelection(self):
    """ 
    Select the output directory of the run to be displayed. Basic checks 
    and corrections are applied.
    """
    tmp0 = str( pyqt.QFileDialog.getExistingDirectory(self, "Select Directory", '.') )
    if len( tmp0 ) > 0:
        if tmp0.split('/')[-1] == 'maps':
            tmp0 = tmp0[:-5]
        tmp1 = tmp0.split('/')[-1]
        self.dirprefix = os.path.join(tmp0,tmp1)
        self.directory = tmp0+'/'

        self.loadData()
        self.createWindow()


def dialogBinID(self):
    """ Create the dialogue for passing the bin/spaxel ID to be displayed """

    dialog=pyqt.QDialog()

    label = pyqt.QLabel('Please select the bin or spaxel ID:              ')

    self.textbox = pyqt.QLineEdit(self)

    buttonBIN = pyqt.QPushButton('BIN ID', self)
    buttonBIN.clicked.connect(self.selectBinIDfromDialog)
    buttonSPAXEL = pyqt.QPushButton('SPAXEL ID', self)
    buttonSPAXEL.clicked.connect(self.selectSpaxelIDfromDialog)
    closeButton = pyqt.QPushButton('Close', self)
    closeButton.clicked.connect(dialog.close)

    setlay = pyqt.QGridLayout()
    setlay.addWidget(label,        0, 0, 1, 2 )
    setlay.addWidget(self.textbox, 1, 0, 1, 2 )
    setlay.addWidget(buttonBIN,    2, 0, 1, 1 )
    setlay.addWidget(buttonSPAXEL, 2, 1, 1, 1 )
    setlay.addWidget(closeButton,  3, 0, 1, 2 )
    dialog.setLayout(setlay)

    dialog.setWindowTitle("Select BIN_ID")
    dialog.exec_()


def selectBinIDfromDialog(self):
    """ Extract the chosen BIN_ID from the dialogBinID """

    try: 
        # Save index of chosen Voronoi-bin
        self.idxBinShort = int(self.textbox.text())     # In Bin arrays
        
        allIdx = np.where( self.table.BIN_ID == self.idxBinShort )[0]
        minIdx = np.argmin( np.sqrt( ( self.table.X[allIdx] - self.table.XBIN[allIdx] )**2 + ( self.table.Y[allIdx] - self.table.YBIN[allIdx] )**2 ) ) # Central spaxel of bin
        self.idxBinLong = allIdx[minIdx]
    
        # Plot data
        self.plotData() 
    except: 
        self.textbox.setText("Error!")


def selectSpaxelIDfromDialog(self):
    """ Extract the chosen Spaxel ID from the dialogBinID """
    
    try: 
        # Save index of chosen Voronoi-bin
        self.idxBinLong  = int(self.textbox.text())                       # In Spaxel arrays
        self.idxBinShort = self.table.BIN_ID[int(self.textbox.text())]    # In Bin arrays
    
        # Plot data
        self.plotData() 
    except: 
        self.textbox.setText("Error!")


def dialogSettings(self):
    """ Create the setting dialogue """

    dialog = pyqt.QDialog()

    # Header
    self.label0 = pyqt.QLabel('General settings:')

    # Restrict to Voronoi region
    self.checkbox0 = pyqt.QCheckBox('Restrict to Voronoi region', dialog)
    if self.restrict2voronoi == 2: self.checkbox0.setChecked(True)

    # Choose gas level
    self.label_gasLevel          = pyqt.QLabel('\nDisplay emissionLines results on bin or spaxel level?')
    self.checkbox_gasLevelBIN    = pyqt.QCheckBox('Bin level', dialog)
    self.checkbox_gasLevelSPAXEL = pyqt.QCheckBox('Spaxel level', dialog)
    if len( self.gasLevelAvailable ) == 2: 
        self.checkbox_gasLevelBIN.setEnabled(True)
        self.checkbox_gasLevelSPAXEL.setEnabled(True)
    else:
        self.checkbox_gasLevelBIN.setEnabled(False)
        self.checkbox_gasLevelSPAXEL.setEnabled(False)
    if self.gasLevel == 'BIN':
        self.checkbox_gasLevelBIN.setChecked(True)
    elif self.gasLevel == 'SPAXEL':
        self.checkbox_gasLevelSPAXEL.setChecked(True)
    self.gasLevel_ButtonGroup = pyqt.QButtonGroup()
    self.gasLevel_ButtonGroup.addButton(self.checkbox_gasLevelBIN,1)
    self.gasLevel_ButtonGroup.addButton(self.checkbox_gasLevelSPAXEL,2)

    # Choose AoN Threshold for emissionLines results
    self.label_AoNThreshold = pyqt.QLabel('\nAoN Threshold for displayed line detections:')
    self.AoNThreshold_Input = pyqt.QLineEdit()
    self.AoNThreshold_Input.setText(str(self.AoNThreshold))

    # Choose LS level
    self.label_LsLevel        = pyqt.QLabel('\nDisplay LS results measured on adapted or original resolution?')
    self.checkbox_LsLevelORIG = pyqt.QCheckBox('Original resolution', dialog)
    self.checkbox_LsLevelADAP = pyqt.QCheckBox('Adapted resolution', dialog)
    if len( self.LsLevelAvailable ) == 2: 
        self.checkbox_LsLevelORIG.setEnabled(True)
        self.checkbox_LsLevelADAP.setEnabled(True)
    else:
        self.checkbox_LsLevelORIG.setEnabled(False)
        self.checkbox_LsLevelADAP.setEnabled(False)
    if self.LsLevel == 'ORIGINAL':
        self.checkbox_LsLevelORIG.setChecked(True)
    elif self.LsLevel == 'ADAPTED':
        self.checkbox_LsLevelADAP.setChecked(True)
    self.LsLevel_ButtonGroup = pyqt.QButtonGroup()
    self.LsLevel_ButtonGroup.addButton(self.checkbox_LsLevelORIG,1)
    self.LsLevel_ButtonGroup.addButton(self.checkbox_LsLevelADAP,2)

    # Choose color of marker
    self.label1 = pyqt.QLabel('\nSelect color of marker:')
    self.checkbox1 = pyqt.QCheckBox('Red', dialog)
    self.checkbox2 = pyqt.QCheckBox('Green', dialog)
    self.checkbox3 = pyqt.QCheckBox('Blue', dialog)
    if self.markercolor == 'r':
        self.checkbox1.setChecked(True)
    elif self.markercolor == 'g':
        self.checkbox2.setChecked(True)
    elif self.markercolor == 'b':
        self.checkbox3.setChecked(True)
    self.bg = pyqt.QButtonGroup()
    self.bg.addButton(self.checkbox1,1)
    self.bg.addButton(self.checkbox2,2)
    self.bg.addButton(self.checkbox3,3)

    # Apply and close buttons
    okButton = pyqt.QPushButton("Apply", dialog)
    closeButton = pyqt.QPushButton("Close", dialog)

    # Set layout
    setlay = pyqt.QVBoxLayout()
    setlay.addWidget(self.label0)
    setlay.addWidget(self.checkbox0)
    #
    setlay.addWidget(self.label_gasLevel)
    setlay.addWidget(self.checkbox_gasLevelBIN)
    setlay.addWidget(self.checkbox_gasLevelSPAXEL)
    #
    setlay.addWidget(self.label_LsLevel)
    setlay.addWidget(self.checkbox_LsLevelORIG)
    setlay.addWidget(self.checkbox_LsLevelADAP)
    #
    setlay.addWidget(self.label_AoNThreshold)
    setlay.addWidget(self.AoNThreshold_Input)
    #
    setlay.addWidget(self.label1)
    setlay.addWidget(self.checkbox1)
    setlay.addWidget(self.checkbox2)
    setlay.addWidget(self.checkbox3)
    #
    setlay.addWidget(okButton)
    setlay.addWidget(closeButton)
    dialog.setLayout(setlay)

    okButton.clicked.connect(self.getSettings)
    closeButton.clicked.connect(dialog.close)
    dialog.setWindowTitle("Settings")
    dialog.exec_()


def getSettings(self):
    """ Extract the settings from the dialogSettings and apply them """

    self.restrict2voronoi = self.checkbox0.checkState()

    if self.checkbox_gasLevelBIN.checkState() == 2  and  \
       self.gasLevel != 'BIN'  and  self.GAS == True:
        self.gasLevelSelected = 'BIN'
        self.loadData()
        self.createWindow()
    elif self.checkbox_gasLevelSPAXEL.checkState() == 2  and  \
         self.gasLevel != 'SPAXEL'  and  self.GAS == True:
        self.gasLevelSelected = 'SPAXEL'
        self.loadData()
        self.createWindow()

    if self.AoNThreshold != float( self.AoNThreshold_Input.text() )  and  self.GAS == True:
        self.AoNThreshold = float( self.AoNThreshold_Input.text() )

    if self.checkbox_LsLevelORIG.checkState() == 2  and  \
       self.LsLevel != 'ORIGINAL'  and  self.LINE_STRENGTH == True:
        self.LsLevelSelected = 'ORIGINAL'
        self.loadData()
        self.createWindow()
    elif self.checkbox_LsLevelADAP.checkState() == 2  and  \
         self.LsLevel != 'ADAPTED'  and  self.LINE_STRENGTH == True:
        self.LsLevelSelected = 'ADAPTED'
        self.loadData()
        self.createWindow()

    if self.checkbox1.checkState() == 2: self.markercolor = 'r'
    if self.checkbox2.checkState() == 2: self.markercolor = 'g'
    if self.checkbox3.checkState() == 2: self.markercolor = 'b'


def dialogInfo(self):
    """ Print the info dialogue with the information from the Config-file """

    # Load data
    with open(self.directory+'CONFIG', "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    totalLength = 0
    for i in conf.keys():
        totalLength += len(conf[i])

    # Create InfoDialog
    dialog = pyqt.QDialog()
    self.label0 = pyqt.QLabel('Chosen Configurations:')
    closeButton = pyqt.QPushButton("Close", dialog)

    # Create table
    self.tableWidget = pyqt.QTableWidget()
    self.tableWidget.setRowCount(totalLength)
    self.tableWidget.setColumnCount(3)
    self.tableWidget.setSizeAdjustPolicy(pyqt.QAbstractScrollArea.AdjustToContents)
    self.tableWidget.setHorizontalHeaderLabels(["Module", "Configs", "Values"])
    self.tableWidget.setVerticalHeaderLabels([""]*totalLength)

    # Add entries
    counter = 0
    for i in conf.keys():
        self.tableWidget.setItem(counter, 0, pyqt.QTableWidgetItem(i))
        for o, key in enumerate(conf[i]):
            self.tableWidget.setItem(counter, 1, pyqt.QTableWidgetItem(key))
            self.tableWidget.setItem(counter, 2, pyqt.QTableWidgetItem(str(conf[i][key])))
            counter += 1
    self.tableWidget.move(0,0)

    # Set layout
    setlay = pyqt.QVBoxLayout()
    setlay.addWidget(self.label0)
    setlay.addWidget(self.tableWidget)
    setlay.addWidget(closeButton)
    dialog.setLayout(setlay)

    closeButton.clicked.connect(dialog.close)
    self.tableWidget.resizeColumnsToContents()

    dialog.setWindowTitle("Info")
    dialog.exec_()


