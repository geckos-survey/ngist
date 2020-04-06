from    astropy.io import fits
import numpy       as np
import os

from PyQt5 import QtCore, QtGui, QtWidgets as pyqt
import matplotlib.pyplot as plt



def onpick(self, event):
    # Capture the event of performing a left-click on the map
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
    pyqt.QMessageBox.about(self, "About", "This is the 'GIST Pipeline'")


def dialogNotAvailable(self, which):
    # Show information if any selected data is not available
    if which == "directory":
        pyqt.QMessageBox.information(self, "Not available", "There is no data available! Please choose another directory! \n:( ")
    else:
        pyqt.QMessageBox.information(self, "Not available", which+" data is not available! :( ")


def dialogRunSelection(self):
    # Select the output directory of the run to be displayed. Basic checks and correction are applied.
    tmp0 = str( pyqt.QFileDialog.getExistingDirectory(self, "Select Directory", '.') )
    if len( tmp0 ) > 0:
        if tmp0.split('/')[-1] == 'maps':
            tmp0 = tmp0[:-5]
        elif tmp0.split('/')[-1] == 'results':
            self.dialogWrongDirectory()
            self.dialogRunSelection()
            return(None)
        tmp1 = tmp0.split('/')[-1].split('_')[-2]
        self.dirprefix = tmp0+'/'+tmp1
        self.directory = tmp0+'/'
        self.loadData()


def dialogWrongDirectory(self):
    pyqt.QMessageBox.warning(self, "Wrong Directory", \
         "You selected a wrong directory. Please navigate into the output directory, e.g. /results/NGC0000_Example/")


def dialogBinID(self):
    # Create the dialogue for passing the bin/spaxel ID to be displayed

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
    # Extract the chosen BIN_ID from the dialogBinID

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
    # Extract the chosen Spaxel ID from the dialogBinID

    try: 
        # Save index of chosen Voronoi-bin
        self.idxBinLong  = int(self.textbox.text())                       # In Spaxel arrays
        self.idxBinShort = self.table.BIN_ID[int(self.textbox.text())]    # In Bin arrays
    
        # Plot data
        self.plotData() 
    except: 
        self.textbox.setText("Error!")


def dialogSettings(self):
    # Create the setting dialogue

    dialog = pyqt.QDialog()

    # Header
    self.label0 = pyqt.QLabel('General settings:')

    # Restrict to Voronoi region
    self.checkbox0 = pyqt.QCheckBox('Restrict to Voronoi region', dialog)
    if self.restrict2voronoi == 2: self.checkbox0.setChecked(True)

    # Choose GANDALF level
    self.label_GandalfLevel          = pyqt.QLabel('\nDisplay Gandalf results on bin or spaxel level?')
    self.checkbox_GandalfLevelBIN    = pyqt.QCheckBox('Bin level', dialog)
    self.checkbox_GandalfLevelSPAXEL = pyqt.QCheckBox('Spaxel level', dialog)
    if len( self.GandalfLevelAvailable ) == 2: 
        self.checkbox_GandalfLevelBIN.setEnabled(True)
        self.checkbox_GandalfLevelSPAXEL.setEnabled(True)
    else:
        self.checkbox_GandalfLevelBIN.setEnabled(False)
        self.checkbox_GandalfLevelSPAXEL.setEnabled(False)
    if self.GandalfLevel == 'BIN':
        self.checkbox_GandalfLevelBIN.setChecked(True)
    elif self.GandalfLevel == 'SPAXEL':
        self.checkbox_GandalfLevelSPAXEL.setChecked(True)
    self.GandalfLevel_ButtonGroup = pyqt.QButtonGroup()
    self.GandalfLevel_ButtonGroup.addButton(self.checkbox_GandalfLevelBIN,1)
    self.GandalfLevel_ButtonGroup.addButton(self.checkbox_GandalfLevelSPAXEL,2)

    # Choose AoN Threshold for Gandalf
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
    setlay.addWidget(self.label_GandalfLevel)
    setlay.addWidget(self.checkbox_GandalfLevelBIN)
    setlay.addWidget(self.checkbox_GandalfLevelSPAXEL)
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
    # Extract the settings from the dialogSettings and apply them

    self.restrict2voronoi = self.checkbox0.checkState()

    if self.checkbox_GandalfLevelBIN.checkState() == 2  and  \
       self.GandalfLevel != 'BIN'  and  self.GANDALF == True:
        self.GandalfLevelSelected = 'BIN'
        self.loadData()
    elif self.checkbox_GandalfLevelSPAXEL.checkState() == 2  and  \
         self.GandalfLevel != 'SPAXEL'  and  self.GANDALF == True:
        self.GandalfLevelSelected = 'SPAXEL'
        self.loadData()

    if self.AoNThreshold != float( self.AoNThreshold_Input.text() )  and  self.GANDALF == True:
        self.AoNThreshold = float( self.AoNThreshold_Input.text() )

    if self.checkbox_LsLevelORIG.checkState() == 2  and  \
       self.LsLevel != 'ORIGINAL'  and  self.LINE_STRENGTH == True:
        self.LsLevelSelected = 'ORIGINAL'
        self.loadData()
    elif self.checkbox_LsLevelADAP.checkState() == 2  and  \
         self.LsLevel != 'ADAPTED'  and  self.LINE_STRENGTH == True:
        self.LsLevelSelected = 'ADAPTED'
        self.loadData()

    if self.checkbox1.checkState() == 2: self.markercolor = 'r'
    if self.checkbox2.checkState() == 2: self.markercolor = 'g'
    if self.checkbox3.checkState() == 2: self.markercolor = 'b'


def dialogInfo(self):
    # Print the info dialogue with the information from the Config-file

    # Load data
    info = fits.open(self.directory+"USED_PARAMS.fits")[0].header
    keyword_list = list(info.keys())
    keyword_list.remove("SIMPLE")
    keyword_list.remove("BITPIX")
    keyword_list.remove("NAXIS")
    keyword_list.remove("EXTEND")
    keyword_list = [y for y in keyword_list if y != 'COMMENT']

    # Create InfoDialog
    dialog = pyqt.QDialog()
    self.label0 = pyqt.QLabel('Chosen Configurations:')
    closeButton = pyqt.QPushButton("Close", dialog)

    # Create table
    self.tableWidget = pyqt.QTableWidget()
    self.tableWidget.setRowCount(len(keyword_list))
    self.tableWidget.setColumnCount(2)
    self.tableWidget.setSizeAdjustPolicy(pyqt.QAbstractScrollArea.AdjustToContents)
    self.tableWidget.setHorizontalHeaderLabels(["Configs", "Values"])
    self.tableWidget.setVerticalHeaderLabels([""]*len(keyword_list))

    for i, o in enumerate(keyword_list):
        self.tableWidget.setItem(i,0, pyqt.QTableWidgetItem(o      ))
        self.tableWidget.setItem(i,1, pyqt.QTableWidgetItem(info[o]))
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
