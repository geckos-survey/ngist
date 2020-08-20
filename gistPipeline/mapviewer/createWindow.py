from PyQt5 import QtCore, QtGui, QtWidgets as pyqt



def returnPlotFunction(self, module, name):
    """ 
    Function to connect the correct plotting routine for the maps to the entries
    in the menu bar. 
    """
    return lambda: self.plotMap(module, name)


def createWindow(self):
    """
    This function creates the menu bar which allows to plot the measured
    quantities as maps. The entries in the menu bar are automatically generated,
    depending on what columns are included in the output files. Note that this
    allows to display any non-default output columns in Mapviewer. 
    """

    # Create main menu bar
    self.menuBar().clear()
    self.menuBar().setNativeMenuBar(False)
    fileMenu  = self.menuBar().addMenu('File')
    tableMenu = self.menuBar().addMenu('Table')
    maskMenu  = self.menuBar().addMenu('Mask')
    kinMenu   = self.menuBar().addMenu('Kinematics')
    gasMenu   = self.menuBar().addMenu('Emission Lines')
    sfhMenu   = self.menuBar().addMenu('SFH')
    lsMenu    = self.menuBar().addMenu('Line Strength')
    aboutMenu = self.menuBar().addMenu('About')

    if self.forAleksandra == True: 
        # This is an easter egg :)
        self.menuBar().setStyleSheet("""
          QMenuBar       { background-color: #FF00FF }
          QMenuBar::item { background-color: #FF00FF }
          QMenu          { background-color: #FF00FF }
        """)


    # ====================
    # FILE MENU
    openButton = pyqt.QAction('Open', self)
    openButton.triggered.connect(self.dialogRunSelection)
    openButton.setShortcut('Ctrl+O')
    fileMenu.addAction(openButton) 

    settingsButton = pyqt.QAction('Settings', self)
    settingsButton.triggered.connect(self.dialogSettings)
    settingsButton.setShortcut('Ctrl+S')
    fileMenu.addAction(settingsButton) 

    binidButton = pyqt.QAction('Select ID', self)
    binidButton.triggered.connect(self.dialogBinID)
    binidButton.setShortcut('Ctrl+B')
    fileMenu.addAction(binidButton) 

    InfoButton = pyqt.QAction('Info', self)
    InfoButton.triggered.connect(self.dialogInfo)
    InfoButton.setShortcut('Ctrl+I')
    fileMenu.addAction(InfoButton) 

    exitButton = pyqt.QAction('Exit', self)
    exitButton.setShortcut('Ctrl+W')
    exitButton.setStatusTip('Exit application')
    exitButton.triggered.connect(self.close)
    fileMenu.addSeparator()
    fileMenu.addAction(exitButton) 


    # ====================
    # TABLE MENU
    for name in self.table.names:
        if name not in ['ID','X','Y','XBIN','YBIN']:
            button = tableMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'TABLE', name))


    # ====================
    # MASK MENU
    if self.MASK == True:
        for name in self.Mask.names:
            button = maskMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'MASK', name))
    elif self.MASK == False: 
        button = maskMenu.addAction("  Not available.  ")
        button.setDisabled(True)


    # ====================
    # stellarKinematics Menu
    if self.KIN == True: 
        for name in self.kinResults.names:
            button = kinMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'KIN', name))
    elif self.KIN == False: 
        button = kinMenu.addAction("  Not available.  ")
        button.setDisabled(True)


    # ====================
    # emissionLines Menu
    if self.GAS == True:
        for name in self.gasResults.names:
            button = gasMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'GAS', name))
    elif self.GAS == False:
        button = gasMenu.addAction("  Not available.  ")
        button.setDisabled(True)


    # ====================
    # starFormationHistories Menu
    if self.SFH == True:
        for name in self.sfhResults.names:
            button = sfhMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'SFH', name))
        if 'V' in self.sfhResults.names:
            sfhKinematicsDiff = pyqt.QMenu('Kinematics Difference', self)
            for name in self.sfhResults.names:
                if name in ['V', 'SIGMA', 'H3', 'H4', 'H5', 'H6']:
                    button = sfhKinematicsDiff.addAction(name+'_DIFF')
                    button.triggered.connect(returnPlotFunction(self, 'SFH', name+'_DIFF'))
            sfhMenu.addMenu(sfhKinematicsDiff) 
    elif self.SFH == False: 
        button = sfhMenu.addAction("  Not available.  ")
        button.setDisabled(True)


    # ====================
    # lineStrengths Menu
    if self.LINE_STRENGTH == True:
        for name in self.lsResults.names:
            button = lsMenu.addAction(name)
            button.triggered.connect(returnPlotFunction(self, 'LS', name))
    elif self.LINE_STRENGTH == False:
        button = lsMenu.addAction("  Not available.  ")
        button.setDisabled(True)


    # ====================
    # ABOUT MENU
    aboutButton = pyqt.QAction('About this pipeline', self)
    aboutButton.triggered.connect(self.dialogAbout)
    aboutMenu.addAction(aboutButton)


