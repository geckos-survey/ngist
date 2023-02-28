import matplotlib.pyplot     as plt
from matplotlib.gridspec     import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg    as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtCore, QtGui, QtWidgets as pyqt



def createFigure(self):
    """
    This function creates the matplotlib figure to be displayed in the GUI of
    the Mapviewer.
    """

    # Create figure and gridspec
    self.figure = plt.figure(figsize=(15,10), dpi=100)
    self.gs = GridSpec(4, 3, width_ratios=[1, 0.75, 0.75])
    
    # Create panels
    self.axes = []
    self.axes.append( plt.subplot( self.gs[:3,0] ) )  # Map
    self.axes.append( plt.subplot( self.gs[0,1:]  ) ) # stellarKinematics
    self.axes.append( plt.subplot( self.gs[1,1:]  ) ) # emissionLines
    self.axes.append( plt.subplot( self.gs[2,1:]  ) ) # SFH spectrum and fit
    self.axes.append( plt.subplot( self.gs[3,1]  ) )  # Age-Metallicity for alpha=0.00
    self.axes.append( plt.subplot( self.gs[3:4,0]) )  # Star formation history
    self.axes.append( plt.subplot( self.gs[3,2]  ) )  # Age-Metallicity for alpha=0.40

    # Create colorbars
    self.divider = make_axes_locatable(self.axes[0])
    self.cax     = self.divider.append_axes("right", size="5%", pad=0.1)
    self.div4    = make_axes_locatable(self.axes[6])
    self.cax4    = self.div4.append_axes("right", size="2.5%", pad=0.05)
 
    # Activate picker and show plot
    self.axes[0].set_picker(True)

    # Add plot to window
    self.canvas = FigureCanvas(self.figure)
    self.toolbar = NavigationToolbar(self.canvas, self)
    self.canvas.mpl_connect('pick_event', self.onpick)
    self.canvas.draw()

    layout = pyqt.QVBoxLayout()
    layout.addWidget(self.toolbar)
    layout.addWidget(self.canvas)
    self.main_widget.setLayout(layout)

 
