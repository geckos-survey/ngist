import matplotlib.pyplot     as plt
from matplotlib.gridspec     import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def createFigure(self):
    """
    This function creates the matplotlib figure to be displayed in the GUI of
    the Mapviewer. The layout of the different modes are also defined here. 
    """

    if self.MODE == 'ALL':

        # Create figure and gridspec
        self.figure = plt.figure(figsize=(15,10), dpi=100)
        self.gs = GridSpec(4, 3, width_ratios=[1, 0.75, 0.75])
        
        # Create panels
        self.axes = []
        self.axes.append( plt.subplot( self.gs[:3,0] ) )  # Map
        self.axes.append( plt.subplot( self.gs[0,1:]  ) ) # PPXF
        self.axes.append( plt.subplot( self.gs[1,1:]  ) ) # GANDALF
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


    if self.MODE == 'LS':

        # Create figure and gridspec
        self.figure = plt.figure(figsize=(15,10), dpi=100)
        self.gs     = GridSpec(4, 2, width_ratios=[1, 1.5])

        # Create panels
        self.axes = []
        self.axes.append( plt.subplot( self.gs[:3,0] ) ) # Map         0
        self.axes.append( plt.subplot( self.gs[0,1]  ) ) # Spectrum    1
        self.axes.append( plt.subplot( self.gs[1,1]  ) ) # Percentiles 2
        self.axes.append( plt.subplot( self.gs[2,1]  ) ) # Percentiles 3
        self.axes.append( plt.subplot( self.gs[3,1]  ) ) # Percentiles 4

        # Create colorbars
        self.divider = make_axes_locatable(self.axes[0])
        self.cax     = self.divider.append_axes("right", size="5%", pad=0.1)
    
        # Activate picker and show plot
        self.axes[0].set_picker(True)
