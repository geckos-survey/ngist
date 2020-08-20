import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()
import matplotlib.ticker as ticker



# ============================================================================ #
#                               P L O T   M A P                                #
# ============================================================================ #
def plotMap(self, module, maptype):
   
    # Clear figure
    self.axes[0].cla()
    self.cax.cla()

    # Plot all spaxels, or only those in Voronoi-region
    if self.restrict2voronoi == 2: 
        idxMap = np.where( self.table.BIN_ID >= 0 )[0]
    else:
        idxMap = np.arange( len(self.table.BIN_ID) )

    self.currentMapQuantity = maptype

    # Voronoi table
    if module == 'TABLE':
        val = self.table[maptype][idxMap]
        if maptype == 'FLUX': val = np.log10(val)

    # Masking
    if module == 'MASK':
        val = self.Mask[maptype][idxMap]

    # stellarKinematics
    if module == 'KIN':
        val = self.kinResults[maptype][idxMap]

    # emissionLines
    if module == 'GAS':
        val = self.gasResults[maptype][idxMap]

        try:
            idx_AoNThreshold = np.where( self.gasResults[maptype.split('_')[0]+'_'+maptype.split('_')[1]+'_AON'][idxMap] < self.AoNThreshold )[0]
            val[idx_AoNThreshold] = np.nan
        except:
            print("WARNING: No AoN threshold is applied to the displayed map of "+maptype)

    # starFormationHistories
    if module == 'SFH':
        if maptype.split('_')[-1] == 'DIFF':
            val = self.kinResults[maptype.split('_')[0]][idxMap] - self.sfhResults[maptype.split('_')[0]][idxMap]
        else:
            val = self.sfhResults[maptype][idxMap]

    # lineStrengths
    if module == 'LS':
        val = self.lsResults[maptype][idxMap]



    # Create image in pixels
    xmin = np.nanmin(self.table.X[idxMap])-1;  xmax = np.nanmax(self.table.X[idxMap])+1
    ymin = np.nanmin(self.table.Y[idxMap])-1;  ymax = np.nanmax(self.table.Y[idxMap])+1
    npixels_x = int( np.round( (xmax - xmin)/self.pixelsize ) + 1 )
    npixels_y = int( np.round( (ymax - ymin)/self.pixelsize ) + 1 )
    i = np.array( np.round( (self.table.X[idxMap] - xmin)/self.pixelsize ), dtype=np.int )
    j = np.array( np.round( (self.table.Y[idxMap] - ymin)/self.pixelsize ), dtype=np.int )
    image = np.full( (npixels_x, npixels_y), np.nan )
    image[i,j] = val

    # Show map
    image = self.axes[0].imshow(np.rot90(image), cmap='sauron', interpolation='none', extent=[xmin-self.pixelsize/2, xmax+self.pixelsize/2, ymin-self.pixelsize/2, ymax+self.pixelsize/2])

    # Define colorbar
    cbar = plt.colorbar(image, cax=self.cax)
    cbar.solids.set_edgecolor("face")

    # Define special labels
    if   maptype == 'FLUX':    cbar.set_label("log( Flux )")
    elif maptype == 'V':       cbar.set_label("v [km/s]")
    elif maptype == 'SIGMA':   cbar.set_label("sigma [km/s]")
    elif maptype == 'AGE':     cbar.set_label("Age [Gyr]")
    elif maptype == 'METALS':  cbar.set_label("[M/H]") 
    elif maptype == 'ALPHA':   cbar.set_label("[Mg/Fe]") 
    else:                      cbar.set_label(maptype)

    self.axes[0].set_title(maptype)
    self.axes[0].set_xlabel('x [arcsec]')
    self.axes[0].set_ylabel('y [arcsec]')

    self.canvas.draw()



# ============================================================================ #
#                              P L O T   D A T A                               #
# ============================================================================ #
def plotData(self):

    ## Print pixel coordinates of click and identified bin
    #print("X, Y = ", self.table.X[self.idxBinLong], self.table.Y[self.idxBinLong])
    #print("BIN_ID = "+str(int(self.idxBinShort)) )
    #print("")



    # Mark selected bin
    try: 
        self.binMarker.remove()
        self.spaxelMarker.remove()
    except: 
        pass
    self.binMarker    = self.axes[0].scatter( self.table.XBIN[self.idxBinLong], self.table.YBIN[self.idxBinLong], color=self.markercolor, marker='x' )
    self.spaxelMarker = self.axes[0].scatter( self.table.X[self.idxBinLong],    self.table.Y[self.idxBinLong],    color=self.markercolor, marker='o' )


    
    # No analysis has been done: Plot plain spectrum without fit
    if self.KIN == False  and  self.GAS == False  and  self.SFH == False: 
        self.plotPlainSpectrum(self.Spectra[self.idxBinShort], self.table.SNRBIN[self.idxBinLong], 1)


    # Plot stellarKinematics fit
    if self.KIN == True: 
        self.plotSpectraKIN(self.Spectra[self.idxBinShort], self.kinBestfit[self.idxBinShort], self.kinGoodpix, 1)
        self.axes[1].set_title("Stellar kinematics: v={:.1f}km/s, sigma={:.1f}km/s, h3={:.2f}, h4={:.2f}".format(self.kinResults.V[self.idxBinLong], self.kinResults.SIGMA[self.idxBinLong], self.kinResults.H3[self.idxBinLong], self.kinResults.H4[self.idxBinLong]), loc='left')
        self.axes[1].set_title("BIN_ID = {:d}".format(self.idxBinShort), loc='right')


    # Plot emissionLines fit
    if self.GAS == True: 
        if self.gasLevel == 'BIN':
            self.plotSpectraGAS(self.Spectra[self.idxBinShort], self.gasBestfit[self.idxBinShort], self.gasGoodpix, 2)
        elif self.gasLevel == 'SPAXEL':
            self.plotSpectraGAS(self.AllSpectra[self.idxBinLong], self.gasBestfit[self.idxBinLong], self.gasGoodpix, 2)
            self.axes[2].set_title("SPAXEL_ID = {:d}".format(self.idxBinLong), loc='right')
        try: 
            line = self.currentMapQuantity.split('_')[0] + '_' + self.currentMapQuantity.split('_')[1]
            self.axes[2].set_title("Emission-line kinematics: v={:.1f}km/s, sigma={:.1f}km/s".format(self.gasResults[line+'_V'][self.idxBinLong], self.gasResults[line+'_S'][self.idxBinLong]), loc='left')
        except: 
            self.axes[2].set_title('Emission-line analysis', loc='left')


    # Plot starFormationHistories results
    if self.SFH == True:
        self.plotSpectraSFH(self.Spectra[self.idxBinShort], self.sfhBestfit[self.idxBinShort], self.sfhGoodpix, 3)
        self.axes[3].set_title("Stellar populations: Age={:.2f} Gyr, [M/H]={:.2f}, [alpha/Fe]={:.2f}".format(self.sfhResults['AGE'][self.idxBinLong], self.sfhResults['METAL'][self.idxBinLong], self.sfhResults['ALPHA'][self.idxBinLong]), loc='left')
        self.axes[3].set_xlabel("$\lambda\ [\AA]$")

        self.axes[4].cla()
        self.cax4.cla()
        self.axes[6].cla()

        # Plot age-metallicity-alpha cube
        pcol3 = self.axes[4].pcolormesh(self.age, self.metals, self.Weights[self.idxBinShort,:,:,0], edgecolors='face')
        self.cbar3 = plt.colorbar(pcol3, cax=self.cax4)
        self.cbar3.solids.set_edgecolor("face")
        self.cbar3.ax.tick_params(labelsize=8) 

        self.axes[4].set_title("Mass Fraction - [alpha/Fe]=0", loc='left')
        self.axes[4].set_ylabel("[M/H]")
        self.axes[4].set_xlabel("Age [Gyr]")

        if self.Weights.shape[3] == 2: 
            # Plot age-metallicity-alpha cube
            pcol4 = self.axes[6].pcolormesh(self.age, self.metals, self.Weights[self.idxBinShort,:,:,1], edgecolors='face')
 
            self.axes[6].set_title("Mass Fraction - [alpha/Fe]=0.40", loc='left')
            self.axes[6].set_xlabel("Age [Gyr]")
       
        # Plot star-formation history
        self.plotSFH(5)


    # Draw figures
    self.canvas.draw()



def plotPlainSpectrum(self, spectra, snr, panel):
    # Clear panels
    self.axes[panel].cla()

    # Plot spectra
    self.axes[panel].plot(self.Lambda, spectra[:], color='k', linewidth=2)
    self.axes[panel].set_xlim([self.Lambda[0], self.Lambda[-1]])
    self.axes[panel].set_ylabel('Flux')

    self.axes[panel].set_title("SNRBIN = {:.1f}".format(snr), loc='left')
    self.axes[panel].set_title("BIN_ID = {:d}".format(self.idxBinShort), loc='right')


def plotSpectraKIN(self, spectra, bestfit, goodpix, panel):

    # Compile information on masked regions 
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)

    # Clear panels
    self.axes[panel].cla()

    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.kinLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.kinLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam],               color='k',         linewidth=2)
    self.axes[panel].plot(self.kinLambda, bestfit[:],                         color='crimson',   linewidth=2)
    self.axes[panel].plot(self.kinLambda, spectra[idxLam] - bestfit + offset, color='limegreen', linewidth=2)

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    self.axes[panel].plot( [self.Lambda[idxLam][0], self.Lambda[idxLam][-1]], [offset,offset], color='k', linewidth=0.5 )
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            self.axes[panel].axvspan(self.Lambda[idxLam][vlines[i]], self.Lambda[idxLam][vlines[i+1]], color='k', alpha=0.1, lw=0)

    self.axes[panel].set_xlim([self.Lambda[idxLam][0], self.Lambda[idxLam][-1]])
    self.axes[panel].set_ylabel('Flux')

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format( np.exp(x) ))
    self.axes[panel].xaxis.set_major_formatter(ticks)


def plotSpectraGAS(self, spectra, bestfit, goodpix, panel):

    # Compile information on masked regions 
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)

    # Clear panels
    self.axes[panel].cla()

    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.gasLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.gasLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    if self.gasLevel == 'BIN': 
        self.axes[panel].plot(self.gasLambda, self.EmissionSubtractedSpectraBIN[self.idxBinShort,:], color='orange', linewidth=2)
    elif self.gasLevel == 'SPAXEL': 
        self.axes[panel].plot(self.gasLambda, self.EmissionSubtractedSpectraSPAXEL[self.idxBinLong,:], color='orange', linewidth=2)

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam],               color='k',         linewidth=2)
    self.axes[panel].plot(self.gasLambda, bestfit[:],                         color='crimson',   linewidth=2)
    self.axes[panel].plot(self.gasLambda, spectra[idxLam] - bestfit + offset, color='limegreen', linewidth=2)

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    self.axes[panel].plot( [self.Lambda[idxLam][0], self.Lambda[idxLam][-1]], [offset,offset], color='k', linewidth=0.5 )
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            self.axes[panel].axvspan(self.Lambda[idxLam][vlines[i]], self.Lambda[idxLam][vlines[i+1]], color='k', alpha=0.1, lw=0)

    self.axes[panel].set_xlim([self.Lambda[idxLam][0], self.Lambda[idxLam][-1]])
    self.axes[panel].set_ylabel('Flux')

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format( np.exp(x) ))
    self.axes[panel].xaxis.set_major_formatter(ticks)


def plotSpectraSFH(self, spectra, bestfit, goodpix, panel):

    # Compile information on masked regions 
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)

    # Clear panels
    self.axes[panel].cla()

    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.sfhLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.sfhLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    try:
        idxMin     = np.where( self.gasLambda == self.sfhLambda[0]  )[0]
        idxMax     = np.where( self.gasLambda == self.sfhLambda[-1] )[0]
        idxLamGand = np.arange(idxMin, idxMax+1)
        self.axes[panel].plot(self.gasLambda[idxLamGand], self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand],                    color='orange',    linewidth=2)
        self.axes[panel].plot(self.gasLambda[idxLamGand], self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand] - bestfit + offset, color='limegreen', linewidth=2)
    except:
        pass

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam], color='k',       linewidth=2)
    self.axes[panel].plot(self.sfhLambda,      bestfit[:],      color='crimson', linewidth=2)

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    self.axes[panel].plot( [self.Lambda[idxLam][0], self.Lambda[idxLam][-1]], [offset,offset], color='k', linewidth=0.5 )
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            self.axes[panel].axvspan(self.Lambda[idxLam][vlines[i]], self.Lambda[idxLam][vlines[i+1]], color='k', alpha=0.1, lw=0)

    self.axes[panel].set_xlim([self.Lambda[idxLam][0], self.Lambda[idxLam][-1]])
    self.axes[panel].set_ylabel('Flux')

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format( np.exp(x) ))
    self.axes[panel].xaxis.set_major_formatter(ticks)


def plotSFH(self, panel):

    # Clear panels
    self.axes[panel].cla()

    # Get star formation history
    collapsed = np.sum( self.Weights, axis=(1,3) )

    # Plot it all
    self.axes[panel].plot(self.age, collapsed[self.idxBinShort, :])
    self.axes[panel].set_xlim([self.age[0], self.age[-1]])
    self.axes[panel].set_ylim(bottom=0)
    self.axes[panel].set_title("Star Formation History; Mean Age: {:.2f}".format(self.sfhResults['AGE'][self.idxBinLong])+" Gyr")
    self.axes[panel].set_xlabel("Age [Gyr]")
    self.axes[panel].set_ylabel("#")



