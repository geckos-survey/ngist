from astropy.io import fits, ascii
import numpy    as np

import matplotlib.pyplot     as plt
from matplotlib              import rcParams
rcParams.update({'figure.autolayout': True})
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()
import matplotlib.ticker as ticker



# ==============================================================================
#                               P L O T   M A P
# ==============================================================================
def plotMap(self, maptype):
   
    # Clear figure
    self.axes[0].cla()
    self.cax.cla()

    # Check if data is actually available
    if self.PPXF == False  and  maptype in ['V','SIGMA','H3','H4']:
        self.dialogNotAvailable("PPXF")
        return()
    if self.GANDALF == False  and  len(maptype.split('_')) == 3: 
        self.dialogNotAvailable("GANDALF")
        return()
    if self.SFH == False  and  maptype in ['AGE', 'METALS', 'ALPHA', 'V_SFH', 'SIGMA_SFH', 'H3_SFH', 'H4_SFH', 'V_Dif', 'SIGMA_Dif', 'H3_Dif', 'H4_Dif']:
        self.dialogNotAvailable("SFH")
        return()
    if self.LINE_STRENGTH == False  and  maptype.split('_')[0] == 'LS':
        self.dialogNotAvailable("Line strength")
        return()

    # Plot all spaxels, or only those in Voronoi-region
    if self.restrict2voronoi == 2: 
        idxMap = np.where( self.table.BIN_ID >= 0 )[0]
    else:
        idxMap = np.arange( len(self.table.BIN_ID) )

    # Voronoi table
    if maptype == 'Table_BINID':   val = self.table.BIN_ID[idxMap]
    if maptype == 'Table_FLUX':    val = np.log10(self.table.FLUX[idxMap])
    if maptype == 'Table_SNR':     val = self.table.SNR[idxMap]
    if maptype == 'Table_SNRBIN':  val = self.table.SNRBIN[idxMap]
    if maptype == 'Table_NSPAX':   val = self.table.NSPAX[idxMap]

    # PPXF
    if maptype == 'PPXF_V':       val = self.ppxf_results[idxMap,0]
    if maptype == 'PPXF_SIGMA':   val = self.ppxf_results[idxMap,1]
    if maptype == 'PPXF_H3':      val = self.ppxf_results[idxMap,2]
    if maptype == 'PPXF_H4':      val = self.ppxf_results[idxMap,3]
    if maptype == 'PPXF_LAMBDAR': val = self.ppxf_results[idxMap,4]

    # GANDALF
    if self.GANDALF == True  and  len(maptype.split('_')) == 3:

        # Find line
        lineIdentifier   = maptype.split('_')[1]
        lambdaIdentifier = maptype.split('_')[2]
        self.line_idx = np.where( np.logical_and( self.listOfLineNames == lineIdentifier, self.gandalf_setup._lambda == float(lambdaIdentifier) ) )[0]
        if self.line_idx.shape[0] == 0:
            val = np.zeros((self.gandalf_results.shape[0])); val[:] = np.nan
        else:
            # Get data; data is copied to apply AoN threshold
            if   maptype.split('_')[0] == 'V':
                val = np.copy( np.ravel( self.gandalf_results[idxMap,0,self.line_idx] ) )
            elif maptype.split('_')[0] == 'S':
                val = np.copy( np.ravel( self.gandalf_results[idxMap,1,self.line_idx] ) )
            elif maptype.split('_')[0] == 'A':
                val = np.copy( np.ravel( self.gandalf_results[idxMap,3,self.line_idx] ) )
            elif maptype.split('_')[0] == 'F':
                val = np.copy( np.log10( np.ravel( self.gandalf_results[idxMap,2,self.line_idx] ) ) )

            # Apply AoN Theshold
            idx_AoNThreshold = np.where( self.gandalf_results[idxMap,4,self.line_idx] < self.AoNThreshold )[0]
            val[idx_AoNThreshold] = np.nan

    # SFH
    if maptype == 'AGE':       val = self.sfh_results[idxMap,0]
    if maptype == 'METALS':    val = self.sfh_results[idxMap,1]
    if maptype == 'ALPHA':     val = self.sfh_results[idxMap,2]
    #
    if maptype == 'V_SFH':     val = self.sfh_results[idxMap,3]
    if maptype == 'SIGMA_SFH': val = self.sfh_results[idxMap,4]
    if maptype == 'H3_SFH':    val = self.sfh_results[idxMap,5]
    if maptype == 'H4_SFH':    val = self.sfh_results[idxMap,6]
    #
    if maptype == 'V_Dif':     val = self.ppxf_results[idxMap,0] - self.sfh_results[idxMap,3]
    if maptype == 'SIGMA_Dif': val = self.ppxf_results[idxMap,1] - self.sfh_results[idxMap,4]
    if maptype == 'H3_Dif':    val = self.ppxf_results[idxMap,2] - self.sfh_results[idxMap,5]
    if maptype == 'H4_Dif':    val = self.ppxf_results[idxMap,3] - self.sfh_results[idxMap,6]

    # Line strength indices
    try: 
        if maptype.split('_')[0] == 'LS':
            maptype = maptype.split('_')[-1]
            idx = np.where( self.lsList == maptype )[0]
            val = self.line_strength[idxMap,idx].reshape( len(idxMap) )
    except: 
        self.dialogNotAvailable(maptype)
        return(None)

    # Handle exception and save maptype
    try: 
        _ = val.shape
        self.maptype = maptype
    except: 
        return(None)

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


# ==============================================================================
#                              P L O T   D A T A
# ==============================================================================
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


    if self.MODE == 'ALL':

        if self.PPXF == False  and  self.GANDALF == False  and  self.SFH == False: 
            self.plotPlainSpectrum( self.Spectra[self.idxBinShort], self.table.SNRBIN[self.idxBinLong], 1 )

        if self.PPXF          == True: 
            self.plotSpectraPPXF( self.Spectra[self.idxBinShort], self.ppxfBestfit[self.idxBinShort], self.ppxfGoodpix, 1 )
            self.axes[1].set_title("pPXF: v={:.1f}km/s, sigma={:.1f}km/s, h3={:.2f}, h4={:.2f}".format( self.ppxf_results[self.idxBinLong,0], self.ppxf_results[self.idxBinLong,1], self.ppxf_results[self.idxBinLong,2], self.ppxf_results[self.idxBinLong,3]), loc='left')
            self.axes[1].set_title("BIN_ID = {:d}".format( self.idxBinShort), loc='right')

        if self.GANDALF       == True: 
            if self.GandalfLevel == 'BIN':
                self.plotSpectraGANDALF( self.Spectra[self.idxBinShort], self.gandalfBestfit[self.idxBinShort], self.gandalfGoodpix, 2 )
            elif self.GandalfLevel == 'SPAXEL':
                self.plotSpectraGANDALF( self.AllSpectra[self.idxBinLong], self.gandalfBestfit[self.idxBinLong], self.gandalfGoodpix, 2 )
                self.axes[2].set_title("SPAXEL_ID = {:d}".format( self.idxBinLong), loc='right')
            try: 
                self.axes[2].set_title("GandALF: v={:.1f}km/s, sigma={:.1f}km/s".format(self.gandalf_results[self.idxBinLong,0,self.line_idx][0], self.gandalf_results[self.idxBinLong,1,self.line_idx][0]), loc='left')
            except: 
                self.axes[2].set_title( 'GandALF', loc='left' )


        if self.SFH == True:
            self.plotSpectraSFH( self.Spectra[self.idxBinShort], self.sfhBestfit[self.idxBinShort], self.sfhGoodpix, 3 )
            self.axes[3].set_title("SFH: Age={:.2f} Gyr, [M/H]={:.2f}, [alpha/Fe]={:.2f}".format( self.sfh_results[self.idxBinLong,0], self.sfh_results[self.idxBinLong,1], self.sfh_results[self.idxBinLong,2] ), loc='left')
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
            self.plotSFH( 5 )


    elif self.MODE == 'LS':

        if self.LINE_STRENGTH == True: 

            labels  = ['AGE','METAL','ALPHA']
            nwalker = int( fits.open(self.directory+'USED_PARAMS.fits')[0].header['HIERARCH NWALKER'] )
            nchain  = int( fits.open(self.directory+'USED_PARAMS.fits')[0].header['HIERARCH NCHAIN']  )

            # Plot Spectrum
            self.plotLSSpectrum( 1 )

            # Plot Percentiles
            if self.LsLevel == "ADAPTED":
                self.plotPercentiles( 2, labels )


    # Draw figures
    self.canvas.draw()


def plotPercentiles(self, panel, labels):

    for param in range(len(labels)):
        self.axes[param+panel].cla()

        self.axes[param+panel].plot( np.arange(90)+5, self.LSPercentiles[self.idxBinShort,5:95,param], color='k', linewidth=2 )
        self.axes[param+panel].set_ylabel(labels[param])
        self.axes[param+panel].set_xlim([0,100])
        self.axes[param+panel].set_title(labels[param]+': {:.2f}'.format( self.LSPercentiles[self.idxBinShort,50,param] ), loc='left' )

        self.axes[param+panel].axhline( self.LSPercentiles[self.idxBinShort,50,param], color='b', linewidth=1 )
        self.axes[param+panel].axvline( 50, color='b', linewidth=1 )

    self.axes[panel+2].set_xlabel('Percentiles')


def plotLSSpectrum(self, panel):

    self.axes[panel].cla()

    idxMin = np.where( self.Lambda == self.gandalfLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.gandalfLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    self.axes[panel].plot(self.gandalfLambda, self.EmissionSubtractedSpectraBIN[self.idxBinShort,:], color='orange', linewidth=2)
    self.axes[panel].plot(self.gandalfLambda, self.Spectra[self.idxBinShort,idxLam], color='k', linewidth=2)

    self.axes[panel].set_title("pPXF: v={:.1f}km/s, sigma={:.1f}km/s, h3={:.2f}, h4={:.2f}".format( self.ppxf_results[self.idxBinLong,0], self.ppxf_results[self.idxBinLong,1], self.ppxf_results[self.idxBinLong,2], self.ppxf_results[self.idxBinLong,3]), loc='left')
    self.axes[panel].set_title("BIN_ID = {:d}".format( self.idxBinShort), loc='right')
    self.axes[panel].set_xlim([self.Lambda[0], self.Lambda[-1]])
    self.axes[panel].set_xlabel("$\lambda\ [\AA]$")
    self.axes[panel].set_ylabel('Flux')

    tab      = ascii.read(self.directory+'lsBands.config', comment='\s*#')
    idx_band = np.where( np.array(tab['names']) == self.maptype )[0]
    if len(idx_band) == 1:
        self.axes[panel].axvspan( np.log(tab['b1'][idx_band]), np.log(tab['b2'][idx_band]), color='k', alpha=0.05, lw=0)
        self.axes[panel].axvspan( np.log(tab['b3'][idx_band]), np.log(tab['b4'][idx_band]), color='k', alpha=0.10, lw=0)
        self.axes[panel].axvspan( np.log(tab['b5'][idx_band]), np.log(tab['b6'][idx_band]), color='k', alpha=0.05, lw=0)

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format( np.exp(x) ))
    self.axes[panel].xaxis.set_major_formatter(ticks)


def plotSFH(self, panel):

    # Clear panels
    self.axes[panel].cla()

    # Get star formation history
    collapsed = np.sum( self.Weights, axis=(1,3) )

    # Plot it all
    self.axes[panel].plot( self.age, collapsed[self.idxBinShort, :] )
    self.axes[panel].set_xlim([self.age[0], self.age[-1]])
    self.axes[panel].set_ylim(bottom=0)
    self.axes[panel].set_title("Star Formation History; Mean Age: {:.2f}".format(self.sfh_results[self.idxBinLong,0])+" Gyr")
    self.axes[panel].set_xlabel("Age [Gyr]")
    self.axes[panel].set_ylabel("#")


def plotSpectraPPXF(self, spectra, bestfit, goodpix, panel):

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
    offset = np.min( bestfit[:] ) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.ppxfLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.ppxfLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam],                    color='k',         linewidth=2)
    self.axes[panel].plot(self.ppxfLambda,     bestfit[:],                         color='crimson',   linewidth=2)
    self.axes[panel].plot(self.ppxfLambda,     spectra[idxLam] - bestfit + offset, color='limegreen', linewidth=2)

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


def plotSpectraGANDALF(self, spectra, bestfit, goodpix, panel):

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
    offset = np.min( bestfit[:] ) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.gandalfLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.gandalfLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    if self.GandalfLevel == 'BIN': 
        self.axes[panel].plot(self.gandalfLambda, self.EmissionSubtractedSpectraBIN[self.idxBinShort,:],   color='orange', linewidth=2)
    elif self.GandalfLevel == 'SPAXEL': 
        self.axes[panel].plot(self.gandalfLambda, self.EmissionSubtractedSpectraSPAXEL[self.idxBinLong,:], color='orange', linewidth=2)

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam],                    color='k',         linewidth=2)
    self.axes[panel].plot(self.gandalfLambda,  bestfit[:],                         color='crimson',   linewidth=2)
    self.axes[panel].plot(self.gandalfLambda,  spectra[idxLam] - bestfit + offset, color='limegreen', linewidth=2)

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
    offset = np.min( bestfit[:] ) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.sfhLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.sfhLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    idxMin     = np.where( self.gandalfLambda == self.sfhLambda[0]  )[0]
    idxMax     = np.where( self.gandalfLambda == self.sfhLambda[-1] )[0]
    idxLamGand = np.arange(idxMin, idxMax+1)


    self.axes[panel].plot(self.gandalfLambda[idxLamGand], self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand],                    color='orange',    linewidth=2)
    self.axes[panel].plot(self.gandalfLambda[idxLamGand], self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand] - bestfit + offset, color='limegreen', linewidth=2)

    self.axes[panel].plot(self.Lambda[idxLam], spectra[idxLam],                    color='k',         linewidth=2)
    self.axes[panel].plot(self.sfhLambda,      bestfit[:],                         color='crimson',   linewidth=2)

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


def plotPlainSpectrum(self, spectra, snr, panel):

    # Clear panels
    self.axes[panel].cla()

    # Plot spectra
    self.axes[panel].plot(self.Lambda, spectra[:], color='k',       linewidth=2)
    self.axes[panel].set_xlim([self.Lambda[0], self.Lambda[-1]])
    self.axes[panel].set_ylabel('Flux')

    self.axes[panel].set_title("SNRBIN = {:.1f}".format(snr), loc='left')
    self.axes[panel].set_title("BIN_ID = {:d}".format( self.idxBinShort), loc='right')


