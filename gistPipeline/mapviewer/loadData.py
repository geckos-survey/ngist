from   astropy.io import fits
import numpy      as np
import os



def loadData(self):
    """ Load all available data to make it accessible for plotting. """

    # Clean all figures
    for i in range( len(self.axes) ):
        self.axes[i].cla()
    self.cax.cla()
    try: 
        self.cax3.cla()
        self.cax4.cla()
    except: pass
    self.canvas.draw()

    if os.path.isfile(self.dirprefix+'_table.fits') == False:
        self.dialogNotAvailable("directory")
        return(None)

    # ==============================
    # Determine what to do
    # ==============================
    # Firstly, check what is available
    self.PPXF          = os.path.isfile(self.dirprefix+'_ppxf.fits')
    if os.path.isfile(self.dirprefix+'_gandalf_BIN.fits') == False  and  os.path.isfile(self.dirprefix+'_gandalf_SPAXEL.fits') == False:
        self.GANDALF = False
        self.GandalfLevelAvailable = []
    if os.path.isfile(self.dirprefix+'_gandalf_BIN.fits') == True   and  os.path.isfile(self.dirprefix+'_gandalf_SPAXEL.fits') == False:
        self.GANDALF = True
        self.GandalfLevelAvailable = ['BIN']
    if os.path.isfile(self.dirprefix+'_gandalf_BIN.fits') == False  and  os.path.isfile(self.dirprefix+'_gandalf_SPAXEL.fits') == True:
        self.GANDALF = True
        self.GandalfLevelAvailable = ['SPAXEL']
    if os.path.isfile(self.dirprefix+'_gandalf_BIN.fits') == True   and  os.path.isfile(self.dirprefix+'_gandalf_SPAXEL.fits') == True:
        self.GANDALF = True
        self.GandalfLevelAvailable = ['BIN', 'SPAXEL']

    if len( self.GandalfLevelAvailable ) == 1:
        self.GandalfLevel = self.GandalfLevelAvailable[0]
    else: 
        self.GandalfLevel = self.GandalfLevelSelected

    self.SFH           = os.path.isfile(self.dirprefix+'_sfh.fits')

    if os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == False:
        self.LINE_STRENGTH = False
        self.LsLevelAvailable = []
    if os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == False:
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ['ORIGINAL']
    if os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == True:
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ['ADAPTED']
    if os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == True:
        self.LINE_STRENGTH = True
        self.LsLevelAvailable = ['ORIGINAL', 'ADAPTED']

    if len( self.LsLevelAvailable ) == 1: 
        self.LsLevel = self.LsLevelAvailable[0]
    else: 
        self.LsLevel = self.LsLevelSelected


    # ==============================
    # Read Bintable
    # ==============================
    self.table     = fits.open(self.dirprefix+'_table.fits')[1].data
    try:
        self.pixelsize = fits.open(self.dirprefix+'_table.fits')[0].header['PIXSIZE']
    except: 
        print( "Keyword 'PIXSIZE' not found in "+self.dirprefix.split('/')[-1]+"_table.fits. Pixelsize is set to 0.20 by default. " )
        self.pixelsize = 0.20
    _, idxConvertShortToLong = np.unique( np.abs(self.table.BIN_ID), return_inverse=True )  # Get transformation array


    # ==============================
    # Read the spectra
    # ==============================
    hdu             = fits.open(self.dirprefix+'_VorSpectra.fits')
    self.Spectra    = hdu[1].data.SPEC
    nbins           = self.Spectra.shape[0]
    npix            = self.Spectra.shape[1]
    self.Lambda     = np.array(hdu[2].data.LOGLAM)
    if self.GandalfLevel == 'SPAXEL':
        self.AllSpectra = fits.open(self.dirprefix+'_AllSpectra.fits')[1].data.SPEC


    # ==============================
    # Read PPXF-results
    # ==============================
    if self.PPXF == True:
        ppxfHDU           = fits.open(self.dirprefix+'_ppxf.fits')[1].data
        ppxf              = np.array( [ppxfHDU.V, ppxfHDU.SIGMA, ppxfHDU.H3, ppxfHDU.H4, ppxfHDU.LAMBDA_R] ).T

        self.ppxf_results = ppxf[idxConvertShortToLong,:]
        self.ppxfBestfit  = fits.open(self.dirprefix+'_ppxf-bestfit.fits')[1].data.BESTFIT
        self.ppxfGoodpix  = fits.open(self.dirprefix+'_ppxf-goodpix.fits')[1].data.GOODPIX

        median_V_stellar       = np.nanmedian( self.ppxf_results[:,0] )
        self.ppxf_results[:,0] = self.ppxf_results[:,0] - median_V_stellar


    # ==============================
    # Read GANDALF-results
    # ==============================
    if os.path.isfile(self.dirprefix+"_gandalf-cleaned_BIN.fits") == True:
        self.EmissionSubtractedSpectraBIN    = np.array( fits.open(self.dirprefix+"_gandalf-cleaned_BIN.fits")[1].data.SPEC )
    if os.path.isfile(self.dirprefix+"_gandalf-cleaned_SPAXEL.fits") == True:
        self.EmissionSubtractedSpectraSPAXEL = np.array( fits.open(self.dirprefix+"_gandalf-cleaned_SPAXEL.fits")[1].data.SPEC )
    if self.GANDALF == True:
        self.gandalf_setup   = fits.open(self.dirprefix+'_gandalf_'+self.GandalfLevel+'.fits')[1].data
        gandalf              = fits.open(self.dirprefix+'_gandalf_'+self.GandalfLevel+'.fits')[2].data
        self.gandalfBestfit  = fits.open(self.dirprefix+'_gandalf-bestfit_'+self.GandalfLevel+'.fits')[1].data.BESTFIT
        self.gandalfGoodpix  = fits.open(self.dirprefix+'_gandalf-goodpix_'+self.GandalfLevel+'.fits')[1].data.GOODPIX

        liste = []
        for itm in self.gandalf_setup.name:
            liste.append( itm.decode('utf-8') )
        self.listOfLineNames = np.array( liste )

        gandalf = np.array([gandalf.V, gandalf.SIGMA, gandalf.FLUX, gandalf.AMPL, gandalf.AON])
        gandalf = np.transpose( gandalf, (1,0,2) )

        if self.GandalfLevel == 'BIN':
            self.gandalf_results = gandalf[idxConvertShortToLong,:,:]
        if self.GandalfLevel == 'SPAXEL':
            self.gandalf_results = gandalf[:,:,:]

        self.gandalf_results[:,0,:] = self.gandalf_results[:,0,:] - median_V_stellar


    # ==============================
    # Read SFH-results
    # ==============================
    if self.SFH == True:
        sfhHDU                = fits.open(self.dirprefix+'_sfh.fits')[1].data
        try:
            sfh                   = np.array( [sfhHDU.AGE, sfhHDU.METAL, sfhHDU.ALPHA, sfhHDU.V, sfhHDU.SIGMA, sfhHDU.H3, sfhHDU.H4 ] ).T
        except:
            print("WARNING: YOU SEEM TO LOAD OUTDATED RESULTS FROM THE SFH MODULE. I HOPE YOU KNOW WHAT YOU ARE DOING ...")
            sfh                   = np.array( [10**sfhHDU.LOGAGE, sfhHDU.METAL, sfhHDU.ALPHA, sfhHDU.V, sfhHDU.SIGMA, sfhHDU.H3, sfhHDU.H4 ] ).T
        self.sfh_results      = sfh[idxConvertShortToLong,:]
        self.sfh_results[:,3] = self.sfh_results[:,3] - median_V_stellar

        self.sfhBestfit  = fits.open(self.dirprefix+'_sfh-bestfit.fits')[1].data.BESTFIT
        self.sfhGoodpix  = fits.open(self.dirprefix+'_sfh-goodpix.fits')[1].data.GOODPIX
    
        # Read the age, metallicity and [Mg/Fe] grid
        grid        = fits.open(self.dirprefix+'_sfh-weights.fits')[2].data
        self.metals = np.unique(grid.METAL)
        self.age    = np.power( 10, np.unique(grid.LOGAGE) )

        # Read weights
        hdu_weights  = fits.open(self.dirprefix+'_sfh-weights.fits')
        nAges        = hdu_weights[0].header['NAGES']
        nMetal       = hdu_weights[0].header['NMETAL']
        nAlpha       = hdu_weights[0].header['NALPHA']
        self.Weights = np.reshape( np.array(hdu_weights[1].data.WEIGHTS), (nbins,nAges,nMetal,nAlpha) )
        self.Weights = np.transpose(self.Weights, (0,2,1,3))


    # ==============================
    # Read LS-results
    # ==============================
    if self.LINE_STRENGTH == True:

        if self.LsLevel == "ORIGINAL": 
            ls          = fits.open(self.dirprefix+'_ls_OrigRes.fits')[1].data
        elif self.LsLevel == "ADAPTED":
            ls          = fits.open(self.dirprefix+'_ls_AdapRes.fits')[1].data
            if len( fits.open(self.dirprefix+'_ls_AdapRes.fits') ) == 3: 
                ls_spp = fits.open(self.dirprefix+'_ls_AdapRes.fits')[2].data

        ls_list = []
        for i in range(0, len(ls.columns) ):
            if ls.columns[i].name.find('ERR_') == -1:
                ls_list.append( ls.columns[i].name )
        ls_list.append( 'AGE'   )
        ls_list.append( 'METAL' )
        ls_list.append( 'ALPHA' )

        line_strength = np.zeros( (len(ls), len(ls_list)) )
        line_strength[:,:] = np.nan
        for i in range( len(ls_list)-3 ): 
            line_strength[:,i] = ls[ls_list[i]]
        if self.LsLevel == "ADAPTED"  and  len( fits.open(self.dirprefix+'_ls_AdapRes.fits') ) == 3: 
            for i in range( len(ls_list)-3, len(ls_list) ):
                line_strength[:,i] = ls_spp[ls_list[i]][:,50]

        self.line_strength = line_strength[idxConvertShortToLong,:]
        self.lsList = np.array( ls_list )

        if self.LsLevel == "ADAPTED"  and  len( fits.open(self.dirprefix+'_ls_AdapRes.fits') ) == 3: 
            self.LSPercentiles = np.zeros((nbins, 101, 3))
            self.LSPercentiles[:,:,0] = ls_spp.AGE
            self.LSPercentiles[:,:,1] = ls_spp.METAL
            self.LSPercentiles[:,:,2] = ls_spp.ALPHA
