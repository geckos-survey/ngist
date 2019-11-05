import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import time
import os
import glob
import logging

from gistPipeline.gistModules import util                as pipeline
from gistPipeline.gistModules import gistPrepare         as util_prepare
from gistPipeline.gistModules import gistPlot_spp        as util_plot
from gistPipeline.gistModules import gistPlot_kinematics as util_plot_kin

try:
    # Try to use local version in sitePackages
    from gistPipeline.sitePackages.ppxf.ppxf      import ppxf
    from gistPipeline.sitePackages.ppxf.ppxf_util import log_rebin, gaussian_filter1d
except:
    # Then use system installed version instead
    from ppxf.ppxf      import ppxf
    from ppxf.ppxf_util import log_rebin, gaussian_filter1d

# Physical constants
C = 299792.458 # speed of light in km/s


"""
PURPOSE: 
  This module performs the extraction of non-parametric star-formation histories
  by full-spectral fitting.  Basically, it acts as an interface between pipeline
  and the pPXF routine from Cappellari & Emsellem 2004
  (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C). 
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, galaxy, noise, velscale, start, goodPixels_sfh, mom, dv,\
        adeg, mdeg, regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i\
        in iter(inQueue.get, 'STOP'): 

        sol, w_row, bestfit, formal_error = run_ppxf(templates, galaxy, noise, velscale, start, goodPixels_sfh, mom, \
                                            dv, adeg, mdeg, regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i) 

        outQueue.put(( i, sol, w_row, bestfit, formal_error ))


def run_ppxf(templates, galaxy_i, noise_i, velscale, start, goodPixels, nmom, dv, adeg, mdeg,\
             regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004 
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C; 
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories. 
    """ 
    pipeline.printProgress(i, nbins, barLength = 50)

    try:

#        noise_i = noise_i * np.sqrt(  / len(goodPixels) )
#        regul_err = 

        pp = ppxf(templates, galaxy_i, noise_i, velscale, start, goodpixels=goodPixels, plot=False, quiet=True,\
              moments=nmom, degree=-1, vsyst=dv, mdegree=mdeg, regul=1./regul_err, fixed=fixed, velscale_ratio=velscale_ratio)
    
#        if i == 0: 
#            print()
#            print( i, pp.chi2 )
#            print( len( goodPixels ) )
#            print( np.sqrt(2 * len(goodPixels)) )
#            print()
     
        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
        w_row   = np.reshape(weights, ncomb) 
    
        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)
    
        return(pp.sol, w_row, pp.bestfit, formal_error)

    except:
        return(np.nan, np.nan, np.nan, np.nan)


def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin. 
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan

    for i in range( nbins ):
        mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
        mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])
    
    return(mean)


def setup_spectral_library(configs, velscale, LSF_Data, LSF_Templates):
    """
    Prepares the spectral template library for the computation of star-formation
    histories. The templates are loaded from disk, shortened to meet the
    spectral range in consideration, convolved to meet the resolution of the
    observed spectra (according to the LSF), log-rebinned, and normalised. In
    addition, they are sorted in a three-dimensional array sampling the
    parameter space in age, metallicity and alpha-enhancement. 
    """
    pipeline.prettyOutput_Running("Preparing the stellar population templates")
    cvel  = 299792.458

    # List of models
    ssp_lib = glob.glob(configs['SSP_LIB']+'*.fits')
    ssp_lib.sort()

    # Extract ages, metallicities and alpha from the templates
    logAge, metal, alpha, metal_str, alpha_str, nAges, nMetal, nAlpha, ncomb = age_metal_alpha(ssp_lib)

    # Read data
    hdu           = fits.open(ssp_lib[0])
    ssp_data      = hdu[0].data
    ssp_head      = hdu[0].header
    lamRange_temp = ssp_head['CRVAL1'] + np.array([0., ssp_head['CDELT1']*(ssp_head['NAXIS1']-1)])

    # Determine length of templates
    template_overhead = np.zeros(2)
    if configs['LMIN'] - lamRange_temp[0] > 150.:
        template_overhead[0] = 150.
    else: 
        template_overhead[0] = configs['LMIN'] - lamRange_temp[0] - 5
    if lamRange_temp[1] - configs['LMAX'] > 150.:
        template_overhead[1] = 150.
    else: 
        template_overhead[1] = lamRange_temp[1] - configs['LMAX'] - 5

    # Shorten templates to size of data
    # Reconstruct full original lamRange
    lamRange_lin = np.arange( lamRange_temp[0], lamRange_temp[-1]+ssp_head['CDELT1'], ssp_head['CDELT1'] )
    # Create new lamRange according to LMIN and LMAX from config-file
    constr = np.array([ configs['LMIN'] - template_overhead[0], configs['LMAX'] + template_overhead[1] ])
    idx_lam = np.where( np.logical_and(lamRange_lin > constr[0], lamRange_lin < constr[1] ) )[0]
    lamRange_temp = np.array([ lamRange_lin[idx_lam[0]], lamRange_lin[idx_lam[-1]] ])
    # Shorten data to size of new lamRange
    ssp_data = ssp_data[idx_lam]

    # Convolve templates to same resolution as data
    if len( np.where( LSF_Data(lamRange_lin[idx_lam]) - LSF_Templates(lamRange_lin[idx_lam]) < 0. )[0] ) != 0:        
        message = "According to the specified LSF's, the resolution of the "+\
                  "templates is lower than the resolution of the data. Exit!"
        pipeline.prettyOutput_Failed("Preparing the stellar population templates")
        print("             "+message)
        logging.critical(message)
        exit(1)
    else:
        FWHM_dif = np.sqrt( LSF_Data(lamRange_lin[idx_lam])**2 - LSF_Templates(lamRange_lin[idx_lam])**2 )
        sigma = FWHM_dif/2.355/ssp_head['CDELT1']

    # Create a four dimensional array to store the three dimensional cube of models
    sspNew, _, _ = log_rebin(lamRange_temp, ssp_data, velscale=velscale)
    templates = np.zeros((sspNew.size, nAges, nMetal, nAlpha)); templates[:,:,:,:] = np.nan

    # Arrays to store properties of the models
    logAge_grid = np.empty((nAges, nMetal, nAlpha))
    metal_grid  = np.empty((nAges, nMetal, nAlpha))
    alpha_grid  = np.empty((nAges, nMetal, nAlpha))

    # Sort the templates in the cube of age, metal, alpha
    # This sorts for alpha
    for i, a in enumerate(alpha_str):
        # This sorts for metals
        for k, mh in enumerate(metal_str):
            files = [s for s in ssp_lib if (mh in s and a in s)]
            # This sorts for ages
            for j, filename in enumerate(files):
                hdu = fits.open(filename)
                ssp = hdu[0].data[idx_lam]
                ssp = gaussian_filter1d(ssp, sigma)
                sspNew, logLam2, _ = log_rebin(lamRange_temp, ssp, velscale=velscale)

                logAge_grid[j, k, i] = logAge[j]
                metal_grid[j, k, i]  = metal[k]
                alpha_grid[j, k, i]  = alpha[i]

                # Normalise templates for light-weighted results
                if configs['NORM_TEMP'] == 'LIGHT':
                    templates[:, j, k, i] = sspNew / np.mean(sspNew)
                else:
                    templates[:, j, k, i] = sspNew 

    # Normalise templates for mass-weighted results
    if configs['NORM_TEMP'] == 'MASS':
        templates = templates / np.mean( templates )

    pipeline.prettyOutput_Done("Preparing the stellar population templates")
    logging.info("Prepared the stellar population templates")

    return(templates, lamRange_temp, logLam2, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha)


def age_metal_alpha(passedFiles):
    """
    Function to extract the values of age, metallicity, and alpha-enhancement
    from standard MILES filenames. Note that this function can automatically
    distinguish between template libraries that do or do not include
    alpha-enhancement. 
    """

    out = np.zeros((len(passedFiles),3)); out[:,:] = np.nan

    files = []
    for i in range( len(passedFiles) ):
        files.append( passedFiles[i].split('/')[-1] )

    for num, s in enumerate(files):
        # Ages
        t = s.find('T')
        age = float( s[t+1 : t+8] )
    
        # Metals
        metal = s[s.find('Z')+1 : t]
        if "m" in metal:
            metal = -float(metal[1:])
        elif "p" in metal:
            metal = float(metal[1:])
        else:
            raise ValueError("             This is not a standard MILES filename")
    
        # Alpha
        if s.find('baseFe') == -1:
            EMILES = False
        elif s.find('baseFe') != -1:
            EMILES = True

        if EMILES == False:
            # Usage of MILES: There is a alpha defined
            e = s.find('E')
            alpha = float( s[e+2 : e+6] )
        elif EMILES == True:
            # Usage of EMILES: There is *NO* alpha defined
            alpha = 0.0

        out[num,:] = age, metal, alpha

    Age   = np.unique( out[:,0] )
    Metal = np.unique( out[:,1] )
    Alpha = np.unique( out[:,2] )
    nAges  = len(Age)
    nMetal = len(Metal)
    nAlpha = len(Alpha)
    ncomb = nAges * nMetal * nAlpha

    metal_str = []
    alpha_str = []
    for i in range( len(Metal) ):
        if Metal[i] > 0:
            mm = 'p'+'{:.2f}'.format(np.abs(Metal[i]))+'T'
        elif Metal[i] < 0:
            mm = 'm'+'{:.2f}'.format(np.abs(Metal[i]))+'T'
        metal_str.append(mm)
    for i in range( len(Alpha) ):
        if EMILES == False:
            alpha_str.append( 'Ep'+'{:.2f}'.format(Alpha[i]) )
        elif EMILES == True:
            alpha_str = ['baseFe']

    return( np.log10(Age), Metal, Alpha, metal_str, alpha_str, nAges, nMetal, nAlpha, ncomb )


def save_sfh(mean_result, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, goodPixels,\
             velscale, logLam1, ncomb, nAges, nMetal, nAlpha, ubins, npix, outdir, rootname):
    """ Save all results to disk. """
    # ========================
    # SAVE KINEMATICS
    outfits_sfh = outdir+rootname+'_sfh.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_sfh.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with stellar kinematics
    cols = []
    cols.append( fits.Column(name='BIN_ID',         format='J', array=ubins             ))
    cols.append( fits.Column(name='AGE',            format='D', array=mean_result[:,0]  ))
    cols.append( fits.Column(name='METAL',          format='D', array=mean_result[:,1]  ))
    cols.append( fits.Column(name='ALPHA',          format='D', array=mean_result[:,2]  ))

    cols.append( fits.Column(name='V',              format='D', array=kin[:,0]          ))
    cols.append( fits.Column(name='SIGMA',          format='D', array=kin[:,1]          ))
    cols.append( fits.Column(name='H3',             format='D', array=kin[:,2]          ))
    cols.append( fits.Column(name='H4',             format='D', array=kin[:,3]          ))
    cols.append( fits.Column(name='H5',             format='D', array=kin[:,4]          ))
    cols.append( fits.Column(name='H6',             format='D', array=kin[:,5]          ))

    cols.append( fits.Column(name='FORM_ERR_V',     format='D', array=formal_error[:,0] ))
    cols.append( fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1] ))
    cols.append( fits.Column(name='FORM_ERR_H3',    format='D', array=formal_error[:,2] ))
    cols.append( fits.Column(name='FORM_ERR_H4',    format='D', array=formal_error[:,3] ))
    cols.append( fits.Column(name='FORM_ERR_H5',    format='D', array=formal_error[:,4] ))
    cols.append( fits.Column(name='FORM_ERR_H6',    format='D', array=formal_error[:,5] ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'PPXF_SFH'

    # Create HDU list and write to file
    priHDU  = pipeline.createGISTHeaderComment( priHDU  )
    dataHDU = pipeline.createGISTHeaderComment( dataHDU )

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    pipeline.prettyOutput_Done("Writing: "+rootname+'_sfh.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE WEIGHTS AND GRID
    outfits_sfh = outdir+rootname+'_sfh-weights.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_sfh-weights.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols = []
    cols.append( fits.Column(name='WEIGHTS', format=str(w_row.shape[1])+'D', array=w_row ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'WEIGHTS'

    logAge_row = np.reshape(logAge_grid, ncomb)
    metal_row  = np.reshape(metal_grid,  ncomb)
    alpha_row  = np.reshape(alpha_grid,  ncomb)

    # Table HDU with grids
    cols = []
    cols.append( fits.Column(name='LOGAGE',  format='D',           array=logAge_row  ))
    cols.append( fits.Column(name='METAL',   format='D',           array=metal_row   ))
    cols.append( fits.Column(name='ALPHA',   format='D',           array=alpha_row   ))
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gridHDU.name = 'GRID'

    # Create HDU list and write to file
    priHDU  = pipeline.createGISTHeaderComment( priHDU  )
    dataHDU = pipeline.createGISTHeaderComment( dataHDU )
    gridHDU = pipeline.createGISTHeaderComment( gridHDU )

    HDUList = fits.HDUList([priHDU, dataHDU, gridHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'NAGES',  value=nAges)
    fits.setval(outfits_sfh,'NMETAL', value=nMetal)
    fits.setval(outfits_sfh,'NALPHA', value=nAlpha)

    pipeline.prettyOutput_Done("Writing: "+rootname+'_sfh-weights.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE BESTFIT
    outfits_sfh = outdir+rootname+'_sfh-bestfit.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_sfh-bestfit.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with SFH bestfit
    cols = []
    cols.append( fits.Column(name='BIN_ID',  format='J',           array=ubins   ))
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=bestfit ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'

    # Create HDU list and write to file
    priHDU  = pipeline.createGISTHeaderComment( priHDU  )
    dataHDU = pipeline.createGISTHeaderComment( dataHDU )

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'VELSCALE',value=velscale)
    fits.setval(outfits_sfh,'CRPIX1',  value=1.0)
    fits.setval(outfits_sfh,'CRVAL1',  value=logLam1[0])
    fits.setval(outfits_sfh,'CDELT1',  value=logLam1[1]-logLam1[0])

    pipeline.prettyOutput_Done("Writing: "+rootname+'_sfh-bestfit.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE GOODPIXELS
    outfits_sfh = outdir+rootname+'_sfh-goodpix.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_sfh-goodpix.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with goodpixels
    cols = []
    cols.append( fits.Column(name='GOODPIX', format='J', array=goodPixels  ))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'

    # Create HDU list and write to file
    priHDU     = pipeline.createGISTHeaderComment( priHDU     )
    goodpixHDU = pipeline.createGISTHeaderComment( goodpixHDU )

    HDUList = fits.HDUList([priHDU, goodpixHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'VELSCALE',value=velscale)
    fits.setval(outfits_sfh,'CRPIX1',  value=1.0)
    fits.setval(outfits_sfh,'CRVAL1',  value=logLam1[0])
    fits.setval(outfits_sfh,'CDELT1',  value=logLam1[1]-logLam1[0])

    pipeline.prettyOutput_Done("Writing: "+rootname+'_sfh-goodpix.fits')
    logging.info("Wrote: "+outfits_sfh)


def runModule_SFH(SFH, PARALLEL, configs, dirPath, velscale, LSF_Data, LSF_Templates, outdir, rootname):
    """
    Starts the computation of non-parametric star-formation histories with pPXF.
    A spectral template library is loaded and sorted in a three-dimensional grid
    sampling age, metallicity, and alpha-enhancement. If emission-subtracted
    spectra are available, those are used. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called. 
    """
    if SFH == True:
        print("\033[0;37m"+" - - - - - Running SFH - - - - -"+"\033[0;39m")
        logging.info(" - - - Running SFH - - - ")

        # Prepare template library
        velscale_ratio = 2
        templates, lamRange_temp, logLam_template, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha = \
                setup_spectral_library(configs, velscale/velscale_ratio, LSF_Data, LSF_Templates)

        # Read spectra
        if os.path.isfile(outdir+rootname+'_gandalf-cleaned_BIN.fits') == True:
            logging.info('Using emission-subtracted spectra')
            hdu = fits.open(outdir+rootname+'_gandalf-cleaned_BIN.fits')
        else:
            logging.error('No emission-subtracted spectra for SFH analysis available. Skipping SFH!')
            pipeline.prettyOutput_Warning('No emission-subtracted spectra for SFH analysis available. Skipping SFH!')
            return(0)
        galaxy        = np.array( hdu[1].data.SPEC )
        nbins         = galaxy.shape[0]
        npix          = galaxy.shape[1]
        ubins         = np.arange(0, nbins)
        noise         = np.full(npix, configs['NOISE'])
        logLam_galaxy = hdu[2].data.LOGLAM
        dv            = (np.log(lamRange_temp[0]) - logLam_galaxy[0])*C

        # Implementation of switch FIXED
        # Do fix kin. info to those obtained with PPXF module
        if configs['FIXED'] == 1:
            logging.info('SFH: Stellar kinematics are FIXED to the results of PPXF')
            # Set fixed option to True
            fixed = [True]*configs['MOM']

            # Read PPXF results
            ppxf_data = fits.open(outdir+rootname+'_ppxf.fits')[1].data
            start = np.zeros((nbins, 7))
            for i in range(nbins):
                start[i,:] = np.array( ppxf_data[i][:7] )
            start = start[:,1:configs['MOM']+1]

        # Do *NOT* fix kin. info to those obtained with PPXF module
        elif configs['FIXED'] == 0:
            logging.info('SFH: Stellar kinematics are NOT FIXED to the results of PPXF')
            # Set fixed option to False and use initial guess from Config-file
            fixed = None
            start = np.zeros((nbins, 2))
            for i in range(nbins):
                start[i,:] = np.array( [0.0, configs['SIGMA']] )

        # Define goodpixels
        goodPixels_sfh = util_prepare.spectralMasking(outdir, logLam_galaxy, 'SFH')

        # Define output arrays
        kin          = np.zeros((nbins,6    ))
        w_row        = np.zeros((nbins,ncomb))
        bestfit      = np.zeros((nbins,npix ))
        formal_error = np.zeros((nbins,6    ))

        # ====================
        # Run PPXF_SFH
        start_time = time.time()
        if PARALLEL == True:
            pipeline.prettyOutput_Running("Running PPXF-SFH in parallel mode")            
            logging.info("Running PPXF-SFH in parallel mode")

            # Create Queues
            inQueue  = Queue()
            outQueue = Queue()
        
            # Create worker processes
            ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
                  for _ in range(configs['NCPU'])]
        
            # Start worker processes
            for p in ps: p.start()
        
            # Fill the queue
            for i in range(nbins):
                inQueue.put( ( templates, galaxy[i,:], noise, velscale, start[i,:], goodPixels_sfh, configs['MOM'], dv,\
                               configs['ADEG'], configs['MDEG'], configs['REGUL_ERR'], fixed, velscale_ratio, npix,\
                               ncomb, nbins, i ) )
        
            # now get the results with indices
            ppxf_tmp = [outQueue.get() for _ in range(nbins)]
        
            # send stop signal to stop iteration
            for _ in range(configs['NCPU']): inQueue.put('STOP')

            # stop processes
            for p in ps: p.join()
        
            # Get output
            index = np.zeros(nbins)
            for i in range(0, nbins):
                index[i]                        = ppxf_tmp[i][0]
                kin[i,:configs['MOM']]          = ppxf_tmp[i][1]
                w_row[i,:]                      = ppxf_tmp[i][2]
                bestfit[i,:]                    = ppxf_tmp[i][3]
                formal_error[i,:configs['MOM']] = ppxf_tmp[i][4]
            # Sort output
            argidx = np.argsort( index )
            kin          = kin[argidx,:]
            w_row        = w_row[argidx,:]
            bestfit      = bestfit[argidx,:]
            formal_error = formal_error[argidx,:]

            pipeline.prettyOutput_Done("Running PPXF in parallel mode", progressbar=True)

        if PARALLEL == False:
            pipeline.prettyOutput_Running("Running PPXF-SFH in serial mode")            
            logging.info("Running PPXF-SFH in serial mode")
            for i in range(nbins):
                kin[i,:configs['MOM']], w_row[i,:], bestfit[i,:], formal_error[i,:configs['MOM']] = run_ppxf\
                    (templates, galaxy[i,:], noise, velscale, start[i,:], goodPixels_sfh, configs['MOM'], dv, \
                    configs['ADEG'], configs['MDEG'], configs['REGUL_ERR'], fixed, velscale_ratio, npix, ncomb, nbins, i)
            pipeline.prettyOutput_Done("Running PPXF-SFH in serial mode", progressbar=True)

        print("             Running PPXF_SFH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))
        logging.info("Running PPXF_SFH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))

        # Check for exceptions which occurred during the analysis
        idx_error = np.where( np.isnan( kin[:,0] ) == True )[0]
        if len(idx_error) != 0:
            pipeline.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
            print("             "+str(idx_error))
            logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
        else:
            print("             "+"There were no problems in the analysis.")
            logging.info("There were no problems in the analysis.")
        print("")

        # Calculate mean age, metallicity and alpha
        mean_results = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins)

        # Save to file
        save_sfh(mean_results, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, goodPixels_sfh, \
                velscale, logLam_galaxy, ncomb, nAges, nMetal, nAlpha, ubins, npix, outdir, rootname)

        # Plot maps
        try:
            pipeline.prettyOutput_Running("Producing SFH maps")
            logging.info("Producing SFH maps")
            util_plot.plot_maps('SFH', outdir)     # Plot ages, metallicities and alpha
            util_plot_kin.plot_maps('SFH', outdir) # Plot kinematics
            pipeline.prettyOutput_Done("Producing SFH maps")
        except:
            pipeline.prettyOutput_Failed("Producing SFH maps")
            logging.warning("Failed to produce SFH maps. Analysis continues!")
            pass

        print("\033[0;37m"+" - - - - - SFH done - - - - -"+"\033[0;39m")
        print("")
        logging.info(" - - - SFH Done - - - \n")
        
    elif SFH == False:
        print("")
        print(pipeline.prettyOutput_WarningPrefix()+"Skipping SFH!")
        print("")
        logging.warning("Skipping SFH analysis!\n")
