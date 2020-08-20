import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import time
import os
import glob
import logging

from printStatus import printStatus 

from gistPipeline.prepareTemplates import _prepareTemplates
from gistPipeline.auxiliary import _auxiliary

# Then use system installed version instead
from ppxf.ppxf      import ppxf

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
        mdeg, regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i\
        in iter(inQueue.get, 'STOP'): 

        sol, w_row, bestfit, formal_error = run_ppxf(templates, galaxy, noise, velscale, start, goodPixels_sfh, mom, \
                                            dv, mdeg, regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i) 

        outQueue.put(( i, sol, w_row, bestfit, formal_error ))


def run_ppxf(templates, galaxy_i, noise_i, velscale, start, goodPixels, nmom, dv, mdeg,\
             regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004 
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C; 
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories. 
    """ 
    printStatus.progressBar(i, nbins, barLength = 50)

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


def save_sfh(mean_result, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, logLam_galaxy, goodPixels,\
             velscale, logLam1, ncomb, nAges, nMetal, nAlpha, npix, config):
    """ Save all results to disk. """
    # ========================
    # SAVE KINEMATICS
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with stellar kinematics
    cols = []
    cols.append(fits.Column(name='AGE',   format='D', array=mean_result[:,0]))
    cols.append(fits.Column(name='METAL', format='D', array=mean_result[:,1]))
    cols.append(fits.Column(name='ALPHA', format='D', array=mean_result[:,2]))

    if config['SFH']['FIXED'] == False:
        cols.append(fits.Column(name='V',     format='D', array=kin[:,0]))
        cols.append(fits.Column(name='SIGMA', format='D', array=kin[:,1]))
        if np.any(kin[:,2]) != 0: cols.append(fits.Column(name='H3', format='D', array=kin[:,2]))
        if np.any(kin[:,3]) != 0: cols.append(fits.Column(name='H4', format='D', array=kin[:,3]))
        if np.any(kin[:,4]) != 0: cols.append(fits.Column(name='H5', format='D', array=kin[:,4]))
        if np.any(kin[:,5]) != 0: cols.append(fits.Column(name='H6', format='D', array=kin[:,5]))
    
        cols.append(fits.Column(name='FORM_ERR_V',     format='D', array=formal_error[:,0]))
        cols.append(fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1]))
        if np.any(formal_error[:,2]) != 0: cols.append(fits.Column(name='FORM_ERR_H3', format='D', array=formal_error[:,2]))
        if np.any(formal_error[:,3]) != 0: cols.append(fits.Column(name='FORM_ERR_H4', format='D', array=formal_error[:,3]))
        if np.any(formal_error[:,4]) != 0: cols.append(fits.Column(name='FORM_ERR_H5', format='D', array=formal_error[:,4]))
        if np.any(formal_error[:,5]) != 0: cols.append(fits.Column(name='FORM_ERR_H6', format='D', array=formal_error[:,5]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SFH'

    # Create HDU list and write to file
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE WEIGHTS AND GRID
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-weights.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')

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
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    gridHDU = _auxiliary.saveConfigToHeader(gridHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU, gridHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'NAGES',  value=nAges)
    fits.setval(outfits_sfh,'NMETAL', value=nMetal)
    fits.setval(outfits_sfh,'NALPHA', value=nAlpha)

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE BESTFIT
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-bestfit.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with SFH bestfit
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=bestfit ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'

    # Table HDU with SFH logLam
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam_galaxy ))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM'

    # Table HDU with SFH goodpixels
    cols = []
    cols.append( fits.Column(name='GOODPIX', format='J', array=goodPixels ))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'

    # Create HDU list and write to file
    priHDU     = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU    = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    logLamHDU  = _auxiliary.saveConfigToHeader(logLamHDU, config['SFH'])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'VELSCALE',value=velscale)
    fits.setval(outfits_sfh,'CRPIX1',  value=1.0)
    fits.setval(outfits_sfh,'CRVAL1',  value=logLam1[0])
    fits.setval(outfits_sfh,'CDELT1',  value=logLam1[1]-logLam1[0])

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')
    logging.info("Wrote: "+outfits_sfh)


def extractStarFormationHistories(config):
    """ 
    Starts the computation of non-parametric star-formation histories with
    pPXF.  A spectral template library sorted in a three-dimensional grid of
    age, metallicity, and alpha-enhancement is loaded.  Emission-subtracted
    spectra are used for the fit. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called.
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config)

    # Prepare template library
    velscale = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+"_BinSpectra.fits")[0].header['VELSCALE']
    velscale_ratio = 2
    templates, lamRange_temp, logLam_template, ntemplates, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha = \
            _prepareTemplates.prepareTemplates_Module(config, config['SFH']['LMIN'], config['SFH']['LMAX'], velscale/velscale_ratio, LSF_Data, LSF_Templates, sortInGrid=True)

    # Read spectra
    if os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits') == True:
        logging.info('Using emission-subtracted spectra at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
        printStatus.done("Using emission-subtracted spectra")
        hdu = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
    else:
        logging.info('Using regular spectra without any emission-correction at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
        printStatus.done("Using regular spectra without any emission-correction")
        hdu = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')

    galaxy        = np.array( hdu[1].data.SPEC )
    logLam_galaxy = hdu[2].data.LOGLAM
    idx_lam       = np.where( np.logical_and( np.exp(logLam_galaxy) > config['SFH']['LMIN'], np.exp(logLam_galaxy) < config['SFH']['LMAX'] ) )[0]
    galaxy        = galaxy[:,idx_lam]
    logLam_galaxy = logLam_galaxy[idx_lam]
    nbins         = galaxy.shape[0]
    npix          = galaxy.shape[1]
    ubins         = np.arange(0, nbins)
    noise         = np.full(npix, config['SFH']['NOISE'])
    dv            = (np.log(lamRange_temp[0]) - logLam_galaxy[0])*C

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config['SFH']['FIXED'] == True:
        logging.info('Stellar kinematics are FIXED to the results obtained before.')
        # Set fixed option to True
        fixed = [True]*config['KIN']['MOM']

        # Read PPXF results
        ppxf_data = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin.fits')[1].data
        start = np.zeros((nbins, config['KIN']['MOM']))
        for i in range(nbins):
            start[i,:] = np.array( ppxf_data[i][:config['KIN']['MOM']] )

    # Do *NOT* fix kinematics to those obtained previously
    elif config['SFH']['FIXED'] == False:
        logging.info('Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties.')
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, 2))
        for i in range(nbins):
            start[i,:] = np.array( [0.0, config['SFH']['SIGMA']] )

    # Define goodpixels
    goodPixels_sfh = _auxiliary.spectralMasking(config, config['SFH']['SPEC_MASK'], logLam_galaxy)

    # Define output arrays
    kin          = np.zeros((nbins,6    ))
    w_row        = np.zeros((nbins,ncomb))
    bestfit      = np.zeros((nbins,npix ))
    formal_error = np.zeros((nbins,6    ))

    # ====================
    # Run PPXF
    start_time = time.time()
    if config['GENERAL']['PARALLEL'] == True:
        printStatus.running("Running PPXF in parallel mode")            
        logging.info("Running PPXF in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()
    
        # Create worker processes
        ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
              for _ in range(config['GENERAL']['NCPU'])]
    
        # Start worker processes
        for p in ps: p.start()
    
        # Fill the queue
        for i in range(nbins):
            inQueue.put( ( templates, galaxy[i,:], noise, velscale, start[i,:], goodPixels_sfh, config['SFH']['MOM'], dv,\
                           config['SFH']['MDEG'], config['SFH']['REGUL_ERR'], fixed, velscale_ratio, npix,\
                           ncomb, nbins, i ) )
    
        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]
    
        # send stop signal to stop iteration
        for _ in range(config['GENERAL']['NCPU']): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()
    
        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]                        = ppxf_tmp[i][0]
            kin[i,:config['SFH']['MOM']]    = ppxf_tmp[i][1]
            w_row[i,:]                      = ppxf_tmp[i][2]
            bestfit[i,:]                    = ppxf_tmp[i][3]
            formal_error[i,:config['SFH']['MOM']] = ppxf_tmp[i][4]
        # Sort output
        argidx = np.argsort( index )
        kin          = kin[argidx,:]
        w_row        = w_row[argidx,:]
        bestfit      = bestfit[argidx,:]
        formal_error = formal_error[argidx,:]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")            
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            kin[i,:config['SFH']['MOM']], w_row[i,:], bestfit[i,:], formal_error[i,:config['SFH']['MOM']] = run_ppxf\
                (templates, galaxy[i,:], noise, velscale, start[i,:], goodPixels_sfh, config['SFH']['MOM'], dv, \
                config['SFH']['MDEG'], config['SFH']['REGUL_ERR'], fixed, velscale_ratio, npix, ncomb, nbins, i)
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print("             Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))
    logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( kin[:,0] ) == True )[0]
    if len(idx_error) != 0:
        printStatus.warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Calculate mean age, metallicity and alpha
    mean_results = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins)

    # Save to file
    save_sfh(mean_results, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, logLam_galaxy, goodPixels_sfh, \
            velscale, logLam_galaxy, ncomb, nAges, nMetal, nAlpha, npix, config)

    # Return
    return(None)


