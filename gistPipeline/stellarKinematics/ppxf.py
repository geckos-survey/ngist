import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import time
import logging
import os

from printStatus import printStatus

from gistPipeline.prepareTemplates import _prepareTemplates
from gistPipeline.auxiliary import _auxiliary

from ppxf.ppxf import ppxf

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE: 
  This module executes the analysis of stellar kinematics in the pipeline. 
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C). 
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, bin_data, noise, velscale, start, goodPixels_ppxf, nmoments,\
        adeg, mdeg, reddening, logLam, offset, velscale_ratio, nsims, nbins, i\
        in iter(inQueue.get, 'STOP'):

        sol, ppxf_reddening, bestfit, optimal_template, mc_results, formal_error = \
          run_ppxf(templates, bin_data, noise, velscale, start, goodPixels_ppxf, nmoments, adeg, mdeg, reddening, logLam, offset, velscale_ratio, nsims, nbins, i)

        outQueue.put(( i, sol, ppxf_reddening, bestfit, optimal_template, mc_results, formal_error ))


def run_ppxf( templates, log_bin_data, log_bin_error, velscale, start, goodPixels, nmoments, adeg, mdeg, reddening, logLam, \
        offset, velscale_ratio, nsims, nbins, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004 
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C; 
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics. 
    """
    printStatus.progressBar( i, nbins, barLength=50 )

    try:
        # Call PPXF
        pp = ppxf(templates, log_bin_data, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
                  quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, reddening=reddening, lam=np.exp(logLam), velscale_ratio=velscale_ratio, vsyst=offset)
    
        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]
        
        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)
    
        # Do MC-Simulations
        sol_MC     = np.zeros((nsims,nmoments))
        mc_results = np.zeros(nmoments)
        for o in range(0, nsims):
            # Add noise to bestfit: 
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit  +  np.random.normal(0, 1, len(log_bin_data)) * np.std( log_bin_data[goodPixels] - pp.bestfit[goodPixels] )
    
            mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
                    quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, velscale_ratio=velscale_ratio, vsyst=offset, bias=0.0)
            sol_MC[o,:] = mc.sol[:]
     
        if nsims != 0:
            mc_results = np.nanstd( sol_MC, axis=0 )
     
        return(pp.sol[:], pp.reddening, pp.bestfit, optimal_template, mc_results, formal_error)

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan )


def calc_LambdaR(config, ppxf_result, nbins):
    """
    Calculate the lambda parameter as a proxy for the projected, specific
    angular momentum of the galaxy (see Emsellem et al. 2007;
    ui.adsabs.harvard.edu/#abs/2007MNRAS.379..401E). Note that this quantity is
    not calculated as integrated value per galaxy, but for every Voronoi-bin
    individually.  
    """
    # Calculate lambda_r
    hdu  = fits.open(os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'])+'_table.fits')
    BIN_ID = hdu[1].data.BIN_ID
    X      = hdu[1].data.X
    Y      = hdu[1].data.Y
    FLUX   = hdu[1].data.FLUX 

    velocity = ppxf_result[:,0] - np.median(ppxf_result[:,0])
    sigma    = ppxf_result[:,1] 

    lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
    for i in range(0, nbins):
        idx = np.where( BIN_ID == i )[0]
        numerator   = 0
        denominator = 0
        for o in range(0, len(idx)):
            radius = np.sqrt( X[idx[o]]**2 + Y[idx[o]]**2 )
            numerator   += FLUX[idx[o]] * radius * np.abs( velocity[i] )
            denominator += FLUX[idx[o]] * radius * np.sqrt( velocity[i]**2 + sigma[i]**2 )
        try:
            lambda_r[i] = numerator / denominator
        except ZeroDivisionError:
            lambda_r[i] = np.nan

    return( lambda_r )


def save_ppxf(config, ppxf_result, ppxf_reddening, mc_results, formal_error, lambda_r,\
              ppxf_bestfit, logLam, goodPixels, optimal_template, logLam_template, npix):
    """ Saves all results to disk. """
    # ========================
    # SAVE RESULTS
    outfits_ppxf = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_kin.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name='V' ,    format='D', array=ppxf_result[:,0]))
    cols.append(fits.Column(name='SIGMA', format='D', array=ppxf_result[:,1]))
    if np.any(ppxf_result[:,2]) != 0: cols.append(fits.Column(name='H3',       format='D', array=ppxf_result[:,2]))
    if np.any(ppxf_result[:,3]) != 0: cols.append(fits.Column(name='H4',       format='D', array=ppxf_result[:,3]))
    if np.any(ppxf_result[:,4]) != 0: cols.append(fits.Column(name='H5',       format='D', array=ppxf_result[:,4]))
    if np.any(ppxf_result[:,5]) != 0: cols.append(fits.Column(name='H6',       format='D', array=ppxf_result[:,5]))
    if np.any(lambda_r[:])      != 0: cols.append(fits.Column(name='LAMBDA_R', format='D', array=lambda_r[:]     ))

    if np.any(mc_results[:,0]) != 0: cols.append(fits.Column(name='ERR_V' ,    format='D', array=mc_results[:,0]))
    if np.any(mc_results[:,1]) != 0: cols.append(fits.Column(name='ERR_SIGMA', format='D', array=mc_results[:,1]))
    if np.any(mc_results[:,2]) != 0: cols.append(fits.Column(name='ERR_H3',    format='D', array=mc_results[:,2]))
    if np.any(mc_results[:,3]) != 0: cols.append(fits.Column(name='ERR_H4',    format='D', array=mc_results[:,3]))
    if np.any(mc_results[:,4]) != 0: cols.append(fits.Column(name='ERR_H5',    format='D', array=mc_results[:,4]))
    if np.any(mc_results[:,5]) != 0: cols.append(fits.Column(name='ERR_H6',    format='D', array=mc_results[:,5]))

    cols.append(fits.Column(name='FORM_ERR_V' ,    format='D', array=formal_error[:,0]))
    cols.append(fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1]))
    if np.any(formal_error[:,2]) != 0: cols.append(fits.Column(name='FORM_ERR_H3', format='D', array=formal_error[:,2]))
    if np.any(formal_error[:,3]) != 0: cols.append(fits.Column(name='FORM_ERR_H4', format='D', array=formal_error[:,3]))
    if np.any(formal_error[:,4]) != 0: cols.append(fits.Column(name='FORM_ERR_H5', format='D', array=formal_error[:,4]))
    if np.any(formal_error[:,5]) != 0: cols.append(fits.Column(name='FORM_ERR_H6', format='D', array=formal_error[:,5]))

    if np.any(np.isnan(ppxf_reddening)) != True: cols.append(fits.Column(name='REDDENING', format='D', array=ppxf_reddening[:]))
    
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'KIN_DATA'
    
    # Create HDU list and write to file
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['KIN'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['KIN'])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)
    
    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_kin.fits')
    logging.info("Wrote: "+outfits_ppxf)
    
    
    # ========================
    # SAVE BESTFIT
    outfits_ppxf = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin-bestfit.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_kin-bestfit.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'

    # Table HDU with PPXF logLam
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=logLam))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM'
   
    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name='GOODPIX', format='J', array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'
     
    # Create HDU list and write to file
    priHDU     = _auxiliary.saveConfigToHeader(priHDU, config['KIN'])
    dataHDU    = _auxiliary.saveConfigToHeader(dataHDU, config['KIN'])
    logLamHDU  = _auxiliary.saveConfigToHeader(logLamHDU, config['KIN'])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config['KIN'])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)
    
    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_kin-bestfit.fits')
    logging.info("Wrote: "+outfits_ppxf)
    
       
    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin-optimalTemplates.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_kin-optimalTemplates.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='OPTIMAL_TEMPLATES', format=str(optimal_template.shape[1])+'D', array=optimal_template ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'OPTIMAL_TEMPLATES'
    
    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=logLam_template) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM_TEMPLATE'
    
    # Create HDU list and write to file
    priHDU    = _auxiliary.saveConfigToHeader(priHDU, config['KIN'])
    dataHDU   = _auxiliary.saveConfigToHeader(dataHDU, config['KIN'])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config['KIN'])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_kin-optimalTemplates.fits')
    logging.info("Wrote: "+outfits)


def extractStellarKinematics(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and 
    saves the outputs following the GIST conventions. 
    """
    # Read data from file
    hdu      = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
    bin_data = np.array( hdu[1].data.SPEC.T )
    logLam   = np.array( hdu[2].data.LOGLAM )
    idx_lam  = np.where( np.logical_and( np.exp(logLam) > config['KIN']['LMIN'], np.exp(logLam) < config['KIN']['LMAX'] ) )[0]
    bin_data = bin_data[idx_lam,:]
    logLam   = logLam[idx_lam]
    npix     = bin_data.shape[0]
    nbins    = bin_data.shape[1]
    ubins    = np.arange(0, nbins)
    velscale = hdu[0].header['VELSCALE']

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config)

    # Prepare templates
    velscale_ratio = 2
    logging.info("Using full spectral library for PPXF")
    templates, lamRange_spmod, logLam_template, ntemplates = _prepareTemplates.prepareTemplates_Module\
            (config, config['KIN']['LMIN'], config['KIN']['LMAX'], velscale/velscale_ratio, LSF_Data, LSF_Templates)[:4]
    templates = templates.reshape( (templates.shape[0], ntemplates) )

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    noise  = np.ones((npix,nbins))
    nsims  = config['KIN']['MC_PPXF']

    # Initial guesses 
    start = np.zeros((nbins,2))
    if os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin-guess.fits') == True:
        printStatus.done("Using V and SIGMA from '"+config['GENERAL']['RUN_ID']+"_kin-guess.fits' as initial guesses")
        logging.info("Using V and SIGMA from '"+config['GENERAL']['RUN_ID']+"_kin-guess.fits' as initial guesses")
        guess      = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin-guess.fits')[1].data
        start[:,0] = guess.V
        start[:,1] = guess.SIGMA
    else: 
        # Use the same initial guess for all bins, as stated in MasterConfig
        printStatus.done("Using V and SIGMA from the MasterConfig file as initial guesses")
        logging.info("Using V and SIGMA from the MasterConfig file as initial guesses")
        start[:,0] = 0.0
        start[:,1] = config['KIN']['SIGMA']

    # Define goodpixels
    goodPixels_ppxf = _auxiliary.spectralMasking(config, config['KIN']['SPEC_MASK'], logLam)

    # Array to store results of ppxf
    ppxf_result        = np.zeros((nbins,6))
    ppxf_reddening     = np.zeros(nbins)
    ppxf_bestfit       = np.zeros((nbins,npix))
    optimal_template   = np.zeros((nbins,templates.shape[0]))
    mc_results         = np.zeros((nbins,6))
    formal_error       = np.zeros((nbins,6))

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
            inQueue.put( ( templates, bin_data[:,i], noise[:,i], velscale, start[i,:], goodPixels_ppxf,\
                            config['KIN']['MOM'], config['KIN']['ADEG'], config['KIN']['MDEG'], config['KIN']['REDDENING'], logLam, offset, velscale_ratio,\
                            nsims, nbins, i) )
    
        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]
    
        # send stop signal to stop iteration
        for _ in range(config['GENERAL']['NCPU']): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()
    
        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]                              = ppxf_tmp[i][0]
            ppxf_result[i,:config['KIN']['MOM']]  = ppxf_tmp[i][1]
            ppxf_reddening[i]                     = ppxf_tmp[i][2]
            ppxf_bestfit[i,:]                     = ppxf_tmp[i][3]
            optimal_template[i,:]                 = ppxf_tmp[i][4]
            mc_results[i,:config['KIN']['MOM']]   = ppxf_tmp[i][5]
            formal_error[i,:config['KIN']['MOM']] = ppxf_tmp[i][6]
        # Sort output
        argidx = np.argsort( index )
        ppxf_result      = ppxf_result[argidx,:]
        ppxf_reddening   = ppxf_reddening[argidx]
        ppxf_bestfit     = ppxf_bestfit[argidx,:]
        optimal_template = optimal_template[argidx,:]
        mc_results       = mc_results[argidx,:]
        formal_error     = formal_error[argidx,:]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    elif config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            ppxf_result[i,:config['KIN']['MOM']], ppxf_reddening[i], ppxf_bestfit[i,:], optimal_template[i,:],\
              mc_results[i,:config['KIN']['MOM']], formal_error[i,:config['KIN']['MOM']] = run_ppxf\
                (templates, bin_data[:,i], noise[:,i], velscale, start[i,:], goodPixels_ppxf,\
                config['KIN']['MOM'], config['KIN']['ADEG'], config['KIN']['MDEG'], config['KIN']['REDDENING'], logLam, offset, velscale_ratio,\
                nsims, nbins, i)
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print("             Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))
    logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]
    if len(idx_error) != 0:
        printStatus.warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Calculate LAMBDA_R
    printStatus.running("Calculating Lambda_R")
    logging.info("Calculating Lambda_R")
    logging.info("To obtain correct lambda_R measurements, the coordinate system must be centred on the centre of the galaxy. Use the 'READ_DATA|ORIGIN' parameter to do so.")
    lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
    lambda_r = calc_LambdaR(config, ppxf_result, nbins)
    printStatus.updateDone("Calculating Lambda_R")

    # Save stellar kinematics to file
    save_ppxf(config, ppxf_result, ppxf_reddening, mc_results, formal_error, lambda_r, ppxf_bestfit, logLam, goodPixels_ppxf, optimal_template, logLam_template, npix)

    # Return
    return(None)
