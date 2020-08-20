from astropy.io import fits, ascii
import numpy as np
from multiprocessing import Queue, Process

import time
import logging
import os

from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary 

from gistPipeline.lineStrengths import lsindex_spec as lsindex
from gistPipeline.lineStrengths import ssppop_fitting as ssppop

# Then use system installed version instead
from ppxf.ppxf_util import gaussian_filter1d

cvel  = 299792.458


"""
PURPOSE: 
  This module executes the measurement of line strength indices in the pipeline.
  Basically, it acts as an interface between pipeline and the line strength
  measurement routines of Kuntschner et al. 2006
  (ui.adsabs.harvard.edu/?#abs/2006MNRAS.369..497K) and their conversion to
  single stellar population equivalent population properties with the MCMC
  algorithm of Martin-Navaroo et al. 2018
  (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M). 
"""


def workerLS(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for wave, spec, espec, redshift, config, lickfile, names, index_names,\
        model_indices, params, tri, labels, nbins, i, MCMC\
        in iter(inQueue.get, 'STOP'):

        if MCMC == True: 
            indices, errors, vals, percentile = run_ls( wave, spec, espec, redshift, config, lickfile, names, index_names,\
            model_indices, params, tri, labels, nbins, i, MCMC )

            outQueue.put(( i, indices, errors, vals, percentile ))

        elif MCMC == False: 
            indices, errors = run_ls( wave, spec, espec, redshift, config, lickfile, names, index_names,\
            model_indices, params, tri, labels, nbins, i, MCMC )

            outQueue.put(( i, indices, errors ))


def run_ls(wave, spec, espec, redshift, config, lickfile, names, index_names,\
           model_indices, params, tri, labels, nbins, i, MCMC): 
    """
    Calls a Python version of the line strength measurement routine of
    Kuntschner et al. 2006 (ui.adsabs.harvard.edu/?#abs/2006MNRAS.369..497K),
    and if required, the MCMC algorithm from Martin-Navaroo et al. 2018
    (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M) to determine SSP
    properties. 
    """
    printStatus.progressBar(i, nbins, barLength = 50)
    nindex = len(index_names)

    try:
        # Measure the LS indices
        names, indices, errors = lsindex.lsindex\
                    (wave, spec, espec, redshift[0], lickfile, sims=config['LS']['MC_LS'], z_err=redshift[1], plot=0)
    
        # Get the indices in consideration
        data  = np.zeros(nindex)
        error = np.zeros(nindex)
        for o in range( nindex ):
            idx = np.where( names == index_names[o] )[0]
            data[o]  = indices[idx]
            error[o] = errors[idx]
    
        if MCMC == True: 
            # Run the conversion of LS indices to SSP properties
            vals   = np.zeros(len(labels)*3+2)
            chains = np.zeros((int(config['LS']['NWALKER']*config['LS']['NCHAIN']/2), len(labels)))
            vals[:], chains[:,:] = ssppop.ssppop_fitting\
                (data, error, model_indices, params, tri, labels, config['LS']['NWALKER'], config['LS']['NCHAIN'], False, 0, i, nbins, "")
        
            percentiles = np.percentile( chains, np.arange(101), axis=0 )
    
            return(indices, errors, vals, percentiles)
    
        elif MCMC == False: 
            return(indices, errors)

    except:
        if MCMC == True: 
            return( np.nan, np.nan, np.nan, np.nan )
        elif MCMC == False: 
            return( np.nan, np.nan )


def save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, config, vals=None, percentile=None ):
    """ Saves all results to disk. """
    # Save results
    if RESOLUTION == 'ORIGINAL': 
        outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls_OrigRes.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_ls_OrigRes.fits')
    if RESOLUTION == 'ADAPTED': 
        outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls_AdapRes.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_ls_AdapRes.fits')
   
    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with results
    cols = []
    if MCMC == True: 
        nparam = len(labels)
        for i in range(nparam):
            cols.append(fits.Column(name=labels[i], format='D', array=percentile[:,50,i]))
        cols.append(fits.Column(name='lnP',  format='D', array=vals[:,-2]))
        cols.append(fits.Column(name='Flag', format='D', array=vals[:,-1]))

    ndim = len(names)
    for i in range(ndim):
        if np.any(np.isnan(ls_indices[:,i])) == False: cols.append(fits.Column(name=names[i],        format='D', array=ls_indices[:,i]))
        if np.any(np.isnan(ls_errors[:,i]))  == False: cols.append(fits.Column(name="ERR_"+names[i], format='D', array=ls_errors[:,i]))
    cols.append(fits.Column(name="FWHM_FLAG", format='I', array=totalFWHM_flag[:]))
    lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    lsHDU.name = "LS_DATA"

    # Extension 2: Table HDU with percentiles
    if MCMC == True:
        cols = []
        nparam = len(labels)
        for i in range(nparam):
            cols.append(fits.Column(name=labels[i]+'_PERCENTILES', format='101D', array=percentile[:,:,i]))
        percentilesHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)) 
        percentilesHDU.anem = "PERCENTILES"

    # Create HDUList
    if  MCMC == False: HDUList = fits.HDUList([priHDU, lsHDU])
    elif MCMC == True: HDUList = fits.HDUList([priHDU, lsHDU, percentilesHDU])

    # Write HDU list to file
    HDUList.writeto(outfits, overwrite=True)

    if RESOLUTION == 'ORIGINAL': 
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_ls_OrigRes.fits')
    if RESOLUTION == 'ADAPTED': 
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_ls_AdapRes.fits')
    logging.info("Wrote: "+outfits)


def saveCleanedLinearSpectra(spec, espec, wave, npix, config):
    """ Save emission-subtracted, linearly binned spectra to disk. """
    outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls-cleaned_linear.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_ls-cleaned_linear.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with cleaned, linear spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=spec ) )
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=espec ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'CLEANED_SPECTRA'

    # Extension 2: Table HDU with wave
    cols = []
    cols.append( fits.Column(name='LAM', format='D', array=wave) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LAM'
     
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_ls-cleaned_linear.fits')
    logging.info("Wrote: "+outfits)


def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline. 
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1]-lamRange[0]) / (n - 1)             # Step in log-space
    lim = lamRange + np.array([-0.5, 0.5])*dLam            # Min and max wavelength in log-space
    borders = np.linspace( lim[0], lim[1], n+1 )           # OLD logLam in log-space
    
    # Wavelength domain
    logLim     = np.exp(lim)                               # Min and max wavelength in Angst.
    lamNew     = np.linspace( logLim[0], logLim[1], m+1 )  # new logLam in Angstroem
    newBorders = np.log(lamNew)                            # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor( (newBorders-lim[0]) / dLam ).astype('int')

    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m-1):
        a = (newBorders[j]   - borders[k[j]])   / dLam
        b = (borders[k[j+1]] - newBorders[j+1]) / dLam
        
        specNew[j] = np.sum( spec[k[j]:k[j+1]] ) - a*spec[k[j]] - b*spec[k[j+1]]

    # Rescale flux
    if flux == True:
        specNew = specNew / ( newBorders[1:] - newBorders[:-1] ) * np.mean( newBorders[1:] - newBorders[:-1] ) * oversample

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1]-lamNew[0])

    return( specNew, lamNew )


def measureLineStrengths(config, RESOLUTION='ORIGINAL'):
    """
    Starts the line strength analysis. Data is read in, emission-subtracted
    spectra are rebinned from logarithmic to linear scale, and the spectra
    convolved to meet the LIS measurement resolution. After the measurement of
    line strength indices and, if required, the estimation of SSP properties,
    the results are saved to file. 
    """
    # Run MCMC only on the indices measured from convoluted spectra
    if config['LS']['TYPE'] == 'SPP'  and  RESOLUTION == "ADAPTED": 
        MCMC = True
    else: 
        MCMC = False

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config)

    # Read the log-rebinned spectra and log-unbin them
    if os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls-cleaned_linear.fits') == False:

        # Read spectra
        if os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits') == True:
            logging.info('Using emission-subtracted spectra at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
            printStatus.done("Using emission-subtracted spectra")            
            hdu_spec = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
        else:
            logging.info('Using regular spectra without any emission-correction at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
            printStatus.done("Using regular spectra without any emission-correction")
            hdu_spec = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
        hdu_espec = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
        idx_lamMin = np.where( hdu_spec[2].data.LOGLAM[0]  == hdu_espec[2].data.LOGLAM )[0]
        idx_lamMax = np.where( hdu_spec[2].data.LOGLAM[-1] == hdu_espec[2].data.LOGLAM )[0]
        idx_lam    = np.arange(idx_lamMin, idx_lamMax+1)
        oldspec  = np.array( hdu_spec[1].data.SPEC   )
        oldespec = np.sqrt( np.array( hdu_espec[1].data.ESPEC )[:,idx_lam] )
        wave     = np.array( hdu_spec[2].data.LOGLAM )

        nbins    = oldspec.shape[0]
        npix     = oldspec.shape[1]
        lamRange = np.array([ wave[0], wave[-1] ])
        spec     = np.zeros( oldspec.shape  )
        espec    = np.zeros( oldespec.shape )

        # Rebin the cleaned spectra from log to lin
        printStatus.running("Rebinning the spectra from log to lin")
        for i in range( nbins ):
            printStatus.progressBar(i, nbins, barLength = 50)
            spec[i,:], wave = log_unbinning( lamRange, oldspec[i,:] )
        printStatus.updateDone("Rebinning the spectra from log to lin", progressbar=True)

        # Rebin the error spectra from log to lin
        printStatus.running("Rebinning the error spectra from log to lin")
        for i in range( nbins ):
            printStatus.progressBar(i, nbins, barLength = 50)
            espec[i,:], _ = log_unbinning( lamRange, oldespec[i,:] )
        printStatus.updateDone("Rebinning the error spectra from log to lin", progressbar=True)

        # Save cleaned, linear spectra
        saveCleanedLinearSpectra(spec, espec, wave, npix, config)
    
    # Read the linearly-binned, cleaned spectra provided by previous LS-run
    else: 
        logging.info("Reading "+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls-cleaned_linear.fits')
        hdu   = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_ls-cleaned_linear.fits')
        spec  = np.array( hdu[1].data.SPEC  )
        espec = np.array( hdu[1].data.ESPEC )
        wave  = np.array( hdu[2].data.LAM   )
        nbins = spec.shape[0]

    # Read PPXF results
    ppxf_data     = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin.fits')[1].data
    redshift      = np.zeros((nbins, 2))                       # Dimensionless z
    redshift[:,0] = np.array( ppxf_data.V[:] ) / cvel          # Redshift
    redshift[:,1] = np.array( ppxf_data.FORM_ERR_V[:] ) / cvel # Error on redshift
    veldisp_kin   = np.array( ppxf_data.SIGMA[:] )

    # Read file defining the LS bands
    lickfile = os.path.join(config['GENERAL']['CONFIG_DIR'],config['LS']['LS_FILE'])
    tab   = ascii.read(lickfile, comment='\s*#')
    names = tab['names']

    # Flag spectra for which the total intrinsic dispersion is larger than the LIS measurement resolution
    totalFWHM_flag = np.zeros(spec.shape[0])

    # Broaden spectra to LIS resolution taking into account the measured velocity dispersion
    if RESOLUTION == "ADAPTED":
        printStatus.running("Broadening the spectra to LIS resolution")
        velscale = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+"_BinSpectra.fits")[0].header['VELSCALE']
        # Iterate over all bins
        for i in range(0, spec.shape[0]):
            printStatus.progressBar(i, nbins, barLength = 50)

            # Convert velocity dispersion of galaxy (from PPXF) to Angstrom
            veldisp_kin_Angst = veldisp_kin[i] * wave / cvel * 2.355  

            # Total dispersion for this bin
            total_dispersion = np.sqrt( LSF_Data(wave)**2 + veldisp_kin_Angst**2 )

            # Difference between total dispersion and LIS measurement resolution
            FWHM_dif = np.sqrt( config['LS']['CONV_COR']**2 - total_dispersion**2 )

            # Convert resolution difference from Angstrom to pixel
            sigma = (FWHM_dif / wave) * cvel / 2.355 / velscale

            # Flag spectrum if the total intrinsic dispersion is larger than the LIS measurement resolution
            idx = np.where( np.isnan( sigma ) == True )[0]
            if len( idx ) > 0:
                sigma[idx] = 0.0
                totalFWHM_flag[i] = 1 

            # Convolve spectra pixel-wise
            spec[i,:]  = gaussian_filter1d(spec[i,:],  sigma)
            espec[i,:] = gaussian_filter1d(espec[i,:], sigma)
        printStatus.updateDone("Broadening the spectra to LIS resolution", progressbar=True)

    # Get indices that are considered in SSP-conversion
    idx         = np.where( tab['spp'] == 1 )[0]
    index_names = tab['names'][idx].tolist()

    # Loading model predictions
    if MCMC == True: 
        modelfile = os.path.join(config['GENERAL']['TEMPLATE_DIR'],config['LS']['SPP_FILE'])
        model_indices, params, tri, labels = ssppop.load_models(modelfile, index_names)
        logging.info("Loading LS model file at "+modelfile)
    elif MCMC == False: 
        model_indices, params, tri, labels = "dummy", "dummy", "dummy", "dummy"

    # Arrays to store results
    ls_indices = np.zeros((nbins, len(names)))
    ls_errors  = np.zeros((nbins, len(names)))
    if MCMC == True: 
        vals       = np.zeros((nbins, len(labels)*3+2))
        percentile = np.zeros((nbins, 101, len(labels)))

    # Run LS Measurements
    start_time = time.time()
    if config['GENERAL']['PARALLEL'] == True:
        printStatus.running("Running lineStrengths in parallel mode")            
        logging.info("Running lineStrengths in parallel mode")            

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()
    
        # Create worker processes
        ps = [Process(target=workerLS, args=(inQueue, outQueue))
              for _ in range(config['GENERAL']['NCPU'])]
    
        # Start worker processes
        for p in ps: p.start()
    
        # Fill the queue
        for i in range(nbins):
            inQueue.put( ( wave, spec[i,:], espec[i,:], redshift[i,:], config, lickfile, names, index_names,\
                           model_indices, params, tri, labels, nbins, i, MCMC ) )

        # now get the results with indices
        ls_tmp = [outQueue.get() for _ in range(nbins)]
    
        # send stop signal to stop iteration
        for _ in range(config['GENERAL']['NCPU']): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()
    
        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]          = ls_tmp[i][0]
            ls_indices[i,:]   = ls_tmp[i][1]
            ls_errors[i,:]    = ls_tmp[i][2]
            if MCMC == True: 
                vals[i,:]         = ls_tmp[i][3]
                percentile[i,:,:] = ls_tmp[i][4]

        # Sort output
        argidx = np.argsort( index )
        ls_indices = ls_indices[argidx,:]
        ls_errors  = ls_errors[argidx,:]
        if MCMC == True: 
            vals       = vals[argidx,:]
            percentile = percentile[argidx,:,:]

        printStatus.updateDone("Running lineStrengths in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running lineStrengths in serial mode")            
        logging.info("Running lineStrengths in serial mode")            

        if MCMC == True:
            for i in range(nbins):
                ls_indices[i,:], ls_errors[i,:], vals[i,:], percentile[i,:,:] = run_ls\
                        (wave, spec[i,:], espec[i,:], redshift[i,:], config, lickfile, names, index_names,\
                        model_indices, params, tri, labels, nbins, i, MCMC)
        elif MCMC == False: 
            for i in range(nbins):
                ls_indices[i,:], ls_errors[i,:] = run_ls\
                        (wave, spec[i,:], espec[i,:], redshift[i,:], config, lickfile, names, index_names,\
                        model_indices, params, tri, labels, nbins, i, MCMC)


        printStatus.updateDone("Running lineStrengths in serial mode", progressbar=True)

    print("             Running lineStrengths on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))
    logging.info("Running lineStrengths on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.all( np.isnan(ls_indices[:,:]), axis=1 ) == True )[0]
    if len(idx_error) != 0:
        printStatus.warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Save Results
    if MCMC == True: 
        save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, config, vals=vals, percentile=percentile)
    elif MCMC == False: 
        save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, config)

    # Repeat analysis with adapted spectral resolution
    if RESOLUTION == 'ORIGINAL':
        measureLineStrengths(config, RESOLUTION='ADAPTED')

    # Return
    return(None)


