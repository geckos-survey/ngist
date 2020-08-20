from astropy.io import fits
import numpy as np

import os
import logging

from printStatus import printStatus

from ppxf.ppxf_util import log_rebin



def prepSpectra(config, cube):
    """
    This function performs the following tasks: 
     * Apply spatial bins to linear spectra; Save these spectra to disk
     * Log-rebin all spectra, regardless of whether the spaxels are masked or not; Save all spectra to disk
     * Apply spatial bins to log-rebinned spectra; Save these spectra to disk
    """

    # Read maskfile
    maskfile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + "_mask.fits"
    mask = fits.open(maskfile)[1].data.MASK
    idxUnmasked = np.where( mask == 0 )[0]
    idxMasked   = np.where( mask == 1 )[0]

    # Read binning pattern
    tablefile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + "_table.fits"
    binNum = fits.open(tablefile)[1].data.BIN_ID[idxUnmasked]

    # Apply spatial bins to linear spectra
    bin_data, bin_error, bin_flux = applySpatialBins(binNum, cube['spec'][:,idxUnmasked], cube['error'][:,idxUnmasked], config['PREPARE_SPECTRA']['VELSCALE'], "lin" )
    # Save spatially binned spectra
    saveBinSpectra(config, bin_data, bin_error, config['PREPARE_SPECTRA']['VELSCALE'], cube['wave'], "lin")

    # Log-rebin spectra
    log_spec, log_error, logLam = log_rebinning(config, cube)
    # Save all log-rebinned spectra
    saveAllSpectra(config, log_spec, log_error, config['PREPARE_SPECTRA']['VELSCALE'], logLam)

    # Apply bins to log spectra
    bin_data, bin_error, bin_flux = applySpatialBins(binNum, log_spec[:,idxUnmasked], log_error[:,idxUnmasked], config['PREPARE_SPECTRA']['VELSCALE'], "log" )
    # Save spatially binned spectra
    saveBinSpectra(config, bin_data, bin_error, config['PREPARE_SPECTRA']['VELSCALE'], logLam, "log")

    return(None)


def log_rebinning(config, cube):
    """
    Logarithmically rebin spectra and error spectra. 
    """
    # Log-rebin the spectra
    printStatus.running("Log-rebinning the spectra")
    log_spec, logLam = run_logrebinning\
            (cube['spec'], config['PREPARE_SPECTRA']['VELSCALE'], len(cube['x']), cube['wave'] )
    printStatus.updateDone("Log-rebinning the spectra", progressbar=True)
    logging.info("Log-rebinned the spectra")

    # Log-rebin the error spectra
    printStatus.running("Log-rebinning the error spectra")
    log_error, _ = run_logrebinning\
            (cube['error'], config['PREPARE_SPECTRA']['VELSCALE'], len(cube['x']), cube['wave'] )
    printStatus.updateDone("Log-rebinning the error spectra", progressbar=True)
    logging.info("Log-rebinned the error spectra")
    
    return(log_spec, log_error, logLam)


def run_logrebinning( bin_data, velscale, nbins, wave ):
    """
    Calls the log-rebinning routine of pPXF (see Cappellari & Emsellem 2004;
    ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
    """
    # Setup arrays
    lamRange = np.array([np.amin(wave),np.amax(wave)])
    sspNew, logLam, _ = log_rebin(lamRange, bin_data[:,0], velscale=velscale)
    log_bin_data = np.zeros([len(logLam),nbins])

    # Do log-rebinning 
    for i in range(0, nbins):
        log_bin_data[:,i] = corefunc_logrebin(lamRange, bin_data[:,i], velscale, len(logLam), i, nbins)

    return(log_bin_data, logLam)


def corefunc_logrebin(lamRange, bin_data, velscale, npix, iterate, nbins):
    """
    Calls the log-rebinning routine of pPXF (see Cappellari & Emsellem 2004;
    ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C). 

    TODO: Should probably be merged with run_logrebinning. 
    """
    try:
        sspNew, logLam, _ = log_rebin(lamRange, bin_data, velscale=velscale)
        printStatus.progressBar(iterate+1, nbins, barLength = 50)
        return(sspNew)

    except:
        out = np.zeros(npix); out[:] = np.nan
        return(out)


def saveAllSpectra(config, log_spec, log_error, velscale, logLam):
    """ Save all logarithmically rebinned spectra to file. """
    outfits_spectra = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + '_AllSpectra.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_AllSpectra.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append( fits.Column(name='SPEC',   format=str(len(log_spec))+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC',  format=str(len(log_spec))+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'
    
    # Create HDU List and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header keywords
    fits.setval(outfits_spectra,'VELSCALE', value=velscale)
    fits.setval(outfits_spectra,'CRPIX1',   value=1.0)
    fits.setval(outfits_spectra,'CRVAL1',   value=logLam[0])
    fits.setval(outfits_spectra,'CDELT1',   value=logLam[1]-logLam[0])

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_AllSpectra.fits')
    logging.info("Wrote: "+outfits_spectra)


def saveBinSpectra(config, log_spec, log_error, velscale, logLam, flag):
    """ Save spatially binned spectra and error spectra are saved to disk. """
    outfile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'])

    if flag == 'log':
        outfits_spectra = outfile+'_BinSpectra.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra.fits')
    elif flag == 'lin':
        outfits_spectra  = outfile+'_BinSpectra_linear.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra_linear.fits')

    npix = len(log_spec)

    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BIN_SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'

    # Create HDU list and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    fits.setval(outfits_spectra,'VELSCALE',value=velscale)
    fits.setval(outfits_spectra,'CRPIX1',  value=1.0)
    fits.setval(outfits_spectra,'CRVAL1',  value=logLam[0])
    fits.setval(outfits_spectra,'CDELT1',  value=logLam[1]-logLam[0])

    if flag == 'log':
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra.fits')
    elif flag == 'lin':
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra_linear.fits')
    logging.info("Wrote: "+outfits_spectra)


def applySpatialBins(binNum, spec, espec, velscale, flag):
    """
    The constructed spatial binning scheme is applied to the spectra.     
    """
    printStatus.running("Applying the spatial bins to "+flag+"-data")
    bin_data, bin_error, bin_flux = spatialBinning( binNum, spec, espec )
    printStatus.updateDone("Applying the spatial bins to "+flag+"-data", progressbar=True)
    logging.info("Applied spatial bins to "+flag+"-data")

    return(bin_data, bin_error, bin_flux)


def spatialBinning( binNum, spec, error ):
    """ Spectra belonging to the same spatial bin are added. """
    ubins     = np.unique(binNum)
    nbins     = len(ubins)
    npix      = spec.shape[0]
    bin_data  = np.zeros([npix,nbins])
    bin_error = np.zeros([npix,nbins])
    bin_flux  = np.zeros(nbins)

    for i in range(nbins):
        k = np.where( binNum == ubins[i] )[0]
        valbin = len(k)
        if valbin == 1:
           av_spec     = spec[:,k]
           av_err_spec = np.sqrt(error[:,k])
        else:
           av_spec     = np.nansum(spec[:,k],axis=1)
           av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
    
        bin_data[:,i]  = np.ravel(av_spec)
        bin_error[:,i] = np.ravel(av_err_spec)
        bin_flux[i]    = np.mean(av_spec,axis=0)
        printStatus.progressBar(i+1, nbins, barLength = 50)

    return(bin_data, bin_error, bin_flux)


