from   astropy.io import fits
import numpy      as np

import glob
import os
import logging

from gistPipeline.gistModules          import util         as pipeline
from gistPipeline.sitePackages.gandalf import gandalf_util as gandalf

try:
    # Try to use local version in sitePackages
    from gistPipeline.sitePackages.ppxf.ppxf_util import log_rebin, gaussian_filter1d
except: 
    # Then use system installed version instead
    from ppxf.ppxf_util import log_rebin, gaussian_filter1d


"""
PURPOSE:
  This file contains a collection of functions necessary to prepare the input 
  data for the following analysis steps. 
"""


def apply_snr_threshold(snr, signal, min_snr):
    """ 
    Select those spaxels that are above the isophote level with a mean 
    signal-to-noise ratio of MIN_SNR. 
    """
    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "
    pipeline.prettyOutput_Running("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr))

    idx_snr = np.where( np.abs(snr - min_snr) < 2. )[0]
    meanmin_signal = np.mean( signal[idx_snr] )
    idx_inside  = np.where( signal >= meanmin_signal )[0]
    idx_outside = np.where( signal < meanmin_signal )[0]

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange( len(snr) )
        idx_outside = np.array([], dtype=np.int64)
    
    pipeline.prettyOutput_Done("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr))
    logging.info("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr)+"\n"\
            +loggingBlanks+"Selected "+str(len(idx_inside))+" spaxels inside and "+str(len(idx_outside))+" outside of the Voronoi region.")

    return(idx_inside, idx_outside)


def log_rebinning(cube, configs, rootname, outdir):
    """
    Logarithmically rebin spectra and error spectra. Save the resulting
    spectra to disk or, in case these are already available, load the spectra
    from disk and skip the rebinning process. 
    """
    # Do log-rebinning and save spectra
    if os.path.isfile(outdir+rootname+'_AllSpectra.fits') == False:

        # Do log-rebinning for spectra
        pipeline.prettyOutput_Running("Log-rebinning the spectra")
        log_spec, logLam = run_logrebinning\
                (cube['spec'], cube['velscale'], len(cube['x']), cube['wave'], configs )
        pipeline.prettyOutput_Done("Log-rebinning the spectra", progressbar=True)
        logging.info("Log-rebinned the spectra")

        # Do log-rebinning for error spectra
        pipeline.prettyOutput_Running("Log-rebinning the error spectra")
        log_error, _ = run_logrebinning\
                (cube['error'], cube['velscale'], len(cube['x']), cube['wave'], configs )
        pipeline.prettyOutput_Done("Log-rebinning the error spectra", progressbar=True)
        logging.info("Log-rebinned the error spectra")
    
        # Save all spectra
        saveAllSpectra(rootname, outdir, log_spec, log_error, cube['velscale'], logLam)
    else:
        # Load all spectra
        log_spec, log_error, logLam = loadAllSpectra(rootname, outdir)

    return(log_spec, log_error, logLam)


def run_logrebinning( bin_data, velscale, nbins, wave, configs ):
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
        pipeline.printProgress(iterate+1, nbins, barLength = 50)
        return(sspNew)

    except:
        out = np.zeros(npix); out[:] = np.nan
        return(out)


def saveAllSpectra(rootname, outdir, log_spec, log_error, velscale, logLam):
    """ Save all logarithmically rebinned spectra to file. """
    outfits_spectra  = outdir+rootname+'_AllSpectra.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_AllSpectra.fits')

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

    pipeline.prettyOutput_Done("Writing: "+rootname+'_AllSpectra.fits')
    logging.info("Wrote: "+outfits_spectra)


def loadAllSpectra(rootname, outdir):
    """ Loads all spectra from file. """
    hdu = fits.open(outdir+rootname+'_AllSpectra.fits')
    log_spec  = np.array( hdu[1].data.SPEC.T  )
    log_error = np.array( hdu[1].data.ESPEC.T )
    logLam    = np.array( hdu[2].data.LOGLAM  )
    return(log_spec, log_error, logLam)


def prepare_sp_templates(configs, velscale, velscale_ratio, LSF_Data, LSF_Templates):
    """
    Prepares the spectral template library. The templates are loaded from disk,
    shortened to meet the spectral range in consideration, convolved to meet the
    resolution of the observed spectra (according to the LSF), log-rebinned, and
    normalised. 
    """
    pipeline.prettyOutput_Running("Preparing the stellar population templates")
    cvel  = 299792.458

    # SSP model library
    sp_models = glob.glob(configs['SSP_LIB']+'*.fits')
    
    # Read data
    hdu_spmod      = fits.open(sp_models[0])
    ssp_data       = hdu_spmod[0].data
    ssp_head       = hdu_spmod[0].header
    lamRange_spmod = ssp_head['CRVAL1'] + np.array([0., ssp_head['CDELT1']*(ssp_head['NAXIS1'] - 1)])

    # Determine length of templates
    template_overhead = np.zeros(2)
    if configs['LMIN'] - lamRange_spmod[0] > 150.:
        template_overhead[0] = 150.
    else: 
        template_overhead[0] = configs['LMIN'] - lamRange_spmod[0] - 5
    if lamRange_spmod[1] - configs['LMAX'] > 150.:
        template_overhead[1] = 150.
    else: 
        template_overhead[1] = lamRange_spmod[1] - configs['LMAX'] - 5

    # Shorten templates to size of data
    # Reconstruct full original lamRange
    lamRange_lin = np.arange( lamRange_spmod[0], lamRange_spmod[-1]+ssp_head['CDELT1'], ssp_head['CDELT1'] )
    # Create new lamRange according to LMIN and LMAX from config-file
    constr = np.array([ configs['LMIN'] - template_overhead[0], configs['LMAX'] + template_overhead[1] ])
    idx_lam = np.where( np.logical_and(lamRange_lin > constr[0], lamRange_lin < constr[1] ) )[0]
    lamRange_spmod = np.array([ lamRange_lin[idx_lam[0]], lamRange_lin[idx_lam[-1]] ])
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
 
    # Create an array to store the templates
    sspNew, logLam_spmod, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale/velscale_ratio)
    templates = np.empty((sspNew.size, len(sp_models)))
   
    # Load templates, convolve and log-rebin them
    for j, file in enumerate(sp_models):
        hdu      = fits.open(file)
        ssp_data = hdu[0].data[idx_lam]
        ssp_data = gaussian_filter1d(ssp_data, sigma)
        templates[:, j], logLam_spmod, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale/velscale_ratio)
   
    # Normalise templates
    templates = normalise_templates(templates, configs['NORM_TEMP'])

    pipeline.prettyOutput_Done("Preparing the stellar population templates")
    logging.info("Prepared the stellar population templates")

    return( templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam_spmod )


def normalise_templates(templates, NORM_TEMP):
    """ 
    Normalise the spectral template library to obtain mass or light-weighted
    results. 
    """
    # Normalise templates in such a way to get mass-weighted results
    if NORM_TEMP == 'MASS':
        templates = templates / np.mean( templates )

    # Normalise templates in such a way to get light-weighted results
    if NORM_TEMP == 'LIGHT':
        for i in range( templates.shape[1] ):
            templates[:,i] = templates[:,i] / np.mean(templates[:,i], axis=0)

    return( templates )


def spectralMasking(outdir, logLam, module):
    """
    Construct a spectral mask, according to the information provided in the file
    spectralMasking_[module].config. Note that this is not considered in the
    emission-line analysis with GandALF, as GandALF uses its own, specific
    emission-line setup file. 
    """

    # Read file
    mask       = np.genfromtxt(outdir+"spectralMasking_"+module+".config", usecols=(0,1))
    goodPixels = np.arange( len(logLam) )

    # In case there is only one mask
    if len( mask.shape ) == 1  and  mask.shape[0] != 0:
        mask = mask.reshape(1,2)

    for i in range( mask.shape[0] ):

        # Define masked pixel range
        minimumPixel = int( np.round( ( np.log( mask[i,0] - mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )
        maximumPixel = int( np.round( ( np.log( mask[i,0] + mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )

        # Handle border of wavelength range
        if minimumPixel < 0:            minimumPixel = 0
        if maximumPixel < 0:            maximumPixel = 0 
        if minimumPixel >= len(logLam): minimumPixel = len(logLam)-1
        if maximumPixel >= len(logLam): maximumPixel = len(logLam)-1

        # Mark masked spectral pixels
        goodPixels[minimumPixel:maximumPixel+1] = -1

    goodPixels = goodPixels[ np.where( goodPixels != -1 )[0] ]

    return(goodPixels)
