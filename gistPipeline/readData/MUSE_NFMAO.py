from   astropy.io          import fits
import numpy               as np

import os
import logging

from gistPipeline.gistModules import util    as pipeline
from gistPipeline.readData    import der_snr as der_snr



# ======================================
# Routine to set DEBUG mode
# ======================================
def set_debug(cube, xext, yext):
    logging.info("DEBUG mode is activated. Instead of the entire cube, only one line of spaxels is used.")
    cube['x']      = cube['x'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['y']      = cube['y'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['snr']    = cube['snr'   ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['signal'] = cube['signal'][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['noise']  = cube['noise' ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    
    cube['spec']   = cube['spec'  ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['error']  = cube['error' ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]

    return(cube)


# ======================================
# Routine to load MUSE-cubes
# ======================================
def read_cube(DEBUG, filename, configs):

    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "

    directory = os.path.dirname(filename)+'/'
    datafile  = os.path.basename(filename)
    rootname  = datafile.split('.')[0]

    # Read MUSE-cube
    pipeline.prettyOutput_Running("Reading the MUSE-NFM cube")
    logging.info("Reading the MUSE-NFM cube: "+filename)

    # Reading the cube
    hdu   = fits.open(filename)
    hdr   = hdu[1].header
    data  = hdu[1].data
    s     = np.shape(data)
    spec  = np.reshape(data,[s[0],s[1]*s[2]])

    # Read the error spectra if available. Otherwise estimate the errors with the der_snr algorithm
    if len(hdu) == 3:
        logging.info("Reading the error spectra from the cube")
        stat  = hdu[2].data
        espec = np.reshape(stat,[s[0],s[1]*s[2]])
    elif len(hdu) == 2:
        logging.info("No error extension found. Estimating the error spectra with the der_snr algorithm")
        espec = np.zeros( spec.shape )
        for i in range( 0, spec.shape[1] ):
            espec[:,i] = der_snr.der_snr( spec[:,i] )

    # Getting the wavelength info
    wave = hdr['CRVAL3']+(np.arange(s[0]))*hdr['CD3_3']
    
    # Getting the spatial coordinates
    xaxis = (np.arange(s[2]) - configs['ORIGIN'][0]) * hdr['CD2_2']*3600.0
    yaxis = (np.arange(s[1]) - configs['ORIGIN'][1]) * hdr['CD2_2']*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x     = np.reshape(x,[s[1]*s[2]])
    y     = np.reshape(y,[s[1]*s[2]])
    pixelsize = 0.025

    logging.info("Extracting spatial information:\n"\
            +loggingBlanks+"* Spatial coordinates are centred to "+str(configs['ORIGIN'])+"\n"\
            +loggingBlanks+"* Spatial pixelsize is "+str(pixelsize))
  
    # De-redshift spectra
    wave = wave / (1+configs['REDSHIFT'])

    # Shorten spectra to required wavelength range
    lmin  = np.min([configs['LMIN_SNR'], configs['LMIN_PPXF'], configs['LMIN_GANDALF'], configs['LMIN_SFH']])
    lmax  = np.max([configs['LMAX_SNR'], configs['LMAX_PPXF'], configs['LMAX_GANDALF'], configs['LMAX_SFH']])
    idx   = np.where( np.logical_and( wave >= lmin, wave <= lmax ) )[0]
    spec  = spec[idx,:]
    espec = espec[idx,:]
    wave  = wave[idx]

    # Computing the SNR per spaxel
    idx_snr   = np.where( np.logical_and.reduce([ \
        wave >= configs['LMIN_SNR'], \
        wave <= configs['LMAX_SNR'], \
        np.logical_or( wave < 5780/(1+configs['REDSHIFT']), wave > 6048/(1+configs['REDSHIFT'])) ]))[0]
    signal  = np.nanmedian(spec[idx_snr,:],axis=0)
    if len(hdu) == 3:
        noise  = np.abs(np.nanmedian(np.sqrt(espec[idx_snr,:]),axis=0))
    elif len(hdu) == 2:
        noise = espec[0,:]    # DER_SNR returns constant error spectra
    snr    = signal / noise
    logging.info("Computing the signal-to-noise ratio per spaxel.")

    # Determine velscale
    cvel      = 299792.458
    velscale  = (wave[1]-wave[0])*cvel/np.mean(wave)
    logging.info("Extracting spectral information:\n"\
            +loggingBlanks+"* Shortened spectra to wavelength range from "+str(lmin)+" to "+str(lmax)+" Angst.\n"\
            +loggingBlanks+"* Spectral pixelsize in velocity space is "+str(velscale)+" km/s")

    # Replacing the np.nan in the laser region by the median of the spectrum
    idx_laser          = np.where( np.logical_and( wave > 5780 / (1+configs['REDSHIFT']), wave < 6048 / (1+configs['REDSHIFT'])) )[0]
    spec[idx_laser,:]  = signal
    espec[idx_laser,:] = noise

    # Storing everything into a structure
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'error':espec, 'snr':snr,\
            'signal':signal, 'noise':noise, 'velscale':velscale, 'pixelsize':pixelsize}

    # Constrain cube to one central row if switch DEBUG is set
    if DEBUG == True: cube = set_debug(cube, s[2], s[1])

    pipeline.prettyOutput_Done("Reading the MUSE-NFM cube")
    print("             Read "+str(len(cube['x']))+" spectra!")
    logging.info("Finished reading the MUSE cube!")

    return(cube)
