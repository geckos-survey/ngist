from   astropy.io          import fits
import numpy               as np

import os
import logging

from gistPipeline.gistModules import util    as pipeline



# ======================================
# Routine to set DEBUG mode
# ======================================
def set_debug(cube, xext, yext):
    cube['x']      = cube['x'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['y']      = cube['y'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['snr']    = cube['snr'   ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['signal'] = cube['signal'][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['noise']  = cube['noise' ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    
    cube['spec']   = cube['spec'  ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['error']  = cube['error' ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]

    return(cube)


# ======================================
# Routine to load CALIFA-cubes
# ======================================
def read_cube(DEBUG, filename, configs):

    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "

    directory = os.path.dirname(filename)+'/'
    datafile  = os.path.basename(filename)
    rootname  = datafile.split('.')[0]

    # Read CALIFA-cube
    pipeline.prettyOutput_Running("Reading the CALIFA V1200 cube")
    logging.info("Reading the CALIFA V1200 cube"+filename)

    # Reading the cube
    hdu   = fits.open(filename)
    hdr   = hdu[0].header
    data  = hdu[0].data
    s     = np.shape(data)
    spec  = np.reshape(data,[s[0],s[1]*s[2]])

    # Read the error spectra
    logging.info("Reading the error spectra from the cube")
    stat  = hdu[1].data
    espec = np.reshape(stat,[s[0],s[1]*s[2]])

    # Getting the wavelength info
    wave = hdr['CRVAL3']+(np.arange(s[0]))*hdr['CDELT3']
    
    # Getting the spatial coordinates
    xaxis = (np.arange(s[2]) - configs['ORIGIN'][0]) * hdr['CD2_2']*3600.0
    yaxis = (np.arange(s[1]) - configs['ORIGIN'][1]) * hdr['CD2_2']*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x     = np.reshape(x,[s[1]*s[2]])
    y     = np.reshape(y,[s[1]*s[2]])
    pixelsize = hdr['CD2_2']*3600.0

    logging.info("Extracting spatial information:\n"\
            +loggingBlanks+"* Spatial coordinates are centred to "+str(configs['ORIGIN'])+"\n"\
            +loggingBlanks+"* Spatial pixelsize is "+str(pixelsize))
  
    # Getting the wavelength information
    cvel      = 299792.458
    wave      = wave / (1+configs['REDSHIFT'])
    idx       = np.where( np.logical_and( wave >= configs['LMIN'], wave <= configs['LMAX'] ) )[0]
    spec      = spec[idx,:]
    espec     = espec[idx,:]
    wave      = wave[idx]
    velscale  = (wave[1]-wave[0])*cvel/np.mean(wave)
    xy_extent = np.array( [data.shape[2], data.shape[1]] )
    logging.info("Extracting spectral information:\n"\
            +loggingBlanks+"* Shortened spectra to wavelength range from "+str(configs['LMIN'])+" to "+str(configs['LMAX'])+" Angst.\n"\
            +loggingBlanks+"* Spectral pixelsize in velocity space is "+str(velscale)+" km/s")

    # Removing obviously defective pixels: Remove spaxel with any nan or negative values
    nspec    = np.shape(spec)[1]
    idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]
    spec     = spec[:,idx_good]
    espec    = espec[:,idx_good]
    x        = x[idx_good]
    y        = y[idx_good]
    logging.info("Removing all spaxels containing nan or having a negative median flux:\n"\
            +loggingBlanks+"* Of "+str(nspec)+" in the cube, "+str(len(idx_good))+" are accepted and "+str(nspec-len(idx_good))+" removed")

    # Computing the SNR per spaxel
    signal = np.nanmedian(spec,axis=0)
    noise  = np.abs(np.nanmedian(espec,axis=0))
    snr    = signal / noise
    logging.info("Computing the signal-to-noise ratio per spaxel.")

    # Storing eveything into a structure
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'error':espec, 'snr':snr,\
            'signal':signal, 'noise':noise, 'velscale':velscale, 'pixelsize':pixelsize}

    # Constrain cube to one central row if switch DEBUG is set
    if DEBUG == True: cube = set_debug(cube, xy_extent[0], xy_extent[1])

    pipeline.prettyOutput_Done("Reading the CALIFA V1200 cube")
    print("             Read "+str(len(cube['x']))+" spectra!")
    logging.info("Finished reading the CALIFA V1200 cube!")

    return(cube)
