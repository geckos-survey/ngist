from   astropy.io          import fits
import numpy               as np

import os
import logging

from printStatus import printStatus

from gistPipeline.readData    import der_snr as der_snr



# ======================================
# Routine to load spectra from plain txt
# ======================================
def readCube(config):

    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "

    # Read spectrum
    printStatus.running("Reading the spectrum")
    logging.info("Reading the spectrum: "+config['GENERAL']['INPUT'])

    # Reading the cube
    data  = np.genfromtxt(config['GENERAL']['INPUT'])
    spec  = np.zeros((data.shape[0],1))
    espec = np.zeros((data.shape[0],1))
    wave  = data[:,0]
    spec[:,0] = data[:,1]
    if data.shape[1] == 3: 
        espec[:,0] = data[:,2]
    else: 
        espec[:,0] = np.ones(data.shape[0])

    # Getting the spatial coordinates
    x         = np.zeros(1)
    y         = np.zeros(1)
    pixelsize = 1.0

    # De-redshift spectra
    wave = wave / (1+config['GENERAL']['REDSHIFT'])
    logging.info("Shifting spectra to rest-frame, assuming a redshift of "+str(config['GENERAL']['REDSHIFT']))

    # Shorten spectra to required wavelength range
    lmin  = config['READ_DATA']['LMIN_TOT']
    lmax  = config['READ_DATA']['LMAX_TOT']
    idx   = np.where( np.logical_and( wave >= lmin, wave <= lmax ) )[0]
    spec  = spec[idx]
    espec = espec[idx]
    wave  = wave[idx]
    logging.info("Shortening spectra to the wavelength range from "+str(config['READ_DATA']['LMAX_TOT'])+"A to "+str(config['READ_DATA']['LMAX_TOT'])+"A.")

    # Computing the SNR per spaxel
    idx_snr = np.where( np.logical_and( wave >= config['READ_DATA']['LMIN_SNR'], wave <= config['READ_DATA']['LMAX_SNR'] ) )[0]
    signal  = np.zeros(1) + np.nanmedian(spec[idx_snr],axis=0)
    noise   = np.zeros(1) + np.abs(np.nanmedian(np.sqrt(espec[idx_snr]),axis=0))
    snr     = signal / noise
    logging.info("Computing the signal-to-noise ratio in the wavelength range from "+str(config['READ_DATA']['LMAX_SNR'])+"A to "+str(config['READ_DATA']['LMAX_SNR'])+"A.")

    # Storing everything into a structure
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'error':espec, 'snr':snr, 'signal':signal, 'noise':noise, 'pixelsize':pixelsize}

    printStatus.updateDone("Reading the spectrum")
    logging.info("Finished reading the spectrum!")

    return(cube)
