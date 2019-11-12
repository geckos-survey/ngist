from   astropy.io          import fits
import numpy               as np

import os
import logging

from gistPipeline.gistModules import util    as pipeline
from gistPipeline.readData    import der_snr as der_snr



# ======================================
# Routine to load spectra from plain txt
# ======================================
def read_cube(DEBUG, filename, configs):

    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "

    directory = os.path.dirname(filename)+'/'
    datafile  = os.path.basename(filename)
    rootname  = datafile.split('.')[0]

    # Read MUSE-cube
    pipeline.prettyOutput_Running("Reading the file")
    logging.info("Reading the file: "+filename)

    # Reading the cube
    data  = np.genfromtxt(filename)
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
    wave = wave / (1+configs['REDSHIFT'])

    # Shorten spectra to required wavelength range
    lmin  = np.min([configs['LMIN_SNR'], configs['LMIN_PPXF'], configs['LMIN_GANDALF'], configs['LMIN_SFH']])
    lmax  = np.max([configs['LMAX_SNR'], configs['LMAX_PPXF'], configs['LMAX_GANDALF'], configs['LMAX_SFH']])
    idx   = np.where( np.logical_and( wave >= lmin, wave <= lmax ) )[0]
    spec  = spec[idx]
    espec = espec[idx]
    wave  = wave[idx]

    # Computing the SNR per spaxel
    idx_snr = np.where( np.logical_and( wave >= configs['LMIN_SNR'], wave <= configs['LMAX_SNR'] ) )[0]
    signal  = np.zeros(1) + np.nanmedian(spec[idx_snr],axis=0)
    noise   = np.zeros(1) + np.abs(np.nanmedian(np.sqrt(espec[idx_snr]),axis=0))
    snr     = signal / noise
    logging.info("Computing the signal-to-noise ratio per spaxel.")

    # Shorten spectra to chosen wavelength range
    cvel      = 299792.458
    velscale  = (wave[1]-wave[0])*cvel/np.mean(wave)
    logging.info("Extracting spectral information:\n"\
            +loggingBlanks+"* Shortened spectra to wavelength range from "+str(lmin)+" to "+str(lmax)+" Angst.\n"\
            +loggingBlanks+"* Spectral pixelsize in velocity space is "+str(velscale)+" km/s")

    # Storing everything into a structure
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'error':espec, 'snr':snr,\
            'signal':signal, 'noise':noise, 'velscale':velscale, 'pixelsize':pixelsize}

    pipeline.prettyOutput_Done("Reading the file")
    logging.info("Finished reading the file!")

    return(cube)
