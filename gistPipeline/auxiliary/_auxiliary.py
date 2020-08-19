from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
import sys
import os
import glob

from gistPipeline._version import __version__



"""
This file contains a selection of functions that are needed at multiple locations in the framework. This includes
functions to print the status of the GIST to stdout, read the line-spread-function from file, and create spectral masks. 

When developing user-defined modules, you can take advantage of these functions or simply include your own tools in the
module. 
"""



def getLSF(config):
    """
    Function to read the given LSF's from file. 
    """
    # Read LSF of observation and templates and construct an interpolation function
    lsfDataFile = os.path.join(config['GENERAL']['CONFIG_DIR'], config['GENERAL']['LSF_DATA'])
    lsfTempFile = os.path.join(config['GENERAL']['CONFIG_DIR'], config['GENERAL']['LSF_TEMP'])
    LSF           = np.genfromtxt(lsfDataFile, comments='#')
    LSF[:,0]      = LSF[:,0] / (1 + config['GENERAL']['REDSHIFT'])
    LSF[:,1]      = LSF[:,1] / (1 + config['GENERAL']['REDSHIFT'])
    LSF_Data      = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')
    LSF           = np.genfromtxt(lsfTempFile, comments='#')
    LSF_Templates = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')
    return(LSF_Data, LSF_Templates)


def spectralMasking(config, file, logLam):
    """ Mask spectral region in the fit. """
    # Read file
    mask        = np.genfromtxt(os.path.join(config['GENERAL']['CONFIG_DIR'],file), usecols=(0,1))
    maskComment = np.genfromtxt(os.path.join(config['GENERAL']['CONFIG_DIR'],file), usecols=(2), dtype=str )
    goodPixels  = np.arange( len(logLam) )

    # In case there is only one mask
    if len( mask.shape ) == 1  and  mask.shape[0] != 0:
        mask        = mask.reshape(1,2)
        maskComment = maskComment.reshape(1)

    for i in range( mask.shape[0] ):

        # Check for sky-lines
        if maskComment[i] == 'sky'  or  maskComment[i] == 'SKY'  or  maskComment[i] == 'Sky': 
            mask[i,0] = mask[i,0] / (1+config['GENERAL']['REDSHIFT'])

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


def addGISTHeaderComment(config):
    """
    Add a GIST header comment in all fits output files. 
    """
    filelist = glob.glob(os.path.join(config['GENERAL']['OUTPUT'],"*.fits")) 

    for file in filelist:
        hdu = fits.open(file)
        for o in range(len(hdu)):
            if "Generated with the GIST pipeline" not in str(hdu[o].header): 

                fits.setval(file, 'COMMENT', value="" , ext=o)
                fits.setval(file, 'COMMENT', value="------------------------------------------------------------------------", ext=o)
                fits.setval(file, 'COMMENT', value="                Generated with the GIST pipeline, V"+__version__+"                  ", ext=o)
                fits.setval(file, 'COMMENT', value="------------------------------------------------------------------------", ext=o)
                fits.setval(file, 'COMMENT', value=" Please cite Bittner et al. 2019 (A&A, 628, A117) and the corresponding ", ext=o)
                fits.setval(file, 'COMMENT', value="       analysis routines if you use this data in any publication.       ", ext=o)
                fits.setval(file, 'COMMENT', value=""                                                                        , ext=o)
                fits.setval(file, 'COMMENT', value="         For a thorough documentation of this software package,         ", ext=o)
                fits.setval(file, 'COMMENT', value="         please see https://abittner.gitlab.io/thegistpipeline          ", ext=o)
                fits.setval(file, 'COMMENT', value="------------------------------------------------------------------------", ext=o)
                fits.setval(file, 'COMMENT', value="", ext=o)

    return(None)


def saveConfigToHeader(hdu, config):
    """
    Save the used section of the MasterConfig file to the header of the output data.
    """
    for i in config.keys():
        hdu.header[i] = config[i]
    return(hdu)


