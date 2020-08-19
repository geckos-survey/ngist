import os
import logging
import numpy as np
from astropy.io import fits

from printStatus import printStatus



def generateSpatialMask(config, cube):
    """ 
    Default implementation of the spatialMasking module. 

    This function masks defunct spaxels, rejects spaxels with a signal-to-noise ration below a given threshold, and
    masks spaxels according to a provided mask file. Finally, all masks are combined and saved. 
    """

    # Mask defunct spaxels
    maskedDefunct = maskDefunctSpaxels(cube)

    # Mask spaxels with SNR below threshold
    maskedSNR = applySNRThreshold(cube['snr'], cube['signal'], config['SPATIAL_MASKING']['MIN_SNR'])

    # Mask spaxels according to spatial mask file
    maskedMask = applyMaskFile(config, cube)

    # Create combined mask
    combinedMaskIdx = np.where( np.logical_or.reduce((maskedDefunct == True, maskedSNR == True, maskedMask == True)) )[0]
    combinedMask = np.zeros(len(cube['snr']), dtype=bool)
    combinedMask[combinedMaskIdx] = True
    logging.info("Combined mask: "+str(len(combinedMaskIdx))+" spaxels are rejected.")
   
    # Save mask to file
    saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config)

    # Return
    return(None)


def maskDefunctSpaxels(cube):
    """
    Mask defunct spaxels, in particular those containing np.nan's or have a 
    negative median. 
    """
    # Select defunct spaxels
    idx_good = np.where( np.logical_and( np.all(np.isnan(cube['spec']) == False, axis=0), np.nanmedian(cube['spec'], axis=0) >  0.0 ))[0]
    idx_bad  = np.where( np.logical_or(  np.any(np.isnan(cube['spec']) == True,  axis=0), np.nanmedian(cube['spec'], axis=0) <= 0.0 ))[0]
    
    logging.info("Masking defunct spaxels: "+str(len(idx_bad))+" spaxels are rejected.")

    masked = np.zeros(len(cube['snr']), dtype=bool)
    masked[idx_good] = False
    masked[idx_bad]  = True
    
    return(masked)


def applySNRThreshold(snr, signal, min_snr):
    """ 
    Mask those spaxels that are above the isophote level with a mean 
    signal-to-noise ratio of MIN_SNR. 
    """
    idx_snr = np.where( np.abs(snr - min_snr) < 2. )[0]
    meanmin_signal = np.mean( signal[idx_snr] )
    idx_inside  = np.where( signal >= meanmin_signal )[0]
    idx_outside = np.where( signal < meanmin_signal )[0]

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange( len(snr) )
        idx_outside = np.array([], dtype=np.int64)
    
    logging.info("Masking low signal-to-noise spaxels: "+str(len(idx_outside))+" spaxels are rejected.")

    masked = np.zeros(len(snr), dtype=bool)
    masked[idx_inside]  = False
    masked[idx_outside] = True

    return(masked)


def applyMaskFile(config, cube):
    """
    Select those spaxels that are unmasked in the input masking file.
    """

    if config['SPATIAL_MASKING']['MASK'] == False  or  config['SPATIAL_MASKING']['MASK'] == None:
        logging.info("No maskfile specified.")
        idxGood = np.arange( len(cube['snr']) )
        idxBad  = np.array([], dtype=np.int64)

    else:
        maskfile = os.path.join(os.path.dirname(config['GENERAL']['INPUT']), config['SPATIAL_MASKING']['MASK'])
    
        if os.path.isfile(maskfile) == True:
            mask = fits.open(maskfile)[1].data
            s = np.shape(mask)
            mask = np.reshape(mask, s[0]*s[1])
        
            idxGood = np.where( mask == 0 )[0]
            idxBad  = np.where( mask == 1 )[0]
    
            logging.info("Masking spaxels according to maskfile: "+str(len(idxBad))+" spaxels are rejected.")
    
        elif os.path.isfile(maskfile) == False: 
            logging.info("No maskfile found at "+maskfile)
            idxGood = np.arange( len(cube['snr']) )
            idxBad  = np.array([], dtype=np.int64)

    masked = np.zeros(len(cube['snr']), dtype=bool)
    masked[idxGood] = False
    masked[idxBad]  = True

    return(masked)


def saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config): 
    """ Save the mask to disk. """
    outfits = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID']) + "_mask.fits"
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+"_mask.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with output data
    # This is an integer array! 0 means unmasked, 1 means masked!
    cols = []
    cols.append(fits.Column(name='MASK',         format='I', array=np.array(combinedMask, dtype=np.int)  ))
    cols.append(fits.Column(name='MASK_DEFUNCT', format='I', array=np.array(maskedDefunct, dtype=np.int) ))
    cols.append(fits.Column(name='MASK_SNR',     format='I', array=np.array(maskedSNR, dtype=np.int)     ))
    cols.append(fits.Column(name='MASK_FILE',    format='I', array=np.array(maskedMask, dtype=np.int)    ))
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "MASKFILE"

    # Create HDU list and write to file
    tbhdu.header['COMMENT'] = "Value 0  -->  unmasked"
    tbhdu.header['COMMENT'] = "Value 1  -->  masked"
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+"_mask.fits")
    logging.info("Wrote mask file: "+outfits)

    return(None)


