#!/usr/bin/env python

from   astropy.io import fits
import numpy      as np
import os
import optparse
import warnings
warnings.filterwarnings('ignore')


def savefitsmaps(flag, outdir):

    runname  = outdir
    rootname = outdir.rstrip('/').split('/')[-1]

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir,rootname)+'_table.fits')
    idx_inside  = np.where( table_hdu[1].data.BIN_ID >= 0        )[0]
    X           = np.array( table_hdu[1].data.X    ) * -1
    Y           = np.array( table_hdu[1].data.Y      )
    FLUX        = np.array( table_hdu[1].data.FLUX   )
    binNum_long = np.array( table_hdu[1].data.BIN_ID )
    ubins       = np.unique( np.abs( np.array( table_hdu[1].data.BIN_ID ) ) )
    pixelsize   = table_hdu[0].header['PIXSIZE']
    wcshdr      = table_hdu[2].header
    
    # Check spatial coordinates
    if len( np.where( np.logical_or( X == 0.0, np.isnan(X) == True ) )[0] ) == len(X):
        print('All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!')
    if len( np.where( np.logical_or( Y == 0.0, np.isnan(Y) == True ) )[0] ) == len(Y):
        print('All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n')

    # Read Results
    if flag == 'KIN':
        hdu = fits.open(os.path.join(outdir,rootname)+'_kin.fits')
    # elif flag == 'SFH':
    #     hdu = fits.open(os.path.join(outdir,rootname)+'_sfh.fits')
        result      = np.zeros((len(ubins),4))
        result[:,0] = np.array( hdu[1].data.V     )
        result[:,1] = np.array( hdu[1].data.SIGMA )
        if hasattr(hdu[1].data, 'H3'): result[:,2] = np.array(hdu[1].data.H3)
        if hasattr(hdu[1].data, 'H4'): result[:,3] = np.array(hdu[1].data.H4)
        labellist = ['V', 'SIG', 'H3', 'H4']

    elif flag == 'SFH':
    # Read results
        sfh_hdu     = fits.open(os.path.join(outdir,rootname)+'_sfh.fits')
        result      = np.zeros((len(ubins),3))
        result[:,0] = np.array( sfh_hdu[1].data.AGE   )
        result[:,1] = np.array( sfh_hdu[1].data.METAL )
        result[:,2] = np.array( sfh_hdu[1].data.ALPHA )
        if len( np.unique( result[:,2] ) ) == 1:
            labellist = ['AGE', 'METAL']
        else:
            labellist = ['AGE', 'METAL', 'ALPHA']
    # Convert results to long version
    result_long  = np.zeros( (len(binNum_long), result.shape[1]) ); result_long[:,:] = np.nan
    for i in range( len(ubins) ):
        idx = np.where( ubins[i] == np.abs(binNum_long) )[0]
        result_long[idx,:]  = result[i,:]
    result = result_long
    result[:,0] = result[:,0] - np.nanmedian( result[:,0] )
    ####### Adding the ability to output maps as fits files
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])
    names = labellist

    for iterate in range(0,len(names)):
        # Prepare main plot
        val = result[:,iterate]
        # Create image in pixels
        xmin = np.min(X);  xmax = np.max(X)
        ymin = np.min(Y);  ymax = np.max(Y)
        npixels_x = int( np.round( (xmax - xmin)/pixelsize ) + 1 )
        npixels_y = int( np.round( (ymax - ymin)/pixelsize ) + 1 )
        i = np.array( np.round( (X - xmin)/pixelsize ), dtype=np.int32 )
        j = np.array( np.round( (Y - ymin)/pixelsize ), dtype=np.int32 )
        image = np.full( (npixels_x, npixels_y), np.nan )
        
        # Reverse the i index to each row of the image
        # because ra increases West-East (right-left in image plane)
        image[i[::-1], j] = val

        # Transpose x and y because numpy uses arr[row, col] and FITS uses 
        # im[ra, dec] = arr[col, row]
        image = image.T

        # make HDU
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(os.path.join(outdir,rootname)+'_'+flag +'_maps.fits', overwrite=True)
    hdu1.close()
    
    
def savefitsmaps_GASmodule(flag, outdir, LEVEL=None, AoNThreshold = 4):
    
    runname  = outdir
    rootname = outdir.rstrip('/').split('/')[-1]

    # Construct a mask for defunct spaxels
    mask = fits.open(os.path.join(outdir,rootname)+'_mask.fits')[1].data.MASK_DEFUNCT
    maskedSpaxel = np.array(mask, dtype=bool)

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir,rootname)+'_table.fits')
    X           = np.array( table_hdu[1].data.X[~maskedSpaxel] ) * -1
    Y           = np.array( table_hdu[1].data.Y[~maskedSpaxel] )
    FLUX        = np.array( table_hdu[1].data.FLUX[~maskedSpaxel] )
    binNum_long = np.array( table_hdu[1].data.BIN_ID[~maskedSpaxel] )
    ubins       = np.unique( np.abs(binNum_long) )
    pixelsize   = table_hdu[0].header['PIXSIZE']
    wcshdr      = table_hdu[2].header

    # Check spatial coordinates
    if len( np.where( np.logical_or( X == 0.0, np.isnan(X) == True ) )[0] ) == len(X):
        print('All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!')
    if len( np.where( np.logical_or( Y == 0.0, np.isnan(Y) == True ) )[0] ) == len(Y):
        print('All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n')

    # Read Gandalf results
    if LEVEL == 'SPAXEL':
        results = fits.open(os.path.join(outdir,rootname)+'_gas_SPAXEL.fits')[1].data[~maskedSpaxel]
    elif LEVEL == 'BIN':
        results = fits.open(os.path.join(outdir,rootname)+'_gas_BIN.fits')[1].data
    elif LEVEL == None:
        print("LEVEL keyword not set!")
    

    # Convert results to long version
    if LEVEL == 'BIN':
        _, idxConvert = np.unique( np.abs(binNum_long), return_inverse=True )
        results = results[idxConvert]
    
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])
    
    # Iterate over all lines
    for line in results.names:
        if line[-3:] == 'AON': continue
        if line in ['EBmV_0', 'EBmV_1']: continue

        data = results[line]
        data_aon = results[line[:-2]+'_AON']

        data[ np.where(data_aon < AoNThreshold)[0] ] = np.nan
        data[ np.where(data == -1)[0] ] = np.nan

        if line.split('_')[-1] == 'V':
            data = data - np.nanmedian(data)

        # Create image in pixels
        xmin = np.min(X);  xmax = np.max(X)
        ymin = np.min(Y);  ymax = np.max(Y)
        npixels_x = int( np.round( (xmax - xmin)/pixelsize ) + 1 )
        npixels_y = int( np.round( (ymax - ymin)/pixelsize ) + 1 )
        col = np.array( np.round( (X - xmin)/pixelsize ), dtype=np.int32 )
        row = np.array( np.round( (Y - ymin)/pixelsize ), dtype=np.int32 )
        image = np.full( (npixels_x, npixels_y), np.nan )

        # reverse the index to flip vertically
        # since WCS transformations - like FITS files - assume
        # that the origin is the lower left pixel of the image 
        # (origin is in top left for numpy arrays)
        image[col[::-1], row] = data

        # Transpose x and y because numpy uses arr[row, col] and FITS uses im[ra, dec] = arr[col, row]
        image = image.T

        image_hdu = fits.ImageHDU(image, header=wcshdr, name=line)
        # Append fits image
        hdu1.append(image_hdu)
                
    hdu1.writeto(os.path.join(outdir,rootname)+'_'+flag +'_maps_' + LEVEL +'.fits', overwrite=True)
    hdu1.close()


def savefitsmaps_LSmodule(flag, outdir, RESOLUTION):
    runname  = outdir
    rootname = outdir.rstrip('/').split('/')[-1]

    # Read bintable
    table_hdu = fits.open(os.path.join(outdir,rootname)+'_table.fits')
    idx_inside  = np.where( table_hdu[1].data.BIN_ID >= 0        )[0]
    X           = np.array( table_hdu[1].data.X    ) * -1
    Y           = np.array( table_hdu[1].data.Y      )
    FLUX        = np.array( table_hdu[1].data.FLUX   )
    binNum_long = np.array( table_hdu[1].data.BIN_ID )
    ubins       = np.unique( np.abs( np.array( table_hdu[1].data.BIN_ID ) ) )
    pixelsize   = table_hdu[0].header['PIXSIZE']
    wcshdr      = table_hdu[2].header
    
    # Check spatial coordinates
    if len( np.where( np.logical_or( X == 0.0, np.isnan(X) == True ) )[0] ) == len(X):
        print('All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!')
    if len( np.where( np.logical_or( Y == 0.0, np.isnan(Y) == True ) )[0] ) == len(Y):
        print('All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n')


    # Read results
    if RESOLUTION == 'ORIGINAL':
        hdu     = fits.open(os.path.join(outdir,rootname)+'_ls_OrigRes.fits')
    elif RESOLUTION =='ADAPTED':
        hdu     = fits.open(os.path.join(outdir,rootname)+'_ls_AdapRes.fits')
    result      = np.zeros((len(ubins),3))
    result[:,0] = np.array( hdu[1].data.Hbeta_o   )
    result[:,1] = np.array( hdu[1].data.Fe5015 )
    result[:,2] = np.array( hdu[1].data.Mgb )

    labellist = ['Hbeta_o', 'Fe5015', 'Mgb']

    # Convert results to long version
    result_long  = np.zeros( (len(binNum_long), result.shape[1]) ); result_long[:,:] = np.nan
    for i in range( len(ubins) ):
        idx = np.where( ubins[i] == np.abs(binNum_long) )[0]
        result_long[idx,:]  = result[i,:]
    result = result_long
    result[:,0] = result[:,0] - np.nanmedian( result[:,0] )
    ####### Adding the ability to output maps as fits files
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.HDUList([primary_hdu])
    names = labellist
    
    for iterate in range(0,len(names)):
        # Prepare main plot
        val = result[:,iterate]
        
        # Create image in pixels
        xmin = np.min(X);  xmax = np.max(X)
        ymin = np.min(Y);  ymax = np.max(Y)
        npixels_x = int( np.round( (xmax - xmin)/pixelsize ) + 1 )
        npixels_y = int( np.round( (ymax - ymin)/pixelsize ) + 1 )
        i = np.array( np.round( (X - xmin)/pixelsize ), dtype=np.int32 )
        j = np.array( np.round( (Y - ymin)/pixelsize ), dtype=np.int32 )
        image = np.full( (npixels_x, npixels_y), np.nan )
        
        # Reverse the i index to each row of the image
        # because ra increases West-East (right-left in image plane)
        image[i[::-1], j] = val

        # Transpose x and y to reorient the image correctly
        # im[ra, dec] = arr[col, row]
        image = image.T
        
        # make HDU
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(os.path.join(outdir,rootname)+'_'+flag +'_maps' + RESOLUTION +'.fits', overwrite=True)
    hdu1.close()
