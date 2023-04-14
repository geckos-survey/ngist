#!/usr/bin/env python

from   astropy.io import fits
import numpy      as np
import os
import optparse
import warnings
warnings.filterwarnings('ignore')
from matplotlib.tri import Triangulation, TriAnalyzer

import matplotlib.pyplot       as     plt
from   mpl_toolkits.axes_grid1 import AxesGrid
from   matplotlib.ticker       import MultipleLocator, FuncFormatter

from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()

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
        image[i,j] = val
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(os.path.join(outdir,rootname)+'_'+flag +'_maps.fits', overwrite=True)
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
        image[i,j] = val
        image_hdu = fits.ImageHDU(image, header=wcshdr, name=names[iterate])
        # Append fits image
        hdu1.append(image_hdu)
    hdu1.writeto(os.path.join(outdir,rootname)+'_'+flag +'_maps' + RESOLUTION +'.fits', overwrite=True)
    hdu1.close()
