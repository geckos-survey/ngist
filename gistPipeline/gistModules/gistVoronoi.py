from   astropy.io             import fits
import numpy                  as np
import scipy.spatial.distance as dist

import functools
import logging

from gistPipeline.gistModules import util as pipeline

try:
    # Try to use local version in sitePackages
    from gistPipeline.sitePackages.voronoi.voronoi_2d_binning import voronoi_2d_binning
except: 
    # Then use system installed version instead
    from vorbin.voronoi_2d_binning import voronoi_2d_binning


"""
PURPOSE: 
  This file contains a collection of functions necessary to Voronoi-bin the data. 
  The Voronoi-binning makes use of the algorithm from Cappellari & Copin 2003
  (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C). 
"""


def sn_func(index, signal=None, noise=None, covar_vor=0.00 ):
    """
       This function is passed to the Voronoi binning routine of Cappellari &
       Copin 2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) and used to
       estimate the noise in the bin from the noise in the spaxels. This
       implementation is identical to the default one, but accounts for spatial
       correlations in the noise by applying an empirical equation (see e.g.
       Garcia-Benito et al. 2015;
       ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G) together with the
       parameter defined in the Config-file. 
    """

    # Add the noise in the spaxels to obtain the noise in the bin
    sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))

    # Account for spatial correlations in the noise by applying an empirical
    # equation (see e.g. Garcia-Benito et al. 2015;
    # ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G)
    sn /= 1 + covar_vor * np.log10(index.size)

    return(sn)


def define_voronoi_bins(VORONOI, x, y, signal, noise, pixelsize, snr, target_snr, covar_vor, idx_inside, idx_outside, rootname, outdir):
    """
    This function applies the Voronoi-binning algorithm of Cappellari & Copin
    2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) to the data. It can
    be accounted for spatial correlations in the noise (see function sn_func()).
    A BIN_ID is assigned to every spaxel. Spaxels which do not satisfy the
    minimum SNR threshold are excluded from the Voronoi-binning, but are
    assigned a negative BIN_ID, with the absolute value of the BIN_ID
    corresponding to the nearest Voronoi-bin that satisfies the minimum SNR
    threshold.  All results are saved in a dedicated table to provide easy means
    of matching spaxels and bins. 
    """
    # Pass a function for the SNR calculation to the Voronoi-binning algorithm,
    # in order to account for spatial correlations in the noise
    sn_func_covariances = functools.partial( sn_func, covar_vor=covar_vor )

    if VORONOI == 1: 
        # Generate the Voronoi bins
        pipeline.prettyOutput_Running("Defining the Voronoi bins")
        logging.info("Defining the Voronoi bins")

        try:
            # Do the Voronoi binning
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, _ = voronoi_2d_binning(x[idx_inside], y[idx_inside],\
                    signal[idx_inside], noise[idx_inside], target_snr, plot=False, quiet=True, pixelsize=pixelsize,
                    sn_func=sn_func_covariances )

            pipeline.prettyOutput_Done("Defining the Voronoi bins")
            print("             "+str(np.max(binNum)+1)+" voronoi bins generated!")
            logging.info(str(np.max(binNum)+1)+" Voronoi bins generated!")

        # Handle common exceptions
        except ValueError as e:

            # Sufficient SNR and no binning needed
            if str(e) == 'All pixels have enough S/N and binning is not needed': 

                pipeline.prettyOutput_Warning("Defining the Voronoi bins")
                print("             "+"The Voronoi-binning routine of Cappellari & Copin (2003) returned the following error:")
                print("             "+str(e)+"\n")
                pipeline.prettyOutput_Warning("Analysis will continue without Voronoi-binning!")
                print("             "+str(len(idx_inside))+" spaxels will be treated as Voronoi-bins.")
                
                logging.warning("Defining the Voronoi bins failed. The Voronoi-binning routine of Cappellari & Copin "+\
                                "(2003) returned the following error: \n"+str(e))
                logging.info("Analysis will continue without Voronoi-binning! "+str(len(idx_inside))+" spaxels will be treated as Voronoi-bins.")

                binNum, xNode, yNode, sn, nPixels = noBinning( x, y, snr, idx_inside )

            # Any uncaught exceptions
            else:
                pipeline.prettyOutput_Failed("Defining the Voronoi bins")
                print("The Voronoi-binning routine of Cappellari & Copin (2003) returned the following error: \n"+str(e))
                logging.error("Defining the Voronoi bins failed. The Voronoi-binning routine of Cappellari & Copin "+\
                              "(2003) returned the following error: \n"+str(e))
                return( True )

    # Do not bin
    else: 
        pipeline.prettyOutput_Done("No Voronoi-bins are generated.")
        logging.info("No Voronoi-bins are generated.")
        binNum, xNode, yNode, sn, nPixels = noBinning( x, y, snr, idx_inside )

    # Find the nearest Voronoi bin for the pixels outside the Voronoi region
    binNum_outside = find_nearest_voronoibin( x, y, idx_outside, xNode, yNode )

    # Generate extended binNum-list: 
    #   Positive binNum indicate the Voronoi bin of the spaxel (for spaxels inside the Voronoi region)
    #   Negative binNum indicate the nearest Voronoi bin of the spaxel (for spaxels outside of the Voronoi region)
    ubins = np.unique(binNum)
    nbins = len(ubins)
    binNum_long = np.zeros( len(x) ); binNum_long[:] = np.nan
    binNum_long[idx_inside]  = binNum
    binNum_long[idx_outside] = -1 * binNum_outside

    # Save bintable: data for *ALL* spectra inside and outside of the Voronoi region!
    save_table(rootname, outdir, x, y, signal, snr, binNum_long, ubins, xNode, yNode, sn, nPixels, pixelsize)

    return(binNum)


def noBinning(x, y, snr, idx_inside):
    """ 
    In case no Voronoi-binning is required/possible, treat spaxels in the input
    data as Voronoi bins, in order to continue the analysis. 
    """
    binNum  = np.arange( 0, len(idx_inside) )
    xNode   = x[idx_inside]
    yNode   = y[idx_inside]
    sn      = snr[idx_inside]
    nPixels = np.ones( len(idx_inside) )

    return( binNum, xNode, yNode, sn, nPixels )


def find_nearest_voronoibin(x, y, idx_outside, xNode, yNode):
    """
    This function determines the nearest Voronoi-bin for all spaxels which do
    not satisfy the minimum SNR threshold. 
    """
    x = x[idx_outside]
    y = y[idx_outside]
    pix_coords = np.concatenate( (x.reshape((len(x),1)),         y.reshape((len(y),1))),         axis=1 )
    bin_coords = np.concatenate( (xNode.reshape((len(xNode),1)), yNode.reshape((len(yNode),1))), axis=1 )

    dists = dist.cdist( pix_coords, bin_coords, 'euclidean' ) 
    closest = np.argmin( dists, axis=1 )

    return(closest)


def save_table(rootname, outdir, x, y, signal, snr, binNum_new, ubins, xNode, yNode, sn, nPixels, pixelsize):
    """ 
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins. 
    """
    outfits_table = outdir+rootname+'_table.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_table.fits')

    # Expand data to spaxel level
    xNode_new = np.zeros( len(x) )
    yNode_new = np.zeros( len(x) )
    sn_new = np.zeros( len(x) )
    nPixels_new = np.zeros( len(x) )
    for i in range( len(ubins) ):
        idx = np.where( ubins[i] == np.abs(binNum_new) )[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        sn_new[idx] = sn[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name='ID',        format='J',   array=np.arange(len(x)) ))
    cols.append(fits.Column(name='BIN_ID',    format='J',   array=binNum_new        ))
    cols.append(fits.Column(name='X',         format='D',   array=x                 ))
    cols.append(fits.Column(name='Y',         format='D',   array=y                 ))
    cols.append(fits.Column(name='FLUX',      format='D',   array=signal            ))
    cols.append(fits.Column(name='SNR',       format='D',   array=snr               ))
    cols.append(fits.Column(name='XBIN',      format='D',   array=xNode_new         ))
    cols.append(fits.Column(name='YBIN',      format='D',   array=yNode_new         ))
    cols.append(fits.Column(name='SNRBIN',    format='D',   array=sn_new            ))
    cols.append(fits.Column(name='NSPAX',     format='J',   array=nPixels_new       ))

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "TABLE"

    # Create HDU list and write to file
    priHDU = pipeline.createGISTHeaderComment( priHDU )
    tbhdu  = pipeline.createGISTHeaderComment( tbhdu  )
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    pipeline.prettyOutput_Done("Writing: "+rootname+'_table.fits')
    logging.info("Wrote Voronoi table: "+outfits_table)


def apply_voronoi_bins(binNum, spec, espec, rootname, outdir, velscale, wave, flag):
    """
    The constructed Voronoi-binning is applied to the underlying spectra. The
    resulting Voronoi-binned spectra are saved to disk. 
    """
    # Apply Voronoi bins
    pipeline.prettyOutput_Running("Applying the Voronoi bins to "+flag+"-data")
    bin_data, bin_error, bin_flux = voronoi_binning( binNum, spec, espec )
    pipeline.prettyOutput_Done("Applying the Voronoi bins to "+flag+"-data", progressbar=True)
    logging.info("Applied Voronoi bins to "+flag+"-data")

    # Save Voronoi binned spectra
    save_vorspectra(rootname, outdir, bin_data, bin_error, velscale, wave, flag)
    return(None)


def voronoi_binning( binNum, spec, error ):
    """ Spectra belonging to the same Voronoi-bin are added. """
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
           av_err_spec = error[:,k]
        else:
           av_spec     = np.nansum(spec[:,k],axis=1)
           av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
    
        bin_data[:,i]  = np.ravel(av_spec)
        bin_error[:,i] = np.ravel(av_err_spec)
        bin_flux[i]    = np.mean(av_spec,axis=0)
        pipeline.printProgress(i+1, nbins, barLength = 50)

    return(bin_data, bin_error, bin_flux)


def save_vorspectra(rootname, outdir, log_spec, log_error, velscale, logLam, flag):
    """ Voronoi-binned spectra and error spectra are saved to disk. """
    if flag == 'log':
        outfits_spectra  = outdir+rootname+'_VorSpectra.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_VorSpectra.fits')
    elif flag == 'lin':
        outfits_spectra  = outdir+rootname+'_VorSpectra_linear.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_VorSpectra_linear.fits')

    npix = len(log_spec)

    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'VOR_SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'

    # Create HDU list and save to file
    priHDU    = pipeline.createGISTHeaderComment( priHDU    )
    dataHDU   = pipeline.createGISTHeaderComment( dataHDU   )
    loglamHDU = pipeline.createGISTHeaderComment( loglamHDU )

    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    fits.setval(outfits_spectra,'VELSCALE',value=velscale)
    fits.setval(outfits_spectra,'CRPIX1',  value=1.0)
    fits.setval(outfits_spectra,'CRVAL1',  value=logLam[0])
    fits.setval(outfits_spectra,'CDELT1',  value=logLam[1]-logLam[0])

    if flag == 'log':
        pipeline.prettyOutput_Done("Writing: "+rootname+'_VorSpectra.fits')
    elif flag == 'lin':
        pipeline.prettyOutput_Done("Writing: "+rootname+'_VorSpectra_linear.fits')
    logging.info("Wrote: "+outfits_spectra)
