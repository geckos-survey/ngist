from astropy.io import fits, ascii
import numpy as np
from multiprocessing import Queue, Process

import time
import logging
import os

from gistPipeline.gistModules import util         as pipeline
from gistPipeline.gistModules import gistPlot_ls  as util_plot_ls
from gistPipeline.gistModules import gistPlot_spp as util_plot_spp

from gistPipeline.sitePackages.lineStrength import lsindex_spec   as lsindex
from gistPipeline.sitePackages.lineStrength import ssppop_fitting as ssppop

try:
    # Try to use local version in sitePackages
    from gistPipeline.sitePackages.ppxf.ppxf_util import gaussian_filter1d
except:
    # Then use system installed version instead
    from ppxf.ppxf_util import gaussian_filter1d

cvel  = 299792.458


"""
PURPOSE: 
  This module executes the measurement of line strength indices in the pipeline.
  Basically, it acts as an interface between pipeline and the line strength
  measurement routines of Kuntschner et al. 2006
  (ui.adsabs.harvard.edu/?#abs/2006MNRAS.369..497K) and their conversion to
  single stellar population equivalent population properties with the MCMC
  algorithm of Martin-Navaroo et al. 2018
  (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M). 
"""


def workerLS(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for wave, spec, espec, redshift, configs, lickfile, names, index_names,\
        model_indices, params, tri, labels, outdir, nbins, i, MCMC\
        in iter(inQueue.get, 'STOP'):

        if MCMC == True: 
            indices, errors, vals, percentile = run_ls( wave, spec, espec, redshift, configs, lickfile, names, index_names,\
            model_indices, params, tri, labels, outdir, nbins, i, MCMC )

            outQueue.put(( i, indices, errors, vals, percentile ))

        elif MCMC == False: 
            indices, errors = run_ls( wave, spec, espec, redshift, configs, lickfile, names, index_names,\
            model_indices, params, tri, labels, outdir, nbins, i, MCMC )

            outQueue.put(( i, indices, errors ))


def run_ls(wave, spec, espec, redshift, configs, lickfile, names, index_names,\
           model_indices, params, tri, labels, outdir, nbins, i, MCMC): 
    """
    Calls a Python version of the line strength measurement routine of
    Kuntschner et al. 2006 (ui.adsabs.harvard.edu/?#abs/2006MNRAS.369..497K),
    and if required, the MCMC algorithm from Martin-Navaroo et al. 2018
    (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M) to determine SSP
    properties. 
    """
    pipeline.printProgress(i, nbins, barLength = 50)
    nindex = len(index_names)

    try:
        # Measure the LS indices
        names, indices, errors = lsindex.lsindex\
                    (wave, spec, espec, redshift[0], lickfile, sims=configs['MC_LS'], z_err=redshift[1], plot=0)
    
        # Get the indices in consideration
        data  = np.zeros(nindex)
        error = np.zeros(nindex)
        for o in range( nindex ):
            idx = np.where( names == index_names[o] )[0]
            data[o]  = indices[idx]
            error[o] = errors[idx]
    
        if MCMC == True: 
            # Run the conversion of LS indices to SSP properties
            vals   = np.zeros(len(labels)*3+2)
            chains = np.zeros((int(configs['NWALKER']*configs['NCHAIN']/2), len(labels)))
            vals[:], chains[:,:] = ssppop.ssppop_fitting\
                (data, error, model_indices, params, tri, labels, configs['NWALKER'], configs['NCHAIN'], False, 0, i, nbins, outdir)
        
            percentiles = np.percentile( chains, np.arange(101), axis=0 )
    
            return(indices, errors, vals, percentiles)
    
        elif MCMC == False: 
            return(indices, errors)

    except:
        if MCMC == True: 
            return( np.nan, np.nan, np.nan, np.nan )
        elif MCMC == False: 
            return( np.nan, np.nan )


def save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, vals=None, percentile=None ):
    """ Saves all results to disk. """
    # Save results
    if RESOLUTION == 'ORIGINAL': 
        outfits = outdir+rootname+'_ls_OrigRes.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_ls_OrigRes.fits')
    if RESOLUTION == 'ADAPTED': 
        outfits = outdir+rootname+'_ls_AdapRes.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_ls_AdapRes.fits')
   
    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with LS output data
    cols = []
    ndim  = len(names)
    for i in range(ndim):
        cols.append( fits.Column(name=names[i],        format='D', array=ls_indices[:,i] ))
        cols.append( fits.Column(name="ERR_"+names[i], format='D', array=ls_errors[:,i]  ))
    cols.append( fits.Column(name="FWHM_FLAG",         format='I', array=totalFWHM_flag[:]  ))
    lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    lsHDU.name = "LS_DATA"

    if MCMC == True: 
        # Extension 2: Table HDU with SSP-equivalent output data
        nparam  = len(labels)
        cols = []
        for i in range(nparam):
            cols.append( fits.Column(name=labels[i],           format='101D', array=percentile[:,:,i] ))
        cols.append( fits.Column(    name='lnP',               format='D',    array=vals[:,-2]        ))
        cols.append( fits.Column(    name='Flag',              format='D',    array=vals[:,-1]        ))
        sspHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)) 
        sspHDU.name = "SSP_DATA"

        # Create HDU list
        HDUList = fits.HDUList([priHDU, lsHDU, sspHDU])

    if MCMC == False: 
        # Create HDU list
        HDUList = fits.HDUList([priHDU, lsHDU])

    # Write HDU list to file
    HDUList.writeto(outfits, overwrite=True)

    if RESOLUTION == 'ORIGINAL': 
        pipeline.prettyOutput_Done("Writing: "+rootname+'_ls_OrigRes.fits')
    if RESOLUTION == 'ADAPTED': 
        pipeline.prettyOutput_Done("Writing: "+rootname+'_ls_AdapRes.fits')
    logging.info("Wrote: "+outfits)


def saveCleanedLinearSpectra(spec, espec, wave, npix, outdir, rootname):
    """ Save emission-subtracted, linearly binned spectra to disk. """
    outfits = outdir+rootname+'_ls-cleaned_linear.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_ls-cleaned_linear.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with cleaned, linear spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=spec ) )
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=espec ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'CLEANED_SPECTRA'

    # Extension 2: Table HDU with wave
    cols = []
    cols.append( fits.Column(name='LAM', format='D', array=wave) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LAM'
     
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_ls-cleaned_linear.fits')
    logging.info("Wrote: "+outfits)


def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline. 
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1]-lamRange[0]) / (n - 1)             # Step in log-space
    lim = lamRange + np.array([-0.5, 0.5])*dLam            # Min and max wavelength in log-space
    borders = np.linspace( lim[0], lim[1], n+1 )           # OLD logLam in log-space
    
    # Wavelength domain
    logLim     = np.exp(lim)                               # Min and max wavelength in Angst.
    lamNew     = np.linspace( logLim[0], logLim[1], m+1 )  # new logLam in Angstroem
    newBorders = np.log(lamNew)                            # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor( (newBorders-lim[0]) / dLam ).astype('int')

    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m-1):
        a = (newBorders[j]   - borders[k[j]])   / dLam
        b = (borders[k[j+1]] - newBorders[j+1]) / dLam
        
        specNew[j] = np.sum( spec[k[j]:k[j+1]] ) - a*spec[k[j]] - b*spec[k[j+1]]

    # Rescale flux
    if flux == True:
        specNew = specNew / ( newBorders[1:] - newBorders[:-1] ) * np.mean( newBorders[1:] - newBorders[:-1] ) * oversample

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1]-lamNew[0])

    return( specNew, lamNew )


def runModule_LINESTRENGTH(LINE_STRENGTH, RESOLUTION, PARALLEL, configs, velscale, LSF_Data, outdir, rootname):
    """
    Starts the line strength analysis. Data is read in, emission-subtracted
    spectra are rebinned from logarithmic to linear scale, and the spectra
    convolved to meet the LIS measurement resolution. After the measurement of
    line strength indices and, if required, the estimation of SSP properties,
    the results are saved to file and the plotting routines executed. 
    """
    # Run MCMC only on the indices measured from convoluted spectra
    if LINE_STRENGTH == 2  and  RESOLUTION == "ADAPTED": 
        MCMC = True
    else: 
        MCMC = False

    # Double check what to do
    if RESOLUTION == "ADAPTED"  and  os.path.isfile(outdir+rootname+'_ls_AdapRes.fits') == False  and  LINE_STRENGTH != 0: 
        SKIP = False
    elif RESOLUTION == "ORIGINAL"  and  os.path.isfile(outdir+rootname+'_ls_OrigRes.fits') == False  and  LINE_STRENGTH != 0: 
        SKIP = False
    else:
        SKIP = True

    if SKIP == False:
        print("\033[0;37m"+" - - - - - Running LINE STRENGTHS - - - - -"+"\033[0;39m")
        logging.info(" - - - Running LINE STRENGTHS - - - ")


        # Read the log-rebinned, cleaned spectra from GANDALF and log-unbin them
        if os.path.isfile(outdir+rootname+'_ls-cleaned_linear.fits') == False:

            # Read cleaned spectra
            logging.info("Reading "+outdir+rootname+"_gandalf-cleaned_BIN.fits")
            hdu_spec  = fits.open(outdir+rootname+'_gandalf-cleaned_BIN.fits')
            hdu_espec = fits.open(outdir+rootname+'_VorSpectra.fits')
            oldspec  = np.array( hdu_spec[1].data.SPEC   )
            oldespec = np.sqrt( np.array( hdu_espec[1].data.ESPEC ) )
            wave     = np.array( hdu_spec[2].data.LOGLAM )
            nbins    = oldspec.shape[0]
            npix     = oldspec.shape[1]
            lamRange = np.array([ wave[0], wave[-1] ])
            spec     = np.zeros( oldspec.shape  )
            espec    = np.zeros( oldespec.shape )
    
            # Rebin the cleaned spectra from log to lin
            pipeline.prettyOutput_Running("Rebinning the cleaned spectra from log to lin")
            for i in range( nbins ):
                pipeline.printProgress(i, nbins, barLength = 50)
                spec[i,:], wave = log_unbinning( lamRange, oldspec[i,:] )
            pipeline.prettyOutput_Done("Rebinning the cleaned spectra from log to lin", progressbar=True)
    
            # Rebin the error spectra from log to lin
            pipeline.prettyOutput_Running("Rebinning the error spectra from log to lin")
            for i in range( nbins ):
                pipeline.printProgress(i, nbins, barLength = 50)
                espec[i,:], _ = log_unbinning( lamRange, oldespec[i,:] )
            pipeline.prettyOutput_Done("Rebinning the error spectra from log to lin", progressbar=True)

            # Save cleaned, linear spectra
            saveCleanedLinearSpectra(spec, espec, wave, npix, outdir, rootname)
        
        # Read the linearly-binned, cleaned spectra provided by previous LS-run
        else: 
            logging.info("Reading "+outdir+rootname+'_ls-cleaned_linear.fits')
            hdu   = fits.open(outdir+rootname+'_ls-cleaned_linear.fits')
            spec  = np.array( hdu[1].data.SPEC  )
            espec = np.array( hdu[1].data.ESPEC )
            wave  = np.array( hdu[2].data.LAM   )
            nbins = spec.shape[0]
 
        # Read PPXF results
        ppxf_data     = fits.open(outdir+rootname+'_ppxf.fits')[1].data
        redshift      = np.zeros((nbins, 2))                       # Dimensionless z
        redshift[:,0] = np.array( ppxf_data.V[:] ) / cvel          # Redshift
        redshift[:,1] = np.array( ppxf_data.FORM_ERR_V[:] ) / cvel # Error on redshift
        veldisp_kin   = np.array( ppxf_data.SIGMA[:] )

        # Read file defining the LS bands
        lickfile = outdir+configs['LS_FILE']
        tab   = ascii.read(lickfile, comment='\s*#')
        names = tab['names']

        # Flag spectra for which the total intrinsic dispersion is larger than the LIS measurement resolution
        totalFWHM_flag = np.zeros(spec.shape[0])

        # Broaden spectra to LIS resolution taking into account the measured velocity dispersion
        if RESOLUTION == "ADAPTED":
            pipeline.prettyOutput_Running("Broadening the spectra to LIS resolution")
            # Iterate over all bins
            for i in range(0, spec.shape[0]):
                pipeline.printProgress(i, nbins, barLength = 50)

                # Convert velocity dispersion of galaxy (from PPXF) to Angstrom
                veldisp_kin_Angst = veldisp_kin[i] * wave / cvel * 2.355  

                # Total dispersion for this bin
                total_dispersion = np.sqrt( LSF_Data(wave)**2 + veldisp_kin_Angst**2 )

                # Difference between total dispersion and LIS measurement resolution
                FWHM_dif = np.sqrt( configs['CONV_COR']**2 - total_dispersion**2 )

                # Convert resolution difference from Angstrom to pixel
                sigma = (FWHM_dif / wave) * cvel / 2.355 / velscale

                # Flag spectrum if the total intrinsic dispersion is larger than the LIS measurement resolution
                idx = np.where( np.isnan( sigma ) == True )[0]
                if len( idx ) > 0:
                    sigma[idx] = 0.0
                    totalFWHM_flag[i] = 1 

                # Convolve spectra pixel-wise
                spec[i,:]  = gaussian_filter1d(spec[i,:],  sigma)
                espec[i,:] = gaussian_filter1d(espec[i,:], sigma)
            pipeline.prettyOutput_Done("Broadening the spectra to LIS resolution", progressbar=True)

        # Get indices that are considered in SSP-conversion
        idx         = np.where( tab['spp'] == 1 )[0]
        index_names = tab['names'][idx].tolist()

        # Loading model predictions
        if MCMC == True: 
            modelfile = configs['SSP_LIB'].rstrip('/')+"_KB_LIS"+str(configs['CONV_COR'])+".fits"
            model_indices, params, tri, labels = ssppop.load_models(modelfile, index_names)
            logging.info("Loading LS model file at "+modelfile)
        elif MCMC == False: 
            model_indices, params, tri, labels = "dummy", "dummy", "dummy", "dummy"

        # Arrays to store results
        ls_indices = np.zeros((nbins, len(names)))
        ls_errors  = np.zeros((nbins, len(names)))
        if MCMC == True: 
            vals       = np.zeros((nbins, len(labels)*3+2))
            percentile = np.zeros((nbins, 101, len(labels)))

        # Run LS Measurements
        start_time = time.time()
        if PARALLEL == True:
            pipeline.prettyOutput_Running("Running LINE_STRENGTH in parallel mode")            
            logging.info("Running LINE_STRENGTH in parallel mode")            

            # Create Queues
            inQueue  = Queue()
            outQueue = Queue()
        
            # Create worker processes
            ps = [Process(target=workerLS, args=(inQueue, outQueue))
                  for _ in range(configs['NCPU'])]
        
            # Start worker processes
            for p in ps: p.start()
        
            # Fill the queue
            for i in range(nbins):
                inQueue.put( ( wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                               model_indices, params, tri, labels, outdir, nbins, i, MCMC ) )

            # now get the results with indices
            ls_tmp = [outQueue.get() for _ in range(nbins)]
        
            # send stop signal to stop iteration
            for _ in range(configs['NCPU']): inQueue.put('STOP')

            # stop processes
            for p in ps: p.join()
        
            # Get output
            index = np.zeros(nbins)
            for i in range(0, nbins):
                index[i]          = ls_tmp[i][0]
                ls_indices[i,:]   = ls_tmp[i][1]
                ls_errors[i,:]    = ls_tmp[i][2]
                if MCMC == True: 
                    vals[i,:]         = ls_tmp[i][3]
                    percentile[i,:,:] = ls_tmp[i][4]

            # Sort output
            argidx = np.argsort( index )
            ls_indices = ls_indices[argidx,:]
            ls_errors  = ls_errors[argidx,:]
            if MCMC == True: 
                vals       = vals[argidx,:]
                percentile = percentile[argidx,:,:]

            pipeline.prettyOutput_Done("Running LINE_STRENGTH in parallel mode", progressbar=True)

        if PARALLEL == False:
            pipeline.prettyOutput_Running("Running LINE_STRENGTH in serial mode")            
            logging.info("Running LINE_STRENGTH in serial mode")            

            if MCMC == True:
                for i in range(nbins):
                    ls_indices[i,:], ls_errors[i,:], vals[i,:], percentile[i,:,:] = run_ls\
                            (wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                            model_indices, params, tri, labels, outdir, nbins, i, MCMC)
            elif MCMC == False: 
                for i in range(nbins):
                    ls_indices[i,:], ls_errors[i,:] = run_ls\
                            (wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                            model_indices, params, tri, labels, outdir, nbins, i, MCMC)


            pipeline.prettyOutput_Done("Running LINE_STRENGTH in serial mode", progressbar=True)

        print("             Running LINE_STRENGTH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))
        logging.info("Running LINE_STRENGTH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))

        # Check for exceptions which occurred during the analysis
        idx_error = np.where( np.all( np.isnan(ls_indices[:,:]), axis=1 ) == True )[0]
        if len(idx_error) != 0:
            pipeline.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
            print("             "+str(idx_error))
            logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
        else:
            print("             "+"There were no problems in the analysis.")
            logging.info("There were no problems in the analysis.")
        print("")

        # Save Results
        if MCMC == True: 
            save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, vals=vals, percentile=percentile)
        elif MCMC == False: 
            save_ls(names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname)

        # Do Plots
        try:
            pipeline.prettyOutput_Running("Producing line strength maps")
            logging.info("Producing line strength maps")
            util_plot_ls.plot_maps(outdir, RESOLUTION)
            if MCMC == True:
                util_plot_spp.plot_maps("LS", outdir)
            pipeline.prettyOutput_Done("Producing line strength maps")
        except:
            pipeline.prettyOutput_Failed("Producing line strength maps")
            logging.warning("Failed to produce line strength maps. Analysis continues!")
            pass

        print("\033[0;37m"+" - - - - - LINE STRENGTHS done - - - - -"+"\033[0;39m")
        print("")
        logging.info(" - - - LINE STRENGTHS Done - - - \n")
        
    elif SKIP == True:
        print("")
        print(pipeline.prettyOutput_WarningPrefix()+"Skipping LINE STRENGTHS!")
        print("")
        logging.warning("Skipping LINE STRENGTHS\n")
