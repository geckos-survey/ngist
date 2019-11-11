import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import time
import logging
import os

from gistPipeline.gistModules import util                as pipeline
from gistPipeline.gistModules import gistPrepare         as util_prepare
from gistPipeline.gistModules import gistPlot_kinematics as util_plot
from gistPipeline.gistModules import gistPlot_lambdar    as util_plot_lambdar

try:
    # Try to use local version in sitePackages
    from gistPipeline.sitePackages.ppxf.ppxf import ppxf
except: 
    # Then use system installed version instead
    from ppxf.ppxf import ppxf

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE: 
  This module executes the analysis of stellar kinematics in the pipeline. 
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C). 
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, bin_data, noise, velscale, start, goodPixels_ppxf, nmoments,\
        adeg, mdeg, offset, velscale_ratio, nsims, nbins, i\
        in iter(inQueue.get, 'STOP'):

        sol, bestfit, optimal_template, mc_results, formal_error = \
          run_ppxf(templates, bin_data, noise, velscale, start, goodPixels_ppxf, nmoments, adeg, mdeg, offset, velscale_ratio, nsims, nbins, i)

        outQueue.put(( i, sol, bestfit, optimal_template, mc_results, formal_error ))


def run_ppxf( templates, log_bin_data, log_bin_error, velscale, start, goodPixels, nmoments, adeg, mdeg,\
        offset, velscale_ratio, nsims, nbins, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004 
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C; 
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics. 
    """
    pipeline.printProgress( i, nbins, barLength=50 )

    try:
        # Call PPXF
        pp = ppxf(templates, log_bin_data, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
                  quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, velscale_ratio=velscale_ratio, vsyst=offset)
    
        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]
        
        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        # Do MC-Simulations
        sol_MC     = np.zeros((nsims,nmoments)); sol_MC[:,:] = np.nan
        mc_results = np.zeros(nmoments);         mc_results  = np.nan
        for o in range(0, nsims):
            # Add noise to bestfit: 
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit  +  np.random.normal(0, 1, len(log_bin_data)) * np.std( log_bin_data[goodPixels] - pp.bestfit[goodPixels] )
    
            mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start, goodpixels=goodPixels, plot=False, \
                    quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, velscale_ratio=velscale_ratio, vsyst=offset, bias=0.0)
            sol_MC[o,:] = mc.sol[:]
     
        if nsims != 0:
            mc_results = np.nanstd( sol_MC, axis=0 )
     
        return(pp.sol[:], pp.bestfit, optimal_template, mc_results, formal_error)

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan )


def calc_LambdaR( ppxf_result, nbins, outdir, rootname ):
    """
    Calculate the lambda parameter as a proxy for the projected, specific
    angular momentum of the galaxy (see Emsellem et al. 2007;
    ui.adsabs.harvard.edu/#abs/2007MNRAS.379..401E). Note that this quantity is
    not calculated as integrated value per galaxy, but for every Voronoi-bin
    individually.  
    """
    # Calculate lambda_r
    hdu  = fits.open(outdir+rootname+'_table.fits')
    BIN_ID = hdu[1].data.BIN_ID
    X      = hdu[1].data.X
    Y      = hdu[1].data.Y
    FLUX   = hdu[1].data.FLUX 

    velocity = ppxf_result[:,0] - np.median(ppxf_result[:,0])
    sigma    = ppxf_result[:,1] 

    lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
    for i in range(0, nbins):
        idx = np.where( BIN_ID == i )[0]
        numerator   = 0
        denominator = 0
        for o in range(0, len(idx)):
            radius = np.sqrt( X[idx[o]]**2 + Y[idx[o]]**2 )
            numerator   += FLUX[idx[o]] * radius * np.abs( velocity[i] )
            denominator += FLUX[idx[o]] * radius * np.sqrt( velocity[i]**2 + sigma[i]**2 )
        lambda_r[i] = numerator / denominator

    return( lambda_r )


def save_ppxf(rootname, outdir, ppxf_result, mc_results, formal_error, lambda_r,\
              ppxf_bestfit, logLam, goodPixels, optimal_template, logLam_template, npix, ubins):
    """ Saves all results to disk. """
    # ========================
    # SAVE RESULTS
    outfits_ppxf = outdir+rootname+'_ppxf.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_ppxf.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with PPXF output data
    cols = []
    cols.append( fits.Column(name='BIN_ID',         format='J', array=ubins             ))
    cols.append( fits.Column(name='V' ,             format='D', array=ppxf_result[:,0]  ))
    cols.append( fits.Column(name='SIGMA',          format='D', array=ppxf_result[:,1]  ))
    cols.append( fits.Column(name='H3',             format='D', array=ppxf_result[:,2]  ))
    cols.append( fits.Column(name='H4',             format='D', array=ppxf_result[:,3]  ))
    cols.append( fits.Column(name='H5',             format='D', array=ppxf_result[:,4]  ))
    cols.append( fits.Column(name='H6',             format='D', array=ppxf_result[:,5]  ))
    cols.append( fits.Column(name='LAMBDA_R',       format='D', array=lambda_r[:]       ))
    
    cols.append( fits.Column(name='ERR_V' ,         format='D', array=mc_results[:,0]   ))
    cols.append( fits.Column(name='ERR_SIGMA',      format='D', array=mc_results[:,1]   ))
    cols.append( fits.Column(name='ERR_H3',         format='D', array=mc_results[:,2]   ))
    cols.append( fits.Column(name='ERR_H4',         format='D', array=mc_results[:,3]   ))
    cols.append( fits.Column(name='ERR_H5',         format='D', array=mc_results[:,4]   ))
    cols.append( fits.Column(name='ERR_H6',         format='D', array=mc_results[:,5]   ))

    cols.append( fits.Column(name='FORM_ERR_V' ,    format='D', array=formal_error[:,0] ))
    cols.append( fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1] ))
    cols.append( fits.Column(name='FORM_ERR_H3',    format='D', array=formal_error[:,2] ))
    cols.append( fits.Column(name='FORM_ERR_H4',    format='D', array=formal_error[:,3] ))
    cols.append( fits.Column(name='FORM_ERR_H5',    format='D', array=formal_error[:,4] ))
    cols.append( fits.Column(name='FORM_ERR_H6',    format='D', array=formal_error[:,5] ))
    
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'PPXF_DATA'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_ppxf.fits')
    logging.info("Wrote: "+outfits_ppxf)
    
    
    # ========================
    # SAVE BESTFIT
    outfits_ppxf = outdir+rootname+'_ppxf-bestfit.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_ppxf-bestfit.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with PPXF bestfit
    cols = []
    cols.append( fits.Column(name='BIN_ID',  format='J',    array=ubins                    ))
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit      ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'

    # Table HDU with PPXF logLam
    cols = []
    cols.append( fits.Column(name='BIN_ID', format='J', array=ubins  ))
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM'
     
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_ppxf-bestfit.fits')
    logging.info("Wrote: "+outfits_ppxf)
    
    
    # ========================
    # SAVE GOODPIXELS
    outfits_ppxf = outdir+rootname+'_ppxf-goodpix.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_ppxf-goodpix.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with goodpixels
    cols = []
    cols.append( fits.Column(name='GOODPIX', format='J',           array=goodPixels        ))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, goodpixHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_ppxf-goodpix.fits')
    logging.info("Wrote: "+outfits_ppxf)


    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = outdir+rootname+'_ppxf-optimalTemplates.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_ppxf-optimalTemplates.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='OPTIMAL_TEMPLATES', format=str(optimal_template.shape[1])+'D', array=optimal_template ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'OPTIMAL_TEMPLATES'
    
    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=logLam_template) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM_TEMPLATE'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_ppxf-optimalTemplates.fits')
    logging.info("Wrote: "+outfits)


def runModule_PPXF(PPXF, PARALLEL, configs, dirPath, velscale, LSF_Data, LSF_Templates, outdir, rootname):
    """
    Starts the analysis of the stellar kinematics. Data is read in,
    emission-line contaminated regions are excluded, and pPXF is executed. The
    lambda parameter is computed, results are saved to disk, and the plotting
    routines called. 
    """
    if PPXF == True:
        print("")
        print("\033[0;37m"+" - - - - - Running PPXF! - - - - - "+"\033[0;39m")
        logging.info(" - - - Running PPXF - - - ")

        # Read data from file
        hdu      = fits.open(outdir+rootname+'_VorSpectra.fits')
        bin_data = np.array( hdu[1].data.SPEC.T )
        logLam   = np.array( hdu[2].data.LOGLAM )
        idx_lam  = np.where( np.logical_and( np.exp(logLam) > configs['LMIN_PPXF'], np.exp(logLam) < configs['LMAX_PPXF'] ) )[0]
        bin_data = bin_data[idx_lam,:]
        logLam   = logLam[idx_lam]
        npix     = bin_data.shape[0]
        nbins    = bin_data.shape[1]
        ubins    = np.arange(0, nbins)

        # Prepare templates
        velscale_ratio = 2
        logging.info("Using full spectral library for PPXF")
        templates, lamRange_spmod, logLam_template, ntemplates = util_prepare.prepareSpectralTemplateLibrary\
                ("PPXF", configs, configs['LMIN_PPXF'], configs['LMAX_PPXF'], velscale, velscale_ratio, LSF_Data, LSF_Templates)[:4]
        templates = templates.reshape( (templates.shape[0], ntemplates) )

        # Last preparatory steps
        offset = (logLam_template[0] - logLam[0])*C
        noise  = np.ones((npix,nbins))
        nsims  = configs['MC_PPXF']

        # Initial guesses 
        start = np.zeros((nbins,2))
        if os.path.isfile(outdir+rootname+'_ppxf-guess.fits') == True:
            # Use a different initial guess for different bins, as provided in the *_ppxf-guess.fits file
            guess      = fits.open(outdir+rootname+'_ppxf-guess.fits')[1].data
            start[:,0] = guess.V
            start[:,1] = guess.SIGMA
        else: 
            # Use the same initial guess for all bins, as stated in MasterConfig
            start[:,0] = 0.0
            start[:,1] = configs['SIGMA']

        # Define goodpixels
        goodPixels_ppxf = util_prepare.spectralMasking(outdir, logLam, 'PPXF')

        # Array to store results of ppxf
        ppxf_result        = np.zeros((nbins,6))
        ppxf_bestfit       = np.zeros((nbins,npix))
        optimal_template   = np.zeros((nbins,templates.shape[0]))
        mc_results         = np.zeros((nbins,6))
        formal_error       = np.zeros((nbins,6))
   
        # ====================
        # Run PPXF
        start_time = time.time()
        if PARALLEL == True:
            pipeline.prettyOutput_Running("Running PPXF in parallel mode")
            logging.info("Running PPXF in parallel mode")

            # Create Queues
            inQueue  = Queue()
            outQueue = Queue()
        
            # Create worker processes
            ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
                  for _ in range(configs['NCPU'])]
        
            # Start worker processes
            for p in ps: p.start()
        
            # Fill the queue
            for i in range(nbins):
                inQueue.put( ( templates, bin_data[:,i], noise[:,i], velscale, start[i,:], goodPixels_ppxf,\
                                configs['MOM'], configs['ADEG'], configs['MDEG'], offset, velscale_ratio,\
                                nsims, nbins, i) )
        
            # now get the results with indices
            ppxf_tmp = [outQueue.get() for _ in range(nbins)]
        
            # send stop signal to stop iteration
            for _ in range(configs['NCPU']): inQueue.put('STOP')

            # stop processes
            for p in ps: p.join()
        
            # Get output
            index = np.zeros(nbins)
            for i in range(0, nbins):
                index[i]                        = ppxf_tmp[i][0]
                ppxf_result[i,:configs['MOM']]  = ppxf_tmp[i][1]
                ppxf_bestfit[i,:]               = ppxf_tmp[i][2]
                optimal_template[i,:]           = ppxf_tmp[i][3]
                mc_results[i,:configs['MOM']]   = ppxf_tmp[i][4]
                formal_error[i,:configs['MOM']] = ppxf_tmp[i][5]
            # Sort output
            argidx = np.argsort( index )
            ppxf_result      = ppxf_result[argidx,:]
            ppxf_bestfit     = ppxf_bestfit[argidx,:]
            optimal_template = optimal_template[argidx,:]
            mc_results       = mc_results[argidx,:]
            formal_error     = formal_error[argidx,:]

            pipeline.prettyOutput_Done("Running PPXF in parallel mode", progressbar=True)

        elif PARALLEL == False:
            pipeline.prettyOutput_Running("Running PPXF in serial mode")
            logging.info("Running PPXF in serial mode")
            for i in range(0, nbins):
                ppxf_result[i,:configs['MOM']], ppxf_bestfit[i,:], optimal_template[i,:],\
                  mc_results[i,:configs['MOM']], formal_error[i,:configs['MOM']] = run_ppxf\
                    (templates, bin_data[:,i], noise[:,i], velscale, start[i,:], goodPixels_ppxf,\
                    configs['MOM'], configs['ADEG'], configs['MDEG'], offset, velscale_ratio,\
                    nsims, nbins, i)
            pipeline.prettyOutput_Done("Running PPXF in serial mode", progressbar=True)
        
        print("             Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))
        logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))

        # Check for exceptions which occurred during the analysis
        idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]
        if len(idx_error) != 0:
            pipeline.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
            print("             "+str(idx_error))
            logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
        else:
            print("             "+"There were no problems in the analysis.")
            logging.info("There were no problems in the analysis.")
        print("")

        # Calculate LAMBDA_R
        pipeline.prettyOutput_Running("Calculating Lambda_R")
        logging.info("Calculating Lambda_R")
        lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
        lambda_r = calc_LambdaR( ppxf_result, nbins, outdir, rootname )
        pipeline.prettyOutput_Done("Calculating Lambda_R")

        # Save stellar kinematics to file
        save_ppxf(rootname, outdir, ppxf_result, mc_results, formal_error, lambda_r, ppxf_bestfit, logLam, goodPixels_ppxf, optimal_template, logLam_template, npix, ubins)

        # Do plotting
        try: 
            pipeline.prettyOutput_Running("Producing stellar kinematics maps")
            logging.info("Producing stellar kinematics maps")
            util_plot.plot_maps('PPXF', outdir)
            util_plot_lambdar.plot_maps(outdir)
            pipeline.prettyOutput_Done("Producing stellar kinematics maps")
        except:
            pipeline.prettyOutput_Failed("Producing stellar kinematics maps")
            logging.warning("Failed to produce stellar kinematics maps. Analysis continues!")
            pass

        print("\033[0;37m"+" - - - - - PPXF done! - - - - -"+"\033[0;39m")
        print("")
        logging.info(" - - - PPXF Done - - - \n")


    elif PPXF == False:
        print("")
        print(pipeline.prettyOutput_WarningPrefix()+"Skipping PPXF!")
        print("")
        logging.warning("Skipping PPXF analysis!\n")
