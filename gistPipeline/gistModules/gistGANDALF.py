from astropy.io import fits
import numpy as np
from multiprocessing import Queue, Process

import time
import os
import logging

from gistPipeline.sitePackages.gandalf import gandalf_util as gandalf

from gistPipeline.gistModules import util             as pipeline
from gistPipeline.gistModules import gistPrepare      as util_prepare
from gistPipeline.gistModules import gistPlot_gandalf as util_plot

# PHYSICAL CONSTANTS
C = np.float64(299792.458) # km/s


"""
PURPOSE:
  This module executes the emission-line analysis of the pipeline. Basically, it
  acts as an interface between pipeline and the pyGandALF routine from Sarzi et
  al. 2006 (ui.adsabs.harvard.edu/?#abs/2006MNRAS.366.1151S;
  ???????????????????????????????????????????????) by implementing the
  input/output as well as preparation of data and eventually calling pyGandALF. 
"""


def workerGANDALF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for spectra, error, stellar_kin, templates, logLam_galaxy, logLam_template, emi_file, redshift, velscale, int_disp,\
        reddening, mdeg, for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL\
        in iter(inQueue.get, 'STOP'):

        weights, emission_templates, bestfit, sol, esol = run_gandalf(spectra, error, stellar_kin, templates,\
                logLam_galaxy, logLam_template, emi_file, redshift, velscale, int_disp,\
                reddening, mdeg, for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL)

        outQueue.put(( i, weights, emission_templates, bestfit, sol, esol ))


def run_gandalf(spectrum, error, stellar_kin, templates, logLam_galaxy, logLam_template, emi_file, redshift,\
                velscale, int_disp, reddening, mdeg, for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL):
    """
    Calls the pyGandALF routine from Sarzi et al. 2006 (ui.adsabs.harvard.edu/?#abs/2006MNRAS.366.1151S; 
    ???????????????????????????????????????????????)
    """
    pipeline.printProgress( i, nbins, barLength=50 )

    # Leave this hardcoded here!
    plot  = False;  degree = -1
    quiet = True ;  log10  = False

    if reddening == None: 
        # Use multiplicative polynomials if REDDENING is not used
        mdegree = mdeg
    else: 
        # Do not use multiplicative polynomials and REDDENING together
        mdegree = 0

    try:
        # Get goodpixels and emission_setup
        goodpixels, emission_setup = getGoodpixelsEmissionSetup\
                ('GANDALF', emi_file, redshift, velscale, logLam_galaxy, logLam_template, npix, outdir)
    
        ## Uncomment the following lines to use stellar velocities as initial guess on GANDALF fit
        #for itm in np.arange(len(emission_setup)):
        #    emission_setup[itm].v = stellar_kin[0] 
    
        # Run GANDALF
        weights, emission_templates, bestfit, sol, esol = gandalf.gandalf\
                ( templates, spectrum, error, velscale, stellar_kin, emission_setup, logLam_galaxy[0], \
                  logLam_galaxy[1]-logLam_galaxy[0], goodpixels, degree, mdegree, int_disp, plot, quiet, log10, reddening,\
                  logLam_template[0], for_errors, velscale_ratio, offset)
    
        return( [weights, emission_templates, bestfit, sol, esol] )
    
    except:
        return( [-1, -1, -1, -1, -1] )


def save_gandalf(GANDALF, LEVEL, outdir, rootname, emission_setup, idx_l, sol, esol, nlines, sol_gas_AoN,\
        stellar_kin, offset, optimal_template, logLam_template, logLam_galaxy, goodpixels, cleaned_spectrum,\
        nweights, emission_weights, for_errors, npix, n_templates, bestfit, emissionSpectra, residuals, reddening):
    """ Saves all results to disk. """
    # ========================
    # PREPARE OUTPUT
    # Convert emission_setup
    iis     = [] ; aas     = []
    names   = [] ; vs      = []            
    lambdas = [] ; ss      = []
    actions = [] ; ffits   = []
    kinds   = [] ; aons    = []
    for ii in idx_l:               
        iis.append(         emission_setup[ii].i       ) 
        names.append(   str(emission_setup[ii].name)   )
        lambdas.append(     emission_setup[ii]._lambda )
        actions.append( str(emission_setup[ii].action) )
        kinds.append(   str(emission_setup[ii].kind)   )
        aas.append(         emission_setup[ii].a       )
        vs.append(          emission_setup[ii].v       )
        ss.append(          emission_setup[ii].s       )
        ffits.append(   str(emission_setup[ii].fit)    )
        aons.append(        emission_setup[ii].aon     )
    
    # Extract solutions
    sol_gas_F     = np.array( sol[:,np.arange(len(idx_l))*4+0] )
    sol_gas_A     = sol[:,np.arange(len(idx_l))*4+1]
    sol_gas_V     = sol[:,np.arange(len(idx_l))*4+2]
    sol_gas_S     = sol[:,np.arange(len(idx_l))*4+3]
    sol_EBmV_MDEG = sol[:,len(idx_l)*4:None]
    if for_errors:  # TODO: THIS SHOULD ACCOUNT FOR MDEG/EBMV ONCE ERROR ESTIMATION IS IMPLEMENTED
        esol_gas_F = esol[:,np.arange(len(idx_l))*4+0]
        esol_gas_A = esol[:,np.arange(len(idx_l))*4+1]
        esol_gas_V = esol[:,np.arange(len(idx_l))*4+2]
        esol_gas_S = esol[:,np.arange(len(idx_l))*4+3]
        esol_EBmV  = esol[:,len(idx_l)*4:None]
   

    # ========================
    # SAVE RESULTS
    outfits = outdir+rootname+'_gandalf_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf_'+LEVEL+'.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with emission_setup data
    cols = []
    cols.append( fits.Column(name='LINE_ID',      format='D',   array=np.arange(nlines)  ))
    cols.append( fits.Column(name='i',            format='D',   array=np.stack(iis)      ))
    cols.append( fits.Column(name='name',         format='15A', array=np.stack(names)    ))
    cols.append( fits.Column(name='_lambda',      format='D',   array=np.stack(lambdas)  ))
    cols.append( fits.Column(name='action',       format='15A', array=np.stack(actions)  ))
    cols.append( fits.Column(name='kind',         format='15A', array=np.stack(kinds))   )
    cols.append( fits.Column(name='a',            format='D',   array=np.stack(aas))     )
    cols.append( fits.Column(name='v',            format='D',   array=np.stack(vs))      )
    cols.append( fits.Column(name='s',            format='D',   array=np.stack(ss))      )
    cols.append( fits.Column(name='fit',          format='15A', array=np.stack(ffits))   )
    cols.append( fits.Column(name='aon',          format='D',   array=np.stack(aons))    )
    emission_setup_HDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    emission_setup_HDU.name = 'EMISSION_SETUP'
    
    # Extension 2: Table HDU with GANDALF output data
    cols = []
    cols.append( fits.Column(name='FLUX',         format=str(nlines)+'D',  array=np.array(sol_gas_F))     )
    cols.append( fits.Column(name='AMPL',         format=str(nlines)+'D',  array=np.array(sol_gas_A))     )
    cols.append( fits.Column(name='V',            format=str(nlines)+'D',  array=np.array(sol_gas_V))     )
    cols.append( fits.Column(name='SIGMA',        format=str(nlines)+'D',  array=np.array(sol_gas_S))     )
    cols.append( fits.Column(name='AON',          format=str(nlines)+'D',  array=np.array(sol_gas_AoN))   )
    if reddening != None: 
        cols.append( fits.Column(name='EBMV',     format=str(len(reddening))+'D', array=np.array(sol_EBmV_MDEG)) )
    if for_errors: # TODO: THIS SHOULD ACCOUNT FOR MDEG/EBMV ONCE ERROR ESTIMATION IS IMPLEMENTED
        cols.append( fits.Column(name='ERR_FLUX',    format=str(nlines)+'D', array=np.array(esol_gas_F))    )
        cols.append( fits.Column(name='ERR_AMPL',    format=str(nlines)+'D', array=np.array(esol_gas_A))    )
        cols.append( fits.Column(name='ERR_V',       format=str(nlines)+'D', array=np.array(esol_gas_V))    )
        cols.append( fits.Column(name='ERR_SIGMA',   format=str(nlines)+'D', array=np.array(esol_gas_S))    )
        cols.append( fits.Column(name='ERR_EBMV',    format='2D',            array=np.array(esol_EBmV))     )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'GANDALF'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, emission_setup_HDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)


    # ========================
    # SAVE BESTFIT
    outfits = outdir+rootname+'_gandalf-bestfit_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-bestfit_'+LEVEL+'.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(bestfit.shape[1])+'D', array=bestfit ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-bestfit_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)
   

    # ========================
    # SAVE EMISSION
    outfits = outdir+rootname+'_gandalf-emission_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-emission_'+LEVEL+'.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='EMISSION', format=str(emissionSpectra.shape[1])+'D', array=emissionSpectra ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'EMISSION'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-emission_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)


    # ========================
    # SAVE RESIDUALS
    outfits = outdir+rootname+'_gandalf-residuals_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-residuals_'+LEVEL+'.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='RESIDUALS', format=str(residuals.shape[1])+'D', array=residuals ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'RESIDUALS'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-residuals_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)


    # ========================
    # SAVE GOODPIXELS
    outfits = outdir+rootname+'_gandalf-goodpix_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-goodpix_'+LEVEL+'.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='GOODPIX', format='J', array=goodpixels ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'GOODPIX'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-goodpix_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)
    
    
    # ========================
    # SAVE CLEANED SPECTRA
    outfits = outdir+rootname+'_gandalf-cleaned_'+LEVEL+'.fits'
    pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-cleaned_'+LEVEL+'.fits')
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='SPEC', format=str(npix)+'D', array=cleaned_spectrum ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'CLEANED_SPECTRA'

    # Extension 2: Table HDU with logLam
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam_galaxy) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM'
     
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)
    
    pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-cleaned_'+LEVEL+'.fits')
    logging.info("Wrote: "+outfits)
   

    # ========================
    # SAVE WEIGHTS
    if ( GANDALF in [1,2] )  or  ( GANDALF == 3 and LEVEL == 'BIN' ): 
        outfits = outdir+rootname+'_gandalf-weights_'+LEVEL+'.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-weights_'+LEVEL+'.fits')
        
        # Primary HDU
        priHDU = fits.PrimaryHDU()
        
        # Extension 1: Table HDU with normalized weights
        cols = []
        cols.append( fits.Column(name='NWEIGHTS', format=str(n_templates)+'D', array=nweights ) )
        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = 'NWEIGHTS'
    
        # Extension 2: Table HDU with weights of emission-lines
        cols = []
        cols.append( fits.Column(name='EWEIGHTS', format=str(emission_weights.shape[1])+'D', array=emission_weights ) )
        eweightsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        eweightsHDU.name = 'EWEIGHTS'
         
        # Create HDU list and write to file
        HDUList = fits.HDUList([priHDU, dataHDU, eweightsHDU])
        HDUList.writeto(outfits, overwrite=True)
        
        pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-weights_'+LEVEL+'.fits')
        logging.info("Wrote: "+outfits)


    # ========================
    # SAVE OPTIMAL TEMPLATE
    if LEVEL == 'BIN':
        outfits = outdir+rootname+'_gandalf-optimalTemplate_'+LEVEL+'.fits'
        pipeline.prettyOutput_Running("Writing: "+rootname+'_gandalf-optimalTemplate_'+LEVEL+'.fits')
        
        # Primary HDU
        priHDU = fits.PrimaryHDU()
        
        # Extension 1: Table HDU with optimal templates
        cols = []
        cols.append( fits.Column(name='OPTIMAL_TEMPLATE', format=str(optimal_template.shape[1])+'D', array=optimal_template ) )
        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = 'OPTIMAL_TEMPLATE'
    
        # Extension 2: Table HDU with logLam_templates
        cols = []
        cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=logLam_template) )
        logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        logLamHDU.name = 'LOGLAM_TEMPLATE'
        
        # Create HDU list and write to file
        HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
        HDUList.writeto(outfits, overwrite=True)
        
        pipeline.prettyOutput_Done("Writing: "+rootname+'_gandalf-optimalTemplate_'+LEVEL+'.fits')
        logging.info("Wrote: "+outfits)


def getGoodpixelsEmissionSetup(MODULE, emi_file, redshift, velscale, logLam_galaxy, logLam_template, npix, outdir):
    """
    Reads information about the emission-lines from the emissionLines.config
    file, in order to 
     * determine which spectral regions are masked in the GandALF fit
     * create a structure which states details on the intended emission-line
       fits
    """
    # Read in emission-line setup file
    eml_file   = outdir+emi_file
    eml_i      = np.genfromtxt(eml_file, dtype='int',   usecols = (0), comments ='#')
    eml_name   = np.genfromtxt(eml_file, dtype='str',   usecols = (1), comments ='#')
    eml_lambda = np.genfromtxt(eml_file, dtype='float', usecols = (2), comments ='#')
    eml_action = np.genfromtxt(eml_file, dtype='str',   usecols = (3), comments ='#')
    eml_kind   = np.genfromtxt(eml_file, dtype='str',   usecols = (4), comments ='#')
    eml_a      = np.genfromtxt(eml_file, dtype='float', usecols = (5), comments ='#')
    eml_v      = np.genfromtxt(eml_file, dtype='int',   usecols = (6), comments ='#')
    eml_s      = np.genfromtxt(eml_file, dtype='int',   usecols = (7), comments ='#')
    eml_fit    = np.genfromtxt(eml_file, dtype='str',   usecols = (8), comments ='#')
    eml_aon    = np.genfromtxt(eml_file, dtype='int',   usecols = (9), comments ='#')

    emission_setup = gandalf.load_emission_setup_new\
            ( np.vstack((eml_i, eml_name, eml_lambda, eml_action, eml_kind, eml_a, eml_v, eml_s, eml_fit, eml_aon)) )

    # Create goodpixels and final emission line setup
    goodpixels, emission_setup = gandalf.mask_emission_lines\
            (npix, redshift, emission_setup, velscale, logLam_galaxy[0], (logLam_galaxy[1]-logLam_galaxy[0]), None, None, 0)    

    # Prepare emission_setup structure for GANDALF, which should only deal with the lines we fit
    i_f = []
    for itm in np.arange(len(emission_setup)): 
        if (emission_setup[itm].action != 'f'): 
            i_f.append(itm)
    emission_setup = [v for i,v in enumerate(emission_setup) if i not in frozenset(i_f)]

    return(goodpixels, emission_setup)


def runModule_GANDALF(GANDALF, PARALLEL, configs, velscale, LSF_Data, LSF_Templates, outdir, rootname):
    """
    Starts the emission-line analysis. Input data is read from file on either
    bin or spaxel level. The LSF is loaded and spectra can be dereddened for
    Galactic extinction in the direction of the target. Spectral pixels and
    emission-lines considered in the fit are determined. After the GandALF fit,
    emission-subtracted spectral are calculated. Results are saved to file and
    the plotting routines executed.
    """
    # Check what we have to do
    if GANDALF == 0:
        print("")
        print(pipeline.prettyOutput_WarningPrefix()+"Skipping GANDALF!")
        print("")
        logging.warning("Skipping GANDALF analysis!\n")
        return(None)
    #
    elif GANDALF == 1:
        LEVEL = 'BIN'
        print("\033[0;37m"+" - - - - - Running GANDALF: BIN - - - - - "+"\033[0;39m")
        logging.info(" - - - Running GANDALF: BIN - - - ")
    #
    elif GANDALF == 2: 
        LEVEL = 'SPAXEL'
        print("\033[0;37m"+" - - - - - Running GANDALF: SPAXEL - - - - - "+"\033[0;39m")
        logging.info(" - - - Running GANDALF: SPAXEL - - - ")
    #
    elif GANDALF == 3: 
        if os.path.isfile(outdir+rootname+'_gandalf_BIN.fits') == True: 
            LEVEL = 'SPAXEL'
            print("\033[0;37m"+" - - - - - Running GANDALF: SPAXEL - - - - - "+"\033[0;39m")
            logging.info(" - - - Running GANDALF: SPAXEL - - - ")
        else: 
            LEVEL = 'BIN'
            print("\033[0;37m"+" - - - - - Running GANDALF: BIN - - - - - "+"\033[0;39m")
            logging.info(" - - - Running GANDALF: BIN - - - ")


    # Oversample the templates by a factor of two
    velscale_ratio = 2
       
    # Read data if we run on BIN level
    if LEVEL == 'BIN':
        # Read spectra from file
        hdu           = fits.open(outdir+rootname+'_VorSpectra.fits')
        spectra       = np.array( hdu[1].data.SPEC.T )
        logLam_galaxy = np.array( hdu[2].data.LOGLAM )
        npix          = spectra.shape[0]
        nbins         = spectra.shape[1]
    
        # Prepare templates
        logging.info("Using full spectral library for GANDALF on BIN level")
        templates, lamRange_spmod, logLam_template = \
                util_prepare.prepare_sp_templates(configs, velscale, velscale_ratio, LSF_Data, LSF_Templates)
        n_templates  = templates.shape[1]
        offset       = (logLam_template[0] - logLam_galaxy[0])*C # km/s
        error        = np.ones((npix,nbins))
   
        # Read stellar kinematics from file
        ppxf = fits.open(outdir+rootname+'_ppxf.fits')[1].data
        stellar_kin      = np.stack((ppxf.V, ppxf.SIGMA, ppxf.H3, ppxf.H4, ppxf.H5, ppxf.H6), axis=1)
        stellar_kin[:,4] = 0.
        stellar_kin[:,5] = 0.
    
        # Rename to keep the code clean
        for_errors = configs['FOR_ERRORS']


    # Read data if we run on SPAXEL level
    elif LEVEL == 'SPAXEL':
        # Read spectra from file
        hdu           = fits.open(outdir+rootname+'_AllSpectra.fits')
        spectra       = np.array( hdu[1].data.SPEC.T )
        logLam_galaxy = np.array( hdu[2].data.LOGLAM )
        npix          = spectra.shape[0]
        nbins         = spectra.shape[1]

        # Prepare templates
        if GANDALF == 2: 
            logging.info("Using full spectral library for GANDALF on SPAXEL level")
            templates, lamRange_spmod, logLam_template = \
                    util_prepare.prepare_sp_templates(configs, velscale, velscale_ratio, LSF_Data, LSF_Templates)
            n_templates = templates.shape[1]
        if GANDALF == 3: 
            logging.info("Using previously extracted optimal templates from the GANDALF BIN level on SPAXEL level")
            hdu             = fits.open(outdir+rootname+'_gandalf-optimalTemplate_BIN.fits')
            templates       = np.array( hdu[1].data.OPTIMAL_TEMPLATE.T )
            logLam_template = np.array( hdu[2].data.LOGLAM_TEMPLATE    )
            n_templates     = 1
        offset      = (logLam_template[0] - logLam_galaxy[0])*C # km/s
        error       = np.ones((npix,nbins))

        # Read stellar kinematics from file
        ppxf             = fits.open(outdir+rootname+'_ppxf.fits')[1].data
        stellar_kin      = np.stack((ppxf.V, ppxf.SIGMA, ppxf.H3, ppxf.H4, ppxf.H5, ppxf.H6), axis=1)
        stellar_kin[:,4] = 0.
        stellar_kin[:,5] = 0.

        # Read bintable
        bintable    = fits.open(outdir+rootname+'_table.fits')[1].data
        binNum_long = np.array( bintable.BIN_ID )
        ubins       = np.unique( np.abs(binNum_long) )

        # Convert stellar kinematics to long version
        stellar_kin_long = np.zeros( (len(binNum_long), stellar_kin.shape[1]) )
        if n_templates == 1: templates_long = np.zeros( (len(binNum_long), templates.shape[0]))
        for i in range( 0, len(ubins) ):
            idx = np.where( ubins[i] == np.abs(binNum_long) )[0]
            stellar_kin_long[idx,:] = stellar_kin[i,:]
            if n_templates == 1: templates_long[idx,:] = templates[:,i]
        stellar_kin = stellar_kin_long
        if n_templates == 1: templates = templates_long

        # Rename to keep the code clean
        for_errors = configs['FOR_ERRORS']

    # LSF function for emission templates
    int_disp      = np.zeros((2,npix))
    int_disp[0,:] = np.exp( logLam_galaxy )
    int_disp[1,:] = C * ( np.array(LSF_Data( np.exp(logLam_galaxy) )) / 2.355 / np.exp( logLam_galaxy ) )

    # Deredden the spectra for the Galactic extinction in the direction of the target
    if (configs['EBmV'] != None):
        dereddening_attenuation = gandalf.dust_calzetti\
                (logLam_galaxy[0], logLam_galaxy[1]-logLam_galaxy[0], npix, -1*configs['EBmV'], 0.0, 0)
        for i in range( spectra.shape[1] ): 
            spectra[:,i] = spectra[:,i]*dereddening_attenuation

    # Get goodpixels and emission_setup
    goodpixels, emission_setup = getGoodpixelsEmissionSetup('GANDALF', configs['EMI_FILE'], \
            0.0, velscale, logLam_galaxy, logLam_template, npix, outdir)

    # Setup output arrays
    nlines = 0
    for itm in np.arange( len(emission_setup) ):
        if emission_setup[itm].action == 'f' and emission_setup[itm].kind == 'l': nlines += 1

    if configs['REDDENING'] == None: 
        reddening_length_sol  = configs['MDEG']
        reddening_length_esol = 0 
    else: 
        reddening_length_sol  = len(configs['REDDENING'])
        reddening_length_esol = len(configs['REDDENING'])

    weights            = np.zeros((nbins, n_templates+nlines))
    emission_templates = np.zeros((nbins, nlines, npix))
    bestfit            = np.zeros((nbins, npix))
    sol                = np.zeros((nbins, nlines*4 + reddening_length_sol ))
    esol               = np.zeros((nbins, nlines*4 + reddening_length_esol))

    nweights           = np.zeros((nbins, n_templates))
    optimal_template   = np.zeros((nbins, templates.shape[0]))
    sol_gas_AoN        = np.zeros((nbins, nlines))
    cleaned_spectrum   = np.zeros((nbins, npix))

    # ========================
    # Run GANDALF
    start_time = time.time()

    if PARALLEL == True:
        pipeline.prettyOutput_Running("Running GANDALF in parallel mode")
        logging.info("Running GANDALF in parallel mode")
       
        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()
    
        # Create worker processes
        ps = [Process(target=workerGANDALF, args=(inQueue, outQueue))
              for _ in range(configs['NCPU'])]
    
        # Start worker processes
        for p in ps: p.start()
    
        # Fill the queue
        if n_templates > 1: 
            for i in range(nbins):
                inQueue.put( ( spectra[:,i], error[:,i], stellar_kin[i,:], templates, logLam_galaxy, logLam_template, \
                               configs['EMI_FILE'], 0.0, velscale, int_disp, configs['REDDENING'], configs['MDEG'],\
                               for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL ) )
        elif n_templates == 1: 
            for i in range(nbins):
                inQueue.put( ( spectra[:,i], error[:,i], stellar_kin[i,:], templates[[i],:].T, logLam_galaxy, logLam_template, \
                               configs['EMI_FILE'], 0.0, velscale, int_disp, configs['REDDENING'], configs['MDEG'],\
                               for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL ) )
     
        # now get the results with indices
        gandalf_tmp = [outQueue.get() for _ in range(nbins)]
    
        # send stop signal to stop iteration
        for _ in range(configs['NCPU']): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()

        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]                  = gandalf_tmp[i][0]
            weights[i,:]              = gandalf_tmp[i][1]
            emission_templates[i,:,:] = gandalf_tmp[i][2]
            bestfit[i,:]              = gandalf_tmp[i][3]
            sol[i,:]                  = gandalf_tmp[i][4]
            esol[i,:]                 = gandalf_tmp[i][5]
        # Sort output
        argidx = np.argsort( index )
        weights            = weights[argidx,:]
        emission_templates = emission_templates[argidx,:,:]
        bestfit            = bestfit[argidx,:]
        sol                = sol[argidx,:]
        esol               = esol[argidx,:]
        
        pipeline.prettyOutput_Done("Running PPXF in parallel mode", progressbar=True)

    elif PARALLEL == False:
        pipeline.prettyOutput_Running("Running GANDALF in serial mode")
        logging.info("Running GANDALF in serial mode")
        if n_templates > 1: 
            for i in range(0, nbins):
                weights[i,:], emission_templates[i,:,:], bestfit[i,:], sol[i,:], esol[i,:] = run_gandalf\
                  (spectra[:,i], error[:,i], stellar_kin[i,:], templates, logLam_galaxy, logLam_template, \
                  configs['EMI_FILE'], 0.0, velscale, int_disp, configs['REDDENING'], configs['MDEG'],\
                  for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL )
        elif n_templates == 1: 
            for i in range(0, nbins):
                weights[i,:], emission_templates[i,:,:], bestfit[i,:], sol[i,:], esol[i,:] = run_gandalf\
                  (spectra[:,i], error[:,i], stellar_kin[i,:], templates[[i],:].T, logLam_galaxy, logLam_template, \
                  configs['EMI_FILE'], 0.0, velscale, int_disp, configs['REDDENING'], configs['MDEG'],\
                  for_errors, offset, velscale_ratio, i, nbins, npix, outdir, LEVEL )

        pipeline.prettyOutput_Done("Running GANDALF in serial mode", progressbar=True)

    print("             Running GANDALF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))
    logging.info("Running GANDALF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, configs['NCPU']))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( bestfit[:,0] == -1 )[0]
    if len(idx_error) != 0:
        pipeline.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Get AoN thresholds from emission_setup
    AoN_thresholds = np.zeros( nlines )
    o = 0
    for itm in np.arange( len(emission_setup) ):
        if emission_setup[itm].action == 'f' and emission_setup[itm].kind == 'l':
            AoN_thresholds[o] = emission_setup[itm].aon
            o = o + 1

    # Get optimal template and calculate emission-subtracted spectra
    # TODO: This should be done within the parallel loop
    idx_l = gandalf.where_eq(emission_setup, 'kind', 'l')
    emission_templates = np.transpose( emission_templates, (0,2,1) )
    for i in range( 0, nbins ):

        if ( GANDALF in [1,2] )  or  ( GANDALF == 3  and  LEVEL == 'BIN' ): 
            # Make the unconvolved optimal stellar template
            nweights[i,:] = weights[i,:n_templates] / np.sum(weights[i,:n_templates])
            for j in range(0,n_templates): 
                optimal_template[i,:] = optimal_template[i,:] + templates[:,j]*nweights[i,j]

        # Calculate emission-subtracted spectra using a AoN threshold
        sol_gas_A      = sol[i, np.arange(len(idx_l))*4+1]  # Array of solutions for amplitudes
        sol_gas_AoN[i,:], cleaned_spectrum[i,:] = gandalf.remouve_detected_emission\
            (spectra[:,i], bestfit[i,:], emission_templates[i,:,:], sol_gas_A, AoN_thresholds, None)

    # Calculate the best fitting emission and the residuals
    emissionSpectrum          = np.sum( emission_templates, axis=2 )
    emissionSubtractedBestfit = bestfit - emissionSpectrum
    residuals                 = spectra - emissionSubtractedBestfit.T

    # Save results to file
    save_gandalf(GANDALF, LEVEL, outdir, rootname, emission_setup, idx_l, sol, esol, nlines, sol_gas_AoN,\
            stellar_kin, offset, optimal_template, logLam_template, logLam_galaxy, goodpixels, cleaned_spectrum,\
            nweights, weights[:,n_templates:], for_errors, npix, n_templates, bestfit, emissionSpectrum, residuals,\
            configs['REDDENING'])

    # Call plotting routines
    try: 
        pipeline.prettyOutput_Running("Producing gas kinematics maps")
        logging.info("Producing gas kinematics maps")
        util_plot.plot_maps(outdir, LEVEL, True)
        pipeline.prettyOutput_Done("Producing gas kinematics maps")
    except:
        pipeline.prettyOutput_Failed("Producing gas kinematics maps")
        logging.warning("Failed to produce gas kinematics maps. Analysis continues!")
        pass

    print("\033[0;37m"+" - - - - - GANDALF Done: "+LEVEL+" - - - - -"+"\033[0;39m")
    print("")
    logging.info(" - - - GANDALF Done: "+LEVEL+" - - - \n")

    # Restart GANDALF if a SPAXEL level run based on a previous BIN level run is intended
    if GANDALF == 3  and  LEVEL == 'BIN': 
        runModule_GANDALF(GANDALF, PARALLEL, configs, velscale, LSF_Data, LSF_Templates, outdir, rootname)
