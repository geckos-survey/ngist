import numpy as np
from astropy.io import fits

import os
import sys
import shutil
import multiprocessing
import logging

from gistPipeline.gistModules import util as pipeline


"""
PURPOSE: 
  This file contains a collection of functions necessary to initialise the
  pipeline. These are in particular, the reading and saving of the configuration
  parameters, basic checks on their plausibility, and the determination of the
  analysis modules that will be executed. Functions to print (to stdout) an
  overview of those parameters at startup are included as well. 
"""


def setup_configs(galnumber, dirPath):
    """
    Reads the main configuration file and saves all parameters in the configs
    structure.
    """
    configs = readConfig(dirPath.configFile, galnumber)
    
    rootname = str(configs['RUN_NAME']).split('_')[0]
    outdir   = dirPath.outputDir+str(configs['RUN_NAME'])+'/'

    if os.path.isfile(dirPath.inputDir+rootname+'.txt') == True:
        datafile = dirPath.inputDir+rootname+'.txt'
    else:
        datafile = dirPath.inputDir+rootname+'.fits'

    if os.path.isdir(outdir) == False: 
        os.mkdir(outdir)
    
    # MAIN SWITCHES
    DEBUG                     = bool(int( configs['DEBUG']        ))
    configs['VORONOI']        = int(      configs['VORONOI']       )
    configs['GANDALF']        = int(      configs['GANDALF']       )
    configs['SFH']            = int(      configs['SFH']           )
    configs['LINE_STRENGTH']  = int(      configs['LINE_STRENGTH'] )
    configs['IFU']            = str(           configs['IFU']      )
    PARALLEL                  = bool(int( configs['PARALLEL']     ))
    configs['NCPU']           = int(      configs['NCPU']          )

    # GENERAL SETTINGS
    configs['LMIN_SNR']       = float(    configs['LMIN_SNR']      )
    configs['LMAX_SNR']       = float(    configs['LMAX_SNR']      )
    configs['LMIN_PPXF']      = float(    configs['LMIN_PPXF']     )
    configs['LMAX_PPXF']      = float(    configs['LMAX_PPXF']     )
    configs['LMIN_GANDALF']   = float(    configs['LMIN_GANDALF']  )
    configs['LMAX_GANDALF']   = float(    configs['LMAX_GANDALF']  )
    configs['LMIN_SFH']       = float(    configs['LMIN_SFH']      )
    configs['LMAX_SFH']       = float(    configs['LMAX_SFH']      )
    configs['ORIGIN']         = [ float(configs['ORIGIN'].split(',')[0]) , float(configs['ORIGIN'].split(',')[1]) ]
    configs['REDSHIFT']       = float(    configs['REDSHIFT']      )
    configs['SIGMA']          = float(    configs['SIGMA']         )
    configs['TARGET_SNR']     = float(    configs['TARGET_SNR']    )
    configs['MIN_SNR']        = float(    configs['MIN_SNR']       )
    configs['COVAR_VOR']      = float(    configs['COVAR_VOR']     )
    configs['SSP_LIB']        = dirPath.spTempDir + str(configs['SSP_LIB'])
    configs['NORM_TEMP']      = str( configs['NORM_TEMP'] )
    
    # PPXF
    configs['MOM']            = int(      configs['MOM']           )
    configs['ADEG']           = int(      configs['ADEG']          )
    configs['MDEG']           = int(      configs['MDEG']          )
    configs['MC_PPXF']        = int(      configs['MC_PPXF']       )

    # GANDALF
    configs['EMI_FILE']       = "emissionLines.config"
    if configs['REDDENING'] == 'None': 
        configs['REDDENING'] = None
    else: 
        reddening = []
        for i in range(0, len( configs['REDDENING'].split(',') )):
            reddening.append( float(configs['REDDENING'].split(',')[i]) )
        configs['REDDENING']      = reddening
    configs['FOR_ERRORS']     = int(      configs['FOR_ERRORS']    )
    if    configs['EBmV'] == 'None': configs['EBmV'] = None
    else: configs['EBmV'] = float(configs['EBmV'])

    # SFH
    configs['REGUL_ERR']      = float(    configs['REGUL_ERR']     )
    configs['FIXED']          = int(      configs['FIXED']         )
    configs['NOISE']          = float(    configs['NOISE']         )

    # LINE STRENGTH
    configs['LS_FILE']        = "lsBands.config"
    configs['CONV_COR']       = float(    configs['CONV_COR']      )
    configs['MC_LS']          = int(      configs['MC_LS']         )
    configs['NWALKER']        = int(      configs['NWALKER']       )
    configs['NCHAIN']         = int(      configs['NCHAIN']        )

    return(PARALLEL, DEBUG, configs, datafile, rootname, outdir)


def parameter_checks(datafile, configs, dirPath, rootname, outdir):
    """
    Performs basic checks on the parameters from the configuration file. If
    necessary, individual analysis modules or the analysis of the entire galaxy
    are skipped. 
    """
    SKIP_GALAXY = False

    if len( configs['RUN_NAME'].split('_') ) != 2: 
        message = "RUN_NAME must be in the form [Galaxyname]_[RunName], e.g. NGC0000_Example and thus must contain only one '_'. This is a fatal error! Galaxy is skipped!"
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.error(message)
        SKIP_GALAXY = True

    # Check that all configuration and input files are in place
    if os.path.isfile(datafile) == False:
        message = datafile+' does not exist. This is a fatal error! Galaxy is skipped!'
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.error(message)
        SKIP_GALAXY = True

    #
    if os.path.isfile(outdir+configs['EMI_FILE']) == False  and  \
       os.path.isfile(dirPath.configDir+configs['EMI_FILE']) == False  and  \
       configs['GANDALF'] in [1,2,3]:
        message = "No "+configs['EMI_FILE']+" provided in the output or configFiles/ directory. GANDALF will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['GANDALF'] = 0
    elif os.path.isfile(outdir+configs['EMI_FILE']) == False  and  \
         os.path.isfile(dirPath.configDir+configs['EMI_FILE']) == True  and  \
         configs['GANDALF'] in [1,2,3]:
        shutil.copyfile( dirPath.configDir+configs['EMI_FILE'], outdir+configs['EMI_FILE'] )
    if os.path.isfile(outdir+configs['LS_FILE']) == False  and  \
       os.path.isfile(dirPath.configDir+configs['LS_FILE']) == False  and  \
       configs['LINE_STRENGTH'] in [1,2]: 
        message = "No "+configs['LS_FILE']+" provided in the output or configFiles/ directory. LINE_STRENGTH will be disabled. Continue"
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['LINE_STRENGTH'] = 0
    elif os.path.isfile(outdir+configs['LS_FILE']) == False  and  \
         os.path.isfile(dirPath.configDir+configs['LS_FILE']) == True  and  \
         configs['LINE_STRENGTH'] in [1,2]: 
        shutil.copyfile( dirPath.configDir+configs['LS_FILE'], outdir+configs['LS_FILE'] )
    if os.path.isfile(outdir+'spectralMasking_PPXF.config') == False  and  \
       os.path.isfile(dirPath.configDir+'spectralMasking_PPXF.config') == False: 
       message = "No spectralMasking_PPXF.config provided in the output or configFiles/ directory. Galaxy is skipped."
       print(pipeline.prettyOutput_FailedPrefix()+message)
       logging.warning(message)
       SKIP_GALAXY = True
    elif os.path.isfile(outdir+'spectralMasking_PPXF.config') == False  and  \
         os.path.isfile(dirPath.configDir+'spectralMasking_PPXF.config') == True: 
        shutil.copyfile( dirPath.configDir+'spectralMasking_PPXF.config', outdir+'spectralMasking_PPXF.config' )
    #
    if os.path.isfile(outdir+'spectralMasking_SFH.config') == False  and  \
       os.path.isfile(dirPath.configDir+'spectralMasking_SFH.config') == False  and  \
       configs['SFH'] == 1: 
       message = "No spectralMasking_SFH.config provided in the output or configFiles/ directory. Galaxy is skipped."
       print(pipeline.prettyOutput_FailedPrefix()+message)
       logging.warning(message)
       SKIP_GALAXY = True
    elif os.path.isfile(outdir+'spectralMasking_SFH.config') == False  and  \
         os.path.isfile(dirPath.configDir+'spectralMasking_SFH.config') == True  and  \
         configs['SFH'] == 1: 
        shutil.copyfile( dirPath.configDir+'spectralMasking_SFH.config', outdir+'spectralMasking_SFH.config' )
        #
    if os.path.isfile( dirPath.configDir+'LSF-Config_'+configs['IFU'] ) == False:
        message = "Data LSF configuration file 'LSF-Config_"+configs['IFU']+"' not found. Galaxy is skipped!"
        logging.error(message)
        print(pipeline.prettyOutput_FailedPrefix()+message)
        SKIP_GALAXY = True
    if os.path.isfile( dirPath.configDir+'LSF-Config_'+configs['SSP_LIB'].split('/')[-2] ) == False:
        message = "Template LSF configuration file 'LSF-Config_"+configs['SSP_LIB'].split('/')[-2]+"' not found. Galaxy is skipped!"
        logging.error(message)
        print(pipeline.prettyOutput_FailedPrefix()+message)
        SKIP_GALAXY = True

    #
    if configs['VORONOI'] not in [0,1]:
        message = "VORONOI has to be either 0 (No) or 1 (Yes). Voronoi-binning will be enabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['VORONOI'] = 1
    if configs['GANDALF'] not in [0,1,2,3]:
        message = "GANDALF has to be either 0 (Off), 1 (BIN level), 2 (SPAXEL level), or 3 (SPAXEL level based on BIN level). GANDALF will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['GANDALF'] = 0
    if configs['FOR_ERRORS'] == 1:
        message = "The error calculation in GANDALF is only compatible with Python=<3.5.5 while pPXF works only with Python=>3.6. In lack of an easy solution for this problem, the error calculation of GANDALF has been turned off. It can be activated manually in sitePackages/gandalf/gandalf_util.py and then used together with Python=<3.5.5. An upgrade will be provided as soon as possible. Sorry :("
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['FOR_ERRORS'] = 0
    if configs['SFH'] not in [0,1]:
        message = "SFH has to be either 0 (No) or 1 (Yes). SFH will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['SFH'] = 0
    if configs['LINE_STRENGTH'] not in [0,1,2]:
        message = "LINE_STRENGTH has to be either 0 (Off), 1 (LS) or 2 (LS + SSP). LINE_STRENGTH will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['LINE_STRENGTH'] = 0
    #
    if configs['NCPU'] > multiprocessing.cpu_count():
        message = "The chosen number of CPU's seems to be higher than the number of cores in the system! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)

    if configs['LMIN_SNR'] > configs['LMAX_SNR']:
        message = "The given minimum wavelength LMIN_SNR is larger than the maximum wavelength LMAX_SNR. I will swap them! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        lmin = configs['LMAX_SNR']
        lmax = configs['LMIN_SNR']
        configs['LMIN_SNR'] = lmin
        configs['LMAX_SNR'] = lmax
    if configs['LMIN_PPXF'] > configs['LMAX_PPXF']:
        message = "The given minimum wavelength LMIN_PPXF is larger than the maximum wavelength LMAX_PPXF. I will swap them! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        lmin = configs['LMAX_PPXF']
        lmax = configs['LMIN_PPXF']
        configs['LMIN_PPXF'] = lmin
        configs['LMAX_PPXF'] = lmax
    if configs['LMIN_GANDALF'] > configs['LMAX_GANDALF']:
        message = "The given minimum wavelength LMIN_GANDALF is larger than the maximum wavelength LMAX_GANDALF. I will swap them! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        lmin = configs['LMAX_GANDALF']
        lmax = configs['LMIN_GANDALF']
        configs['LMIN_GANDALF'] = lmin
        configs['LMAX_GANDALF'] = lmax
    if configs['LMIN_SFH'] > configs['LMAX_SFH']:
        message = "The given minimum wavelength LMIN_SFH is larger than the maximum wavelength LMAX_SFH. I will swap them! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        lmin = configs['LMAX_SFH']
        lmax = configs['LMIN_SFH']
        configs['LMIN_SFH'] = lmin
        configs['LMAX_SFH'] = lmax

    if configs['LMIN_GANDALF'] > configs['LMIN_SFH']  or  configs['LMAX_GANDALF'] < configs['LMAX_SFH']: 
        message = "The wavelength ranges of the SFH module cannot be larger than the GANDALF wavelength range."
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.error(message)
        SKIP_GALAXY = True

    if configs['ORIGIN'] == [0.,0.]: 
        message = "If ORIGIN is not set to the coordinates of the galaxy centre, the calculation of lambda_r will be wrong! Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)

    if os.path.isdir(configs['SSP_LIB']) == False:
        message = "No spectral template library found in "+configs['SSP_LIB']+". This is a fatal error! Galaxy will be skipped."
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.error(message)
        SKIP_GALAXY = True
    if configs['LINE_STRENGTH'] == 2  and  os.path.isfile( configs['SSP_LIB'].rstrip('/')+"_KB_LIS"+str(configs['CONV_COR'])+".fits" ) == False: 
        message = "No model file for line strength indices found in "+configs['SSP_LIB'].rstrip('/')+"_KB_LIS"+str(configs['CONV_COR'])+".fits"+". LINE_STRENGTH will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['LINE_STRENGTH'] = 0 
    if configs['NORM_TEMP'] not in ['MASS', 'LIGHT']:
        message = "NORM_TEMP has to be either be MASS or LIGHT. NORM_TEMP will be set to LIGHT. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['NORM_TEMP'] = 'LIGHT'

    if configs['FIXED'] not in [0,1]:
        message = "FIXED has to be either 0 (not fixed) or 1 (fixed). FIXED is set to fixed. Continue"
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['FIXED'] = 1
    
    if configs['LINE_STRENGTH'] in [2]  and  configs['MC_LS'] == 0: 
        message = "The conversion of LS indices to SSP's requires the calculation of errors on the line strength indices. LINE_STRENGTH will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['LINE_STRENGTH'] = 0
    if configs['LINE_STRENGTH'] in [1,2]  and  configs['GANDALF'] not in [1,3]  and  os.path.isfile(outdir+rootname+"_gandalf-cleaned_BIN.fits") == False: 
        message = "LINE_STRENGTH requires emission subtracted spectra. LINE_STRENGTH will be disabled. Continue."
        print(pipeline.prettyOutput_WarningPrefix()+message)
        logging.warning(message)
        configs['LINE_STRENGTH'] = 0

    return(SKIP_GALAXY, configs)


def determine_tasks(configs, dirPath, rootname, outdir):
    """
    Considering the parameters set in the configuration file and the output
    which is already available, it is decided which analysis modules will be
    executed. 
    """
    # USED_PARAMS.FITS
    if os.path.isfile(outdir+'USED_PARAMS.fits') == True:
        SKIP_GALAXY = checkConfig(configs, dirPath, outdir)
    else:
        saveConfig(configs, outdir, dirPath)
        SKIP_GALAXY = False

    # DEFINE VORONOI BINS
    if os.path.isfile(outdir+rootname+'_table.fits') == True:
        DEFINE_VORONOI_BINS = False
    else:
        DEFINE_VORONOI_BINS = True

    # APPLY VORONOI BINS 
    if os.path.isfile(outdir+rootname+'_VorSpectra.fits') == True:
        APPLY_VORONOI_BINS = False
    else: 
        APPLY_VORONOI_BINS = True

    # PPXF
    if os.path.isfile(outdir+rootname+'_ppxf.fits') == True:
        PPXF = False
    else:
        PPXF = True

    # GANDALF
    if   os.path.isfile(outdir+rootname+'_gandalf_SPAXEL.fits') == True  and os.path.isfile(outdir+rootname+'_gandalf_BIN.fits') == True:
        GANDALF = 0 # False
    elif os.path.isfile(outdir+rootname+'_gandalf_SPAXEL.fits') == False and configs['GANDALF'] == 2:
        GANDALF = 2 # SPAXEL level
    elif os.path.isfile(outdir+rootname+'_gandalf_BIN.fits')    == False and configs['GANDALF'] == 1:
        GANDALF = 1 # BIN level
    elif configs['GANDALF'] == 3: 
        GANDALF = 3
    else:
        GANDALF = 0 # False

    # SFH
    if os.path.isfile(outdir+rootname+'_sfh.fits') == False  and  configs['SFH'] == 1:
        SFH = True
    else:
        SFH = False

    # LINE_STRENGTH
    if configs['LINE_STRENGTH'] == 1  and  os.path.isfile(outdir+rootname+'_ls_OrigRes.fits') == True  and  os.path.isfile(outdir+rootname+'_ls_AdapRes.fits') == True: 
        LINE_STRENGTH = 0
    elif configs['LINE_STRENGTH'] == 2  and  os.path.isfile(outdir+rootname+'_ls_OrigRes.fits') == True  and  os.path.isfile(outdir+rootname+'_ls_AdapRes.fits') == True: 
        if len( fits.open(outdir+rootname+'_ls_AdapRes.fits') ) == 3:
            LINE_STRENGTH = 0 
        else: 
            LINE_STRENGTH = 2
    else: 
        LINE_STRENGTH = configs['LINE_STRENGTH']

    # IN CASE EVERYTHING IS ALREADY DONE
    if PPXF == False  and  GANDALF == 0  and  SFH == False  and  LINE_STRENGTH == 0:
        message = "There is nothing to do for "+str(configs['RUN_NAME'])+". Galaxy will be skipped!"
        print(pipeline.prettyOutput_WarningPrefix()+message)  
        logging.warning(message)
        SKIP_GALAXY = True

    return(SKIP_GALAXY, DEFINE_VORONOI_BINS, APPLY_VORONOI_BINS, PPXF, GANDALF, SFH, LINE_STRENGTH)


def printConfigs_Configs(configs):
    """ Prints overview of the chosen parameters to stdout and in the logfile. """
    os.system('clear')
    infoString = (
    "\n"
    "\033[0;37m"+"************************************************************"+"\033[0;39m\n"
    "\033[0;37m"+"*            T H E   G I S T   P I P E L I N E             *"+"\033[0;39m\n"
    "\033[0;37m"+"************************************************************"+"\033[0;39m\n"
    "\n"
    "\033[0;37m"+"Pipeline runs with the following settings:"+"\033[0;39m\n"
    "   * Run Name:           " + str(configs['RUN_NAME']) + "\n"
    "   * IFU:                " + configs['IFU'] + "\n"
    "   * PARALLEL:           " + str(bool(int(configs['PARALLEL']))) + "\n"
    "   * NCPU:               "+str(int(configs['NCPU'])) + "\n"
    "\n"
    "   * Lambda SNR:         " + str(configs['LMIN_SNR']) +     " - "+str(configs['LMAX_SNR']) +     " Angst." + "\n"
    "   * Lambda PPXF:        " + str(configs['LMIN_PPXF']) +    " - "+str(configs['LMAX_PPXF']) +    " Angst." + "\n"
    "   * Lambda GANDALF:     " + str(configs['LMIN_GANDALF']) + " - "+str(configs['LMAX_GANDALF']) + " Angst." + "\n"
    "   * Lambda SFH:         " + str(configs['LMIN_SFH']) +     " - "+str(configs['LMAX_SFH']) +     " Angst." + "\n"
    "   * Origin coord.:      " + str(configs['ORIGIN']) + "\n"
    "   * Redshift:           " + str(configs['REDSHIFT']) + "\n"
    "   * Init. sigma:        " + str(configs['SIGMA'])    + " km/s" + "\n"
    "   * Target SNR:         " + str(configs['TARGET_SNR']) + "\n"
    "   * Min. SNR:           " + str(configs['MIN_SNR']) + "\n"
    "   * Covariances factor: " + str(configs['COVAR_VOR']) + "\n"
    "\n"
    "   * SSP library:        " + str(configs['SSP_LIB']).split('/')[-2] + "\n"
    "   * Normalise temp.:    " + str(configs['NORM_TEMP']) + "\n"
    "\n"
    "\033[0;37m"+"PPXF settings:"+"\033[0;39m" + "\n"
    "   * Gauss-Hermite Mom.: " + str(configs['MOM']) + "\n"
    "   * Add. polyn. Degree: " + str(configs['ADEG']) + "\n"
    "   * Mul. polyn. Degree: " + str(configs['MDEG']) + "\n"
    "   * Num. of MC Sims.:   " + str(configs['MC_PPXF']) + "\n"
    "\n"
    "\033[0;37m"+"GANDALF settings:"+"\033[0;39m" + "\n"
    "   * Reddening:          " + str(configs['REDDENING']) + "\n"
    "   * E(B-V):             " + str(configs['EBmV']) + "\n"
    "   * Errors:             " + str(configs['FOR_ERRORS']) + "\n"
    "\n"
    "\033[0;37m"+"PPXF_SFH settings:"+"\033[0;39m" + "\n"
    "   * Regul. Error:       " + str(configs['REGUL_ERR']) + "\n"
    "   * Fixed kin. Param.:  " + str(configs['FIXED']) + "\n"
    "   * Noise parameter:    " + str(configs['NOISE']) + "\n"
    "\n"
    "\033[0;37m"+"LINE_STRENGTH settings:"+"\033[0;39m" + "\n"
    "   * Num. of MC Sims.:   " + str(configs['MC_LS']) + "\n"
    "   * Conv. Correction:   " + str(configs['CONV_COR']) + "\n"
    "   * Number of Walker:   " + str(configs['NWALKER']) + "\n"
    "   * Number of Chains:   " + str(configs['NCHAIN']) + "\n"
    "\n"
    "\033[0;37m"+"************************************************************"+"\033[0;39m"
    )
    print( infoString )
    logging.info( infoString )


def printConfigs_Tasks( DEBUG, PPXF, GANDALF, SFH, LINE_STRENGTH ):
    """ Prints which analysis modules will be executed to stdout and in the logfile. """

    if   GANDALF == 0: gandalfString = "   * GANDALF:            " + "False"
    elif GANDALF == 1: gandalfString = "   * GANDALF:            " + "BIN level"
    elif GANDALF == 2: gandalfString = "   * GANDALF:            " + "SPAXEL level"
    elif GANDALF == 3: gandalfString = "   * GANDALF:            " + "SPAXEL based on BIN level"

    if   LINE_STRENGTH == 0: lsString = "   * LINE_STRENGTH:      " + "False"
    elif LINE_STRENGTH == 1: lsString = "   * LINE_STRENGTH:      " + "LS"
    elif LINE_STRENGTH == 2: lsString = "   * LINE_STRENGTH:      " + "LS + SSP"

    if DEBUG == True: debugString = ( pipeline.prettyOutput_WarningPrefix()+"RUNNING IN DEBUG MODE!\n"
                                      "             Remember to clean output directory afterwards!\n\n" )
    else: debugString = ""

    infoString = (                  
    "\n"
    "\033[0;37m"+"Running:"+"\033[0;39m\n"
    "   * PPXF:               " + str(PPXF) + "\n" +
    gandalfString + "\n" 
    "   * SFH:                " + str(SFH) + "\n" + 
    lsString + "\n"
    "\n"
    "\033[0;37m"+"************************************************************"+"\033[0;39m\n"
    + debugString
    )
    print( infoString )
    logging.info( infoString )


def checkConfig(configs, dirPath, outdir):
    """
    Checks if the parameters set in the configuration file as well as the LSF
    are consistent with those saved in "USED_PARAMS.fits" in the output
    directory of the respective run. If the parameters are not consistent, a
    warning will be printed and the galaxy skipped. 
    """
    SKIP_GALAXY = False

    # 1: Check parameters in Config-file
    logfile          = fits.open(outdir+'USED_PARAMS.fits')[0].header
    ignoredSwitches  = ['DEBUG', 'GANDALF', 'SFH', 'LINE_STRENGTH', 'PARALLEL', 'NCPU', 'NOISE']
    detectedProblems = []

    # Check for problems
    for item in configs.keys():
        if item in ignoredSwitches:
            continue
        elif item == 'SSP_LIB':
            if str(configs[item]).split('/')[-2] != str(logfile[item]):
                detectedProblems.append( [item, configs[item].split('/')[-2], logfile[item]] )
        elif str(configs[item]) != str(logfile[item]):
            detectedProblems.append( [item, configs[item], logfile[item]] )

    # Print message
    if len(detectedProblems) != 0:
        print(pipeline.prettyOutput_FailedPrefix()+"The parameters given in Config and USED_PARAMS.fits are not identical. The following conflicts were detected:")
        print("             "+"Item         Config       USED_PARAMS.fits")
        for i in range( len(detectedProblems) ):
            print("             "+"{:13}{:13}{:13}".format(str(detectedProblems[i][0]), str(detectedProblems[i][1]), str(detectedProblems[i][2])) )
        SKIP_GALAXY = True

    # 2: Check LSF
    lengthLSF     = fits.open(outdir+"USED_PARAMS.fits")[1].data.LAMBDA.shape[0]
    savedLSF      = np.zeros((lengthLSF,2))
    savedLSF[:,0] = np.array( fits.open(outdir+"USED_PARAMS.fits")[1].data.LAMBDA )
    savedLSF[:,1] = np.array( fits.open(outdir+"USED_PARAMS.fits")[1].data.FWHM   )
    configLSF     = np.genfromtxt(dirPath.configDir+'LSF-Config_'+configs['IFU'], comments='#')

    if savedLSF.shape != configLSF.shape  or  np.allclose(savedLSF, configLSF) == False:
        print(pipeline.prettyOutput_FailedPrefix()+"The given LSF in "+outdir+"USED_PARAMS.fits and "+dirPath.configDir+"LSF-Config_"+configs['IFU']+" are not identical!")
        SKIP_GALAXY = True

    return(SKIP_GALAXY)


def saveConfig(configs, outdir, dirPath):
    """
    Saves the used configuration parameters and LSF to "USED_PARAMS.fits" in the
    output directory of the run. 
    """
    # Save the used configurations to USED_PARAMS.fits
    outfits  = outdir+'USED_PARAMS.fits'

    # Primary Header: Info from Config-file
    priHeader = fits.Header()
    for item in configs.keys():
        if item == 'SSP_LIB':
            priHeader['hierarch '+item] = str(configs[item].split('/')[-2])
        else:
            priHeader['hierarch '+item] = str(configs[item])
    priHDU = fits.PrimaryHDU(header=priHeader)

    # Extension 1: Table HDU with LSF information
    LSF_Config = np.genfromtxt(dirPath.configDir+'LSF-Config_'+configs['IFU'], comments='#')
    
    cols = []
    cols.append( fits.Column(name='LAMBDA', format='D', array=LSF_Config[:,0]) )
    cols.append( fits.Column(name='FWHM',   format='D', array=LSF_Config[:,1]) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'LSF'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)


def readConfig(name, galnumber):
    """ Read the configuration file and pass the info to setup_configs. """
    # TODO: This might be easier done with astropy.io.ascii ??!!
    clines = []
    for line in open(name, 'r'):
        li=line.strip()
        if not li.startswith("#"):
            clines.append( line.rstrip() )
    
    flag   = clines[0].split()
    option = clines[galnumber].split()

    # Check if flag actually contains the flags
    if flag[0] != "RUN_NAME": 
        message = "The Configuration file does not seem to contain a header line ('RUN_NAME  DEBUG  VORONOI ...'). This is a fatal error. Exit!"
        pipeline.prettyOutput_Failed( message )
        exit(1)

    configs = { }
    for i in range( len(flag) ):
        configs[flag[i]] = option[i]
    return(configs)


