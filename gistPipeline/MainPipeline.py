#!/usr/bin/env python

# ======================================================================================================================
#
#                                          T H E   G I S T   P I P E L I N E
#
# ======================================================================================================================



import os
os.environ["MKL_NUM_THREADS"]     = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"]     = "1" 

import numpy             as np
from   astropy.io        import fits, ascii
from   scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")
import logging
import time
import sys
import optparse
import importlib.util

import matplotlib
matplotlib.use('pdf')

from gistPipeline.gistModules import util           as pipeline
from gistPipeline.gistModules import gistInitialize as util_init
from gistPipeline.gistModules import gistPrepare    as util_prepare
from gistPipeline.gistModules import gistVoronoi    as util_voronoi
from gistPipeline.gistModules import gistPPXF       as util_ppxf
from gistPipeline.gistModules import gistGANDALF    as util_gandalf
from gistPipeline.gistModules import gistSFH        as util_sfh
from gistPipeline.gistModules import gistLS         as util_ls



def handleUncaughtException(exceptionType, exceptionValue, exceptionTraceback):
    logging.error("Uncaught Exception", exc_info=(exceptionType, exceptionValue, exceptionTraceback))
    sys.__excepthook__(exceptionType, exceptionValue, exceptionTraceback)
    print("\n")
    pipeline.prettyOutput_Failed("FATAL ERROR! The execution of the pipeline is terminated.")
    print("")



def runPipeline(galnumber, dirPath):
# ==============================================================================
#                                 S T A R T U P 
# ==============================================================================
    # Read Config-file and setup the configs structure
    PARALLEL, DEBUG, configs, datafile, rootname, outdir = util_init.setup_configs(galnumber, dirPath) 
    
    # Setup logfile
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(filename= outdir+'LOGFILE',
                        level   = logging.INFO,
                        format  = '%(asctime)s - %(levelname)-8s - %(module)s: %(message)s',
                        datefmt = '%m/%d/%y %H:%M:%S' )
    logging.Formatter.converter = time.gmtime    
    logging.info("\n\n# ================================================\n"\
            +"#   THE GIST PIPELINE, V2.1  "\
            +"#   STARTUP OF PIPELINE\n# "+time.strftime("  %d-%b-%Y %H:%M:%S %Z", time.gmtime())+\
            "\n# ================================================")

    # Redirect uncaught exceptions to logfile
    sys.excepthook = handleUncaughtException

    # Print configurations on screen
    util_init.printConfigs_Configs(configs)

    # Basic checks on input values
    SKIP_GALAXY, configs = util_init.parameter_checks(datafile, configs, dirPath, rootname, outdir)
    if SKIP_GALAXY == True: 
        message = " - - - GALAXY IS SKIPPED! - - - "
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.critical(message)
        return(None)

    # Determine pipeline tasks
    SKIP_GALAXY, DEFINE_VORONOI_BINS, APPLY_VORONOI_BINS, PPXF, GANDALF, SFH, LINE_STRENGTH, \
            = util_init.determine_tasks(configs, dirPath, rootname, outdir)
    if SKIP_GALAXY == True: 
        message = " - - - GALAXY IS SKIPPED! - - - "
        print(pipeline.prettyOutput_FailedPrefix()+message)
        logging.critical(message)
        return(None)

    # Print configurations on screen
    util_init.printConfigs_Tasks(DEBUG, PPXF, GANDALF, SFH, LINE_STRENGTH)

    # Import the readData routine, depending on the chosen instrument
    if os.path.isfile( dirPath.configDir+configs['IFU']+'.py' ) == True: 
        spec = importlib.util.spec_from_file_location("", dirPath.configDir+configs['IFU']+'.py')
        logging.info("Using the user-defined read-in routine in "+dirPath.configDir+configs['IFU']+".py")
    elif configs['IFU']+".py" in os.listdir( os.path.dirname(os.path.realpath(__file__))+"/readData/" ): 
        spec = importlib.util.spec_from_file_location("", os.path.dirname(os.path.realpath(__file__))+"/readData/"+configs['IFU']+'.py')
        logging.info("Using the default read-in routine for "+configs['IFU'])
    else:
        logging.info("No read-in routine found for "+configs['IFU']+" found. Galaxy is skipped!")
        pipeline.prettyOutput_Failed("No read-in routine for "+configs['IFU']+" found. Galaxy is skipped!")
        return(None)
    readCUBE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readCUBE)
        


# ==============================================================================
#                            P R E P A R E   D A T A 
# ==============================================================================
    print("\033[0;37m"+" - - - - - Running Preparation! - - - - - "+"\033[0;39m")
    logging.info(" - - - Running Preparation - - - ")

    # Read IFU cube
    cube = readCUBE.read_cube(DEBUG, datafile, configs)

    # Reject defunct spaxels and apply SNR threshold
    idx_inside, idx_outside = util_prepare.rejectDefunctSpaxels_applySNRThreshold(cube, configs)

    if DEFINE_VORONOI_BINS == True:
        # Define Voronoi bins and save table
        binNum = util_voronoi.define_voronoi_bins\
                 (configs['VORONOI'], cube['x'], cube['y'], cube['signal'], cube['noise'], cube['pixelsize'], \
                  cube['snr'], configs['TARGET_SNR'], configs['COVAR_VOR'], idx_inside, idx_outside, rootname, outdir)
        if type(binNum) == bool: 
            message = " - - - GALAXY IS SKIPPED! - - - "
            pipeline.prettyOutput_Failed(message)
            logging.critical(message)
            return(None)
    else:
        # Load Voronoi table
        binNum = np.array( fits.open(outdir+rootname+'_table.fits')[1].data.BIN_ID )
        binNum = binNum[np.where(binNum >= 0)[0]]

    if APPLY_VORONOI_BINS == True:
        # Apply bins to linear spectra
        util_voronoi.apply_voronoi_bins(binNum, cube['spec'][:,idx_inside], \
                cube['error'][:,idx_inside], rootname, outdir, cube['velscale'], cube['wave'], 'lin')
    
        # Log-rebin spectra and apply bins, or read them from file
        log_spec, log_error, logLam = util_prepare.log_rebinning(cube, configs, rootname, outdir)
        util_voronoi.apply_voronoi_bins(binNum, log_spec[:,idx_inside], log_error[:,idx_inside], rootname,\
                outdir, cube['velscale'], logLam, 'log')

    # Read LSF of observation and templates and construct an interpolation function
    LSF           = np.genfromtxt(dirPath.configDir+'LSF-Config_'+configs['IFU'], comments='#')
    LSF[:,0]      = LSF[:,0] / (1 + configs['REDSHIFT'])
    LSF[:,1]      = LSF[:,1] / (1 + configs['REDSHIFT'])
    LSF_Data      = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')
    LSF           = np.genfromtxt(dirPath.configDir+'LSF-Config_'+configs['SSP_LIB'].split('/')[-2], comments='#')
    LSF_Templates = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')

    print("\033[0;37m"+" - - - - - Preparation done! - - - - -"+"\033[0;39m")
    logging.info(" - - - Preparation Done - - - \n")



# ==============================================================================
#                             D O   A N A L Y S I S 
# ==============================================================================

    # RUN PPXF
    util_ppxf.runModule_PPXF(PPXF, PARALLEL, configs, dirPath, cube['velscale'], LSF_Data, LSF_Templates, outdir, rootname)

    # RUN GANDALF
    util_gandalf.runModule_GANDALF(GANDALF, PARALLEL, configs, cube['velscale'], LSF_Data, LSF_Templates, outdir, rootname)

    # RUN SFH
    util_sfh.runModule_SFH(SFH, PARALLEL, configs, dirPath, cube['velscale'], LSF_Data, LSF_Templates, outdir, rootname)

    # RUN LINE_STRENGTH
    util_ls.runModule_LINESTRENGTH(LINE_STRENGTH, "ORIGINAL", PARALLEL, configs, cube['velscale'], LSF_Data, outdir, rootname)
    util_ls.runModule_LINESTRENGTH(LINE_STRENGTH, "ADAPTED",  PARALLEL, configs, cube['velscale'], LSF_Data, outdir, rootname)

    # Add other modules here:
    #   ...

    print(" *************** ")
    print(" **   DONE!   ** ")
    print(" *************** ")
    logging.info(" - DONE! :)")



# ==============================================================================
#                           M A I N   F U N C T I O N 
# ==============================================================================
def main(args=None):

    # Capture command-line arguments
    parser = optparse.OptionParser(usage="%gistPipeline [options] arg")
    parser.add_option("--working-dir", dest="workingDirectory", type="string", help="State the absolute path of your working directory.")
    parser.add_option("--config-file", dest="configFile",       type="string", help="State the absolute path of the Config-file. If not defined, MasterConfig is used instead.")
    parser.add_option("--spec-temp",   dest="spTempDir",        type="string", help="State the absolute path of the directory containing the spectral template library. If not defined, workingDir/spectralTemplates is used instead.")
    (dirPath, args) = parser.parse_args()

    # Set defaults, if necessary
    if dirPath.workingDirectory == None: 
        pipeline.prettyOutput_Failed("Required argument '--working-dir' not supplied. Exit!")
        exit(1)
    elif dirPath.workingDirectory[-1] != '/':
        dirPath.workingDirectory = dirPath.workingDirectory + '/'
    if dirPath.configFile == None: 
        dirPath.configFile = dirPath.workingDirectory + "configFiles/MasterConfig"
    if dirPath.spTempDir == None: 
        dirPath.spTempDir = dirPath.workingDirectory + "spectralTemplates/"
    if dirPath.spTempDir[-1] != '/': 
        dirPath.spTempDir = dirPath.spTempDir + '/'

    dirPath.inputDir  = dirPath.workingDirectory + "inputData/"
    dirPath.outputDir = dirPath.workingDirectory + "results/"
    dirPath.configDir = dirPath.workingDirectory + "configFiles/"

    # Check if Config-file exists
    if os.path.isfile(dirPath.configFile) == False:
        pipeline.prettyOutput_Failed("Config-file at "+dirPath.configFile+" not found. Exit!")
        exit(1)

    # Iterate over galaxies in Config-file
    ngalaxies = len( ascii.read(dirPath.configFile) )
    if ngalaxies == 0: 
        message = "The number of runs defined in the Config-file seems to be 0. Maybe the header is commented out? Exit."
        pipeline.prettyOutput_Failed(message)
        exit(1)
    for galnumber in range(ngalaxies):
        runPipeline(galnumber+1, dirPath)
        print("\n")


if __name__ == '__main__':
    # Call the main function
    main()
