import numpy    as np
from astropy.io import fits
from astropy import table

from multiprocess import Queue, Process

import time
import os
import glob
import logging

from printStatus import printStatus

from gistPipeline.prepareTemplates import _prepareTemplates, prepare_gas_templates
from gistPipeline.auxiliary import _auxiliary

# Then use system installed version instead
from ppxf.ppxf      import ppxf

# Physical constants
C = 299792.458 # speed of light in km/s


"""
PURPOSE:
  This module executes the emission-line analysis of the pipeline.
  Uses the pPXF implementation, replacing Gandalf.
  Module written for gist-geckos based on the gist SFH module
  combined with the PHANGS DAP emission line module.
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for i, templates, galaxy_i, noise_i, velscale, start, goodPixels, tpl_comp, moments, offset,\
        mdeg, fixed, velscale_ratio, tied, gas_comp, gas_names, nbins\
        in iter(inQueue.get, 'STOP'):
        gas_sol, gas_error, chi2, gas_flux, gas_flux_error, gas_names, bestfit, gas_bestfit, star_sol, star_err = \
        run_ppxf(templates, galaxy_i, noise_i, velscale, start, goodPixels, tpl_comp, moments, offset, mdeg, fixed, velscale_ratio, tied, gas_comp, gas_names, i, nbins)

        outQueue.put((i, gas_sol, gas_error, chi2, gas_flux, gas_flux_error, gas_names, bestfit, gas_bestfit, star_sol, star_err ))


def run_ppxf(templates, galaxy_i, noise_i, velscale, start, goodPixels, tpl_comp, moments, offset, mdeg,\
             fixed, velscale_ratio, tied, gas_comp, gas_names, i, nbins):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.
    """
    printStatus.progressBar(i, nbins, barLength = 50)

    try:


        pp = ppxf(templates, galaxy_i, noise_i, velscale, start, goodpixels=goodPixels, plot=False, quiet=True,\
              component = tpl_comp, moments=moments, degree=-1, vsyst=offset, mdegree=mdeg, fixed=fixed, velscale_ratio=velscale_ratio,\
              tied=tied, gas_component=gas_comp, gas_names=gas_names)


        # note here I am only passing out of the function the gas kinematics (pp.sol[1:])
        # to also save the stellar kinematics one would need to add pp.sol[0] and same for pp.error (which is appended at the end now for fun)

        return(pp.sol[1:], pp.error[1:], pp.chi2, pp.gas_flux, pp.gas_flux_error, pp.gas_names, pp.bestfit, pp.gas_bestfit, pp.sol[0], pp.error[0])

    except:
        # raise exception
        print('bad gas')
        return(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

def tidy_up_fluxes_and_kinematics(gas_kinematics, kinematics_all_err,gas_flux,\
                                  gas_flux_error, emldb, eml_tying, gas_moments):
    """
    Function taken from util_templates.py of PHANGS DAP. Tidies up some stuff.
    """
    w = (emldb['action']=='f')
    linesfitted = emldb[w]
    tpli_fitted = eml_tying['tpli'][w]
    nbins =gas_kinematics.shape[0]
    ngascomp =  len(np.unique(eml_tying['comp']))
    ngastpl = len(np.unique(tpli_fitted))

    fluxes_final = np.zeros( (nbins, len(tpli_fitted)) )
    fluxes_err_final = np.zeros((nbins, len(tpli_fitted)))
    vel_final = np.zeros((nbins, len(tpli_fitted)))
    vel_err_final = np.zeros((nbins, len(tpli_fitted)))
    sigma_final = np.zeros((nbins, len(tpli_fitted)))
    sigma_err_final = np.zeros((nbins, len(tpli_fitted)))
    if gas_moments >=3:
        h3_final = np.zeros((nbins, len(tpli_fitted)))
    if gas_moments ==4:
        h4_final = np.zeros((nbins, len(tpli_fitted)))

    for j in range(ngastpl):
        #which line(s) correspond to this template
        templates_x= np.arange(len(tpli_fitted))[tpli_fitted==j]
        #component number of this template
        component_x = eml_tying['comp'][j]
        # if more than one line in this template...
        if len(templates_x)>1:
            total_flux_template = np.sum(linesfitted['A_i'][templates_x])
            stronger_line = templates_x[linesfitted['A_i'][templates_x] ==1.][0]
            weaker_lines = templates_x[linesfitted['A_i'][templates_x] !=1.]
            fluxes_final[:,stronger_line] = gas_flux[:, j]/total_flux_template
            fluxes_err_final[:,stronger_line] = gas_flux_error[:, j]/total_flux_template

            for kk in range(len(weaker_lines)):
                fluxes_final[:, weaker_lines[kk]] = \
                    gas_flux[:, j]/total_flux_template *\
                    linesfitted['A_i'][weaker_lines[kk]]
                fluxes_err_final[:, weaker_lines[kk]] = \
                    gas_flux_error[:, j]/total_flux_template *\
                    linesfitted['A_i'][weaker_lines[kk]]

        else:
            fluxes_final[:, templates_x[0]] = gas_flux[:, j]
            fluxes_err_final[:, templates_x[0]] = gas_flux_error[:, j]


    for j in range(ngastpl):
        #which line(s) correspond to this template
        # linesfitted[templates_x] will give you the info on this line(s)
        templates_x= np.arange(len(tpli_fitted))[tpli_fitted==j]
        #component number of this template
        component_x = eml_tying['comp'][j]
        # now list all the components that can have
        # sigma and velocity group number of this template
        sigma_x = eml_tying['comp'] [ eml_tying['sgrp']== eml_tying['sgrp'][j]]
        v_x = eml_tying['comp'][eml_tying['vgrp'] == eml_tying['vgrp'][j]]

        # in case there is a component with a lower number than component_x
        # in this velocity/sigma group it means the  kinematics of that line
        # is tied to the one of that component, so replace  component_x with component_vx
        # in order to get the ERRORS of the primary (aka free) component
        # otherwise you would get zerro errors because line is tied
        component_vx = np.min(v_x)
        component_sx = np.min(sigma_x)

        #components tied which have therefore zero errors
        # THESE LINES causes ENDLESS problems because of the ==0 conditions
        # tied_vel  = np.arange(ngascomp)[kinematics_all_err[0, :, 0]==0]
        # tied_sigma  = np.arange(ngascomp)[kinematics_all_err[0, :, 1]==0]

        #
        # if component_x in tied_vel:
        #     component_vx = v_x[np.array([i not in tied_vel for i in v_x])][0]
        # if component_x in tied_sigma:
        #     component_sx = sigma_x[np.array([i not in tied_sigma for i in sigma_x])][0]

        for kk in range(len(templates_x)):
            vel_final[:, templates_x[kk]] = gas_kinematics[:,component_x, 0 ]
            sigma_final[:, templates_x[kk]] = gas_kinematics[:,component_x, 1 ]

            vel_err_final[:, templates_x[kk]] = kinematics_all_err[:,component_vx, 0 ]
            sigma_err_final[:, templates_x[kk]] = kinematics_all_err[:,component_sx, 1 ]

            extra = None
            if gas_moments >=3:
                h3_final[:, templates_x[kk]] = gas_kinematics[:,component_x, 2 ]
                extra = {}
                extra['h3']=h3_final
            if gas_moments ==4:
                h4_final[:, templates_x[kk]] = gas_kinematics[:,component_x, 3 ]
                extra['h4']=h4_final



    return(linesfitted, fluxes_final, fluxes_err_final, vel_final,vel_err_final, \
           sigma_final,sigma_err_final, extra)


### THIS WILL NEED TO BE FIXED. IT"S CURRENTLY COPIED DIRECTLY FROM THE PHANGS DAP
def save_ppxf_emlines(config, rootname, outdir, level, linesfitted,
        gas_flux_in_units, gas_err_flux_in_units,vel_final, vel_err_final,
        sigma_final_measured, sigma_err_final, chi2, templates_sigma, bestfit, gas_bestfit, stkin, spectra, error, logLam_galaxy, ubins, npix, extra):

        # ========================
        # SAVE RESULTS
        outfits_ppxf = rootname+'/'+outdir+'_emlines_'+level+'.fits'
        printStatus.running("Writing: "+outfits_ppxf.split('/')[-1])
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_emlines_"+level+".fits")

        # Primary HDU
        priHDU = fits.PrimaryHDU()

        # Table HDU with PPXF output data
        cols = []
        cols.append( fits.Column(name='BIN_ID',         format='J', array=ubins             ))
        cols.append( fits.Column(name='V_STARS2',         format='D', array=stkin[:,0]             ))
        cols.append( fits.Column(name='SIGMA_STARS2',         format='D', array=stkin[:,1]             ))
        cols.append( fits.Column(name='CHI2_TOT',         format='D', array=chi2             ))
        #emission lines names
        names = np.char.array( linesfitted['name'] )+np.char.array(['{:d}'.format(int(j)) for j in linesfitted['lambda']])


        for i in range(len(names)):
            cols.append( fits.Column(name=names[i]+'_FLUX' , format='D', array=gas_flux_in_units[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_FLUX_ERR' , format='D', array=gas_err_flux_in_units[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_VEL' , format='D', array=vel_final[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_VEL_ERR' , format='D', array=vel_err_final[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_SIGMA' , format='D', array=sigma_final_measured[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_SIGMA_ERR' , format='D', array=sigma_err_final[:,i]  ))
            cols.append( fits.Column(name=names[i]+'_SIGMA_CORR' , format='D', array=np.zeros( sigma_final_measured.shape[0] ) + templates_sigma[i]  ))

            if (extra is not None) and ('h3' in extra.keys()):
                cols.append( fits.Column(name=names[i]+'_H3' , format='D', array=extra['h3'][:,i]  ))
            if (extra is not None) and ('h4' in extra.keys()):
                cols.append( fits.Column(name=names[i]+'_H4' , format='D', array=extra['h4'][:,i]  ))

        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = 'EMISSION'

        # Create HDU list and write to file
        HDUList = fits.HDUList([priHDU, dataHDU])
        HDUList.writeto(outfits_ppxf, overwrite=True)

        printStatus.running("Writing: "+outfits_ppxf.split('/')[-1])
        logging.info("Wrote: "+outfits_ppxf)


        # ========================
        # SAVE BESTFIT
        outfits_ppxf = rootname+'/'+outdir+'_ppxf-bestfit-emlines_'+level+'.fits'
        printStatus.running("Writing: "+outfits_ppxf.split('/')[-1])
        cleaned = spectra.T - gas_bestfit
        spec = spectra.T
        err = error.T
        # Primary HDU
        # priHDU = fits.PrimaryHDU()
        #
        # # Table HDU with PPXF bestfit
        # cols = []
        # ##cols.append( fits.Column(name='BIN_ID',  format='J',    array=ubins                    ))
        # ##cols.append( fits.Column(name='LOGLAM', format=str(logLam_galaxy.shape[0])+'D', array=logLam_galaxy      ))
        # cols.append( fits.Column(name='LOGLAM', format='D', array=logLam_galaxy ))
        # cols.append( fits.Column(name='SPEC', format=str(spec.shape[1])+'D', array=spec      ))
        # cols.append( fits.Column(name='ERR', format=str(err.shape[1])+'D', array=err      ))
        # ##cols.append( fits.Column(name='BESTFIT', format=str(bestfit.shape[1])+'D', array=bestfit      ))
        # ##cols.append( fits.Column(name='GAS_BESTFIT', format=str(gas_bestfit.shape[1])+'D', array=gas_bestfit      ))
        # ##cols.append( fits.Column(name='GAS_CLEANED', format=str(cleaned.shape[1])+'D', array=cleaned      ))
        # # cols.append( fits.Column(name='MASK', format='D', array=mask_for_original_spectra      ))
        # dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        # dataHDU.name = 'FIT'

        # Primary HDU
        priHDU = fits.PrimaryHDU()

        # Extension 1: Table HDU with cleaned spectrum
        cols = []
        cols.append(
            fits.Column(name="SPEC", format=str(npix) + "D", array=cleaned)
        )
        cols.append(
            fits.Column(name="ESPEC", format=str(npix) + "D", array=err)
        )
        dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        dataHDU.name = "CLEANED_SPECTRA"

        # Extension 2: Table HDU with logLam
        cols = []
        cols.append(fits.Column(name="LOGLAM", format="D", array=logLam_galaxy))
        logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        logLamHDU.name = "LOGLAM"

        # Extension 3: Table HDU with bestfit
        cols = []
        cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=bestfit))
        bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        bestfitHDU.name = "BESTFIT"

        # Extension 3: Table HDU with gas bestfit
        cols = []
        cols.append(fits.Column(name="GAS_BESTFIT", format=str(npix) + "D", array=gas_bestfit))
        gas_bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        gas_bestfitHDU.name = "GAS_BESTFIT"

        # Create HDU list and write to file
        priHDU = _auxiliary.saveConfigToHeader(priHDU, config["GAS"])
        dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["GAS"])
        logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["GAS"])
        bestfitHDU = _auxiliary.saveConfigToHeader(bestfitHDU, config["GAS"])
        gas_bestfitHDU = _auxiliary.saveConfigToHeader(gas_bestfitHDU, config["GAS"])


        HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, bestfitHDU, gas_bestfitHDU])
        HDUList.writeto(outfits_ppxf, overwrite=True)

        printStatus.running("Writing: "+outfits_ppxf.split('/')[-1])
        logging.info("Wrote: "+outfits_ppxf)


# def save_sfh(mean_result, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, logLam_galaxy, goodPixels,\
#              velscale, logLam1, ncomb, nAges, nMetal, nAlpha, npix, config):
#     """ Save all results to disk. """
#     # ========================
#     # SAVE KINEMATICS
#     outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh.fits'
#     printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')
#
#     # Primary HDU
#     priHDU = fits.PrimaryHDU()
#
#     # Table HDU with stellar kinematics
#     cols = []
#     cols.append(fits.Column(name='AGE',   format='D', array=mean_result[:,0]))
#     cols.append(fits.Column(name='METAL', format='D', array=mean_result[:,1]))
#     cols.append(fits.Column(name='ALPHA', format='D', array=mean_result[:,2]))
#
#     if config['SFH']['FIXED'] == False:
#         cols.append(fits.Column(name='V',     format='D', array=kin[:,0]))
#         cols.append(fits.Column(name='SIGMA', format='D', array=kin[:,1]))
#         if np.any(kin[:,2]) != 0: cols.append(fits.Column(name='H3', format='D', array=kin[:,2]))
#         if np.any(kin[:,3]) != 0: cols.append(fits.Column(name='H4', format='D', array=kin[:,3]))
#         if np.any(kin[:,4]) != 0: cols.append(fits.Column(name='H5', format='D', array=kin[:,4]))
#         if np.any(kin[:,5]) != 0: cols.append(fits.Column(name='H6', format='D', array=kin[:,5]))
#
#         cols.append(fits.Column(name='FORM_ERR_V',     format='D', array=formal_error[:,0]))
#         cols.append(fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1]))
#         if np.any(formal_error[:,2]) != 0: cols.append(fits.Column(name='FORM_ERR_H3', format='D', array=formal_error[:,2]))
#         if np.any(formal_error[:,3]) != 0: cols.append(fits.Column(name='FORM_ERR_H4', format='D', array=formal_error[:,3]))
#         if np.any(formal_error[:,4]) != 0: cols.append(fits.Column(name='FORM_ERR_H5', format='D', array=formal_error[:,4]))
#         if np.any(formal_error[:,5]) != 0: cols.append(fits.Column(name='FORM_ERR_H6', format='D', array=formal_error[:,5]))
#
#     dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     dataHDU.name = 'SFH'
#
#     # Create HDU list and write to file
#     priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
#     dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
#     HDUList = fits.HDUList([priHDU, dataHDU])
#     HDUList.writeto(outfits_sfh, overwrite=True)
#
#     printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')
#     logging.info("Wrote: "+outfits_sfh)
#
#
#     # ========================
#     # SAVE WEIGHTS AND GRID
#     outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-weights.fits'
#     printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')
#
#     # Primary HDU
#     priHDU = fits.PrimaryHDU()
#
#     # Table HDU with weights
#     cols = []
#     cols.append( fits.Column(name='WEIGHTS', format=str(w_row.shape[1])+'D', array=w_row ))
#     dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     dataHDU.name = 'WEIGHTS'
#
#     logAge_row = np.reshape(logAge_grid, ncomb)
#     metal_row  = np.reshape(metal_grid,  ncomb)
#     alpha_row  = np.reshape(alpha_grid,  ncomb)
#
#     # Table HDU with grids
#     cols = []
#     cols.append( fits.Column(name='LOGAGE',  format='D',           array=logAge_row  ))
#     cols.append( fits.Column(name='METAL',   format='D',           array=metal_row   ))
#     cols.append( fits.Column(name='ALPHA',   format='D',           array=alpha_row   ))
#     gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     gridHDU.name = 'GRID'
#
#     # Create HDU list and write to file
#     priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
#     dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
#     gridHDU = _auxiliary.saveConfigToHeader(gridHDU, config['SFH'])
#     HDUList = fits.HDUList([priHDU, dataHDU, gridHDU])
#     HDUList.writeto(outfits_sfh, overwrite=True)
#
#     fits.setval(outfits_sfh,'NAGES',  value=nAges)
#     fits.setval(outfits_sfh,'NMETAL', value=nMetal)
#     fits.setval(outfits_sfh,'NALPHA', value=nAlpha)
#
#     printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')
#     logging.info("Wrote: "+outfits_sfh)
#
#
#     # ========================
#     # SAVE BESTFIT
#     outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-bestfit.fits'
#     printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')
#
#     # Primary HDU
#     priHDU = fits.PrimaryHDU()
#
#     # Table HDU with SFH bestfit
#     cols = []
#     cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=bestfit ))
#     dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     dataHDU.name = 'BESTFIT'
#
#     # Table HDU with SFH logLam
#     cols = []
#     cols.append( fits.Column(name='LOGLAM', format='D', array=logLam_galaxy ))
#     logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     logLamHDU.name = 'LOGLAM'
#
#     # Table HDU with SFH goodpixels
#     cols = []
#     cols.append( fits.Column(name='GOODPIX', format='J', array=goodPixels ))
#     goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     goodpixHDU.name = 'GOODPIX'
#
#     # Create HDU list and write to file
#     priHDU     = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
#     dataHDU    = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
#     logLamHDU  = _auxiliary.saveConfigToHeader(logLamHDU, config['SFH'])
#     goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config['SFH'])
#     HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
#     HDUList.writeto(outfits_sfh, overwrite=True)
#
#     fits.setval(outfits_sfh,'VELSCALE',value=velscale)
#     fits.setval(outfits_sfh,'CRPIX1',  value=1.0)
#     fits.setval(outfits_sfh,'CRVAL1',  value=logLam1[0])
#     fits.setval(outfits_sfh,'CDELT1',  value=logLam1[1]-logLam1[0])
#
#     printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')
#     logging.info("Wrote: "+outfits_sfh)

def performEmissionLineAnalysis(config): #This is your main emission line fitting loop
    #print("")
    print("\033[0;37m"+" - - - - - Running Emission Lines Fitting - - - - - "+"\033[0;39m")
    logging.info(" - - - Running Emission Lines Fitting - - - ")

    # #--> some bookkeeping - all commented out for now Amelia
    # # if there is only one spectrum to fit, make sure to reformat it
    # if log_spec.ndim==1:
    #     log_error= np.expand_dims(log_error, axis=1)
    #     log_spec= np.expand_dims(log_spec, axis=1)
    # nbins    = log_spec.shape[1]
    # ubins    = np.arange(0, nbins)
    # npix_in     = log_spec.shape[0]
    # n_spaxels_per_bin = np.zeros(nbins)
    # if bin_id is None:
    #     bin_id = ubins
    # # number of spaxels per bin
    # for i in range(nbins):
    #     windx = (bin_id ==i)
    #     n_spaxels_per_bin[i]=np.sum(windx)
    # velscale_ratio = 1
    # # check if wavelength is in vacuum
    # if 'WAV_VACUUM' in configs:
    #     wav_in_vacuum = configs['WAV_VACUUM']
    # else:
    #     wav_in_vacuum = False

    # for now the number of gas moments is fixed to 2 (i.e. v and sigma, no h3 and h4 etc for gas)


    # # regenerate the LSF interpolation function for simplicity, NOTE that I am using rest-frame wavelength
    # LSF_InterpolationFunction  = interp1d(np.exp(logLam)/(1+configs['REDSHIFT']/cvel), LSF/(1+configs['REDSHIFT']/cvel),
    #      'linear', fill_value = 'extrapolate')

    ## --------------------- ##
    # For output filenames
    # BUT - currently only testing on BIN
    if config["GAS"]["LEVEL"] in ["BIN", "SPAXEL"]:
        currentLevel = config["GAS"]["LEVEL"]
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"
        )
        == False
    ):
        currentLevel = "BIN"
    elif (
        config["GAS"]["LEVEL"] == "BOTH"
        and os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_gas_BIN.fits"
        )
        == True
    ):
        currentLevel = "SPAXEL"

    # Oversample the templates by a factor of two
    velscale_ratio = 1 # 2 ####### ??????? I don't know if this is the case now

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "GAS")
    ## --------------------- ##

    # if 'MDEG_EMS' in configs:
    #     emi_mpol_deg = configs['MDEG_EMS']
    # else:
    #     emi_mpol_deg = 8
    emi_mpol_deg = config['GAS']['MDEG'] # Should be ~8
    ## --------------------- ##
    # Read data if we run on BIN level
    if currentLevel == "BIN":
        # Read spectra from file
        hdu = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_BinSpectra.fits"
        )
        spectra = np.array(hdu[1].data.SPEC.T)
        error = np.array(hdu[1].data.ESPEC.T)
        logLam_galaxy = np.array(hdu[2].data.LOGLAM)
        idx_lam = np.where(
            np.logical_and(
                np.exp(logLam_galaxy) > config["GAS"]["LMIN"],
                np.exp(logLam_galaxy) < config["GAS"]["LMAX"],
            )
        )[0]
        spectra = spectra[idx_lam, :]
        error = error[idx_lam, :]  # AJB added
        logLam_galaxy = logLam_galaxy[idx_lam]
        npix = spectra.shape[0]
        nbins = spectra.shape[1]
        ubins = np.arange(0, nbins)
        nstmom = config['KIN']['MOM'] # Usually = 4
        velscale = hdu[0].header["VELSCALE"]
        # # the wav range of the data (observed)
        LamRange = (np.exp(logLam_galaxy[0]), np.exp(logLam_galaxy[-1]))

        #Determining the number of spaxels per bin
        hdu2 = fits.open(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])+ "_table.fits")
        bin_id = hdu2['TABLE'].data['BIN_ID']
        n_spaxels_per_bin = np.zeros(nbins)
        if bin_id is None:
            bin_id = ubins
        # number of spaxels per bin
        for i in range(nbins):
            windx = (bin_id ==i)
            n_spaxels_per_bin[i]=np.sum(windx)

        # Create empty mask in bin-level run: There are no masked bins, only masked spaxels!
        maskedSpaxel = np.zeros(nbins, dtype=bool)
        maskedSpaxel[:] = False

        # Prepare templates - This is for the stellar templates
        logging.info("Using full spectral library for ppxf on BIN level")
        (
            templates,
            lamRange_spmod,
            logLam_template,
            n_templates,
        ) = _prepareTemplates.prepareTemplates_Module(
            config,
            config["GAS"]["LMIN"],
            config["GAS"]["LMAX"],
            velscale / velscale_ratio,
            LSF_Data,
            LSF_Templates,
            "GAS",
        )[
            :4
        ]
        star_templates = templates.reshape((templates.shape[0], n_templates))

        offset = (logLam_template[0] - logLam_galaxy[0]) * C  # km/s
        # error        = np.ones((npix,nbins))
        ## --------------------- ##



    # define the velocity scale in kms
    #velscale = (logLam[1]-logLam[0])*cvel
    #do not use additive polynomials whehn determining emission line fluxes!!
    #adeg_lines=-1 # but this is hardcoded, so I don't think you need it

    # #--> generate the stellar templates --> Done above now
    # templates_info = util_templates.prepare_sp_templates\
    #     (configs, velscale, velscale_ratio, LSF_InterpolationFunction, LamRange, wav_in_vacuum)
    # star_templates,lamRange_spmod,logLam_template,min_wav_fit,max_wav_fit, nstpl = \
    #     templates_info['templates'], templates_info['WavRange'],templates_info['logLam'],\
    #     templates_info['min_wav_to_fit'],templates_info['max_wav_to_fit'],templates_info['templates'].shape[1]
    # if only_lines==True:
    #     # the correct way of only fitting emission lines would have been to
    #     # completely remove the stellar tempaltes from the components list
    #     # but it is simpler to just set them all to zero and keep the
    #     star_templates=star_templates*0.0
    #     # it's usually not good to fit without polynomials if you get rid of the stellar templates
    #     adeg_lines=3

    # --> generate the gas templates
    emldb=table.Table.read('./configFiles/'+config['GAS']['EMI_FILE'] , format='ascii') # Now using the PHANGS emission line config file. NB change '/configFiles' to dirPath or something like that
    # if wav_in_vacuum: # I dunno if we need this - will it ever be in a vaccumm?
    #     emldb['lambda'] = air_to_vac(emldb['lambda'])
    #eml_fwhm_angstr = LSF_InterpolationFunction(emldb['lambda']) #from PHANGS. Not needed anymore
    eml_fwhm_angstr = LSF_Templates(emldb['lambda'])
    # note that while the stellar templates are expanded in wavelength to cover +/- 150 Angstrom around the observed spectra (buffer)
    # emission line tempaltes are only generated for lines whose central wavelength lies within the min and max rest-frame waveelngth of the data
    gas_templates, gas_names, line_wave, eml_tying = \
        prepare_gas_templates.generate_emission_lines_templates(emldb, LamRange, config, logLam_template, eml_fwhm_angstr)
    ngastpl = gas_templates.shape[1]

    # --> stack vertically stellar and gas templates
    templates = np.column_stack([star_templates, gas_templates])
    # New stuff that you need later that has come from util_templates.py define_emission_line_input_for_ppxf
    tpl_comp = np.append(np.zeros(star_templates.shape[1], dtype=int), eml_tying['comp']+1)
    #total number of templates
    n_templ= len(tpl_comp)
    #total number of (kinematic) components
    n_comp = len(np.unique(tpl_comp))
    # select gas components
    gas_comp = tpl_comp>0
    # two moments per kinematics component
    moments = np.ones(n_comp, dtype=int)+1
    moments[0] = config['KIN']['MOM']
    moments[1:] = 2 # gas moments hardcoded for now
    # total number of moments
    n_tot_moments = np.sum(moments)
    #      Parse the velocity and sigma groups into tied parameters
    tied_flat = np.empty(n_tot_moments, dtype=object)
    tpl_index = np.arange(n_templ)

    tpl_vgrp = np.append(np.zeros(star_templates.shape[1], dtype=int), eml_tying['vgrp']+1)
    tpl_sgrp = np.append(np.zeros(star_templates.shape[1], dtype=int), eml_tying['sgrp']+1)

    for i in range(n_comp):
        # Do not allow tying to fixed components?
        if moments[i] < 0:
            continue
        # Velocity group of this component
        indx = np.unique(tpl_comp[tpl_index[tpl_vgrp == i]])
        if len(indx) > 1:
            parn = [ 0 + np.sum(np.absolute(moments[:j])) for j in indx ]
            tied_flat[parn[1:]] = 'p[{0}]'.format(parn[0])

        # Sigma group of this component
        indx = np.unique(tpl_comp[tpl_index[tpl_sgrp == i]])
        if len(indx) > 1:
            parn = [ 1 + np.sum(np.absolute(moments[:j])) for j in indx ]
            tied_flat[parn[1:]] = 'p[{0}]'.format(parn[0])

    tied_flat[[t is None for t in tied_flat ]] = ''

    # reshape the tied array so it matches the shape of start
    tied = []
    track = 0
    for i in range(n_comp):
        tied.append(list( tied_flat[track:track+np.absolute(moments[i])]) )
        track = track+np.absolute(moments[i])
 # Now, I have no way of knowing if the above is correct. I'm not sure if tied should be empty


    # --> cut the spectra to the wavelength range of the templates
    # I think the below is already taken care of.
    # logLam_cut, log_spec_cut, log_error_cut, npix, mask_for_original_spectra,wav_cov_templates =\
    #    util_templates.cut_spectra_to_match_templates(logLam, log_spec, log_error, config['GAS']['LMAX'],config['GAS']['LMIN'])

    # # --> Define goodpixels
    # wav_mask_ppxf, _ = util_templates.get_Mask\
    #         ('EMSLINES', configs['EMI_FILE'], configs['REDSHIFT'], velscale, logLam_cut,
    #         log_error_cut, wav_in_vacuum)
    #
    # # merge with the spectral coverage mask
    # mask_for_original_spectra [wav_cov_templates] =wav_mask_ppxf
    # # get the per-pixel mask by combining the wavelength mask with the info on the Err array
    # mask_pixels = util_templates.get_pixel_mask(wav_mask_ppxf, log_error_cut)
    # # finally transform to boolean for input to ppxf
    # mask_pixels = mask_pixels.astype(bool)

    #--> define the systemic velocity due to template and data starting wavelength offset
    # NOTE: this is only correct if velscale ==1!!
    offset = ((logLam_template[0] - logLam_galaxy[0]) + np.log(1 +\
         config['GENERAL']['REDSHIFT']/C ) )* C

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config['GAS']['FIXED'] == True:
        logging.info('Stellar kinematics are FIXED to the results obtained before.')
        # Set fixed option to True
        fixed = [True]*config['KIN']['MOM']

        # Read PPXF results, add not just stellar, but 3x gas guesses
        ppxf_data = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin.fits')[1].data
        #start = [[np.zeros((nbins, config['KIN']['MOM']))]] # old
        start, fixed = [], []
        for i in range(nbins):
            #start[i,:] = np.array( ppxf_data[i][:config['KIN']['MOM']] ) # old one (needs to be an array?)
            s = [ppxf_data[i][:config['KIN']['MOM']], [ppxf_data[i][0],50], [ppxf_data[i][0],50], [ppxf_data[i][0],50]] # Here I am setting the starting stellar kin guess to the stellar kin results, and the starting gas vel guess to the stellar vel and starting gas sigma to 50
            start.append(s)
            f = [[1,1,1,1],[0,0],[0,0],[0,0]] # Fix the stellar kinematics, but not the gas
            fixed.append(f)


    # Do *NOT* fix kinematics to those obtained previously
    elif config['GAS']['FIXED'] == False:
            logging.info('Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties.')
            # Set fixed option to False and use initial guess from Config-file
            fixed = None
            start = np.zeros((nbins, 2))
            start, fixed = [], []
            for i in range(nbins):
                #start[i,:] = np.array( [0.0, config['SFH']['SIGMA']] ) # old
                s = [[0,config['KIN']['SIGMA']], [0,50], [0,50], [0,50]] # Here the velocity guesses are all zero, the sigma guess is the stell kins sig guess for stars and 50 for the gas
                start.append(s)
                f = [[0,0,0,0],[0,0],[0,0],[0,0]] # Don't fix any of the kinematics because we're fitting them all!
                fixed.append(f)

    # Define goodpixels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! - this bit is from gist, but do I want to replace it?
    goodPixels_sfh = _auxiliary.spectralMasking(config, config['SFH']['SPEC_MASK'], logLam_galaxy)


    # # if you provide ppxf_results from a previous fit, you can now fix the
    # # kinematics to that of that previous fit - from PHANGS
    # if ppxf_results is not None:
    #     stellar_kinematics = np.copy(ppxf_results)
    #     nstmom =ppxf_results.shape[1]
    #     fixed_st_kin=True # if you want to test what happens you can set this to false, then
    #     #code will use ppxf_results just a starting guess, but not fix the kinematics
    #
    # #otherwise the stellar_kinematics array just cointains reasonable starting guesses
    # # but we allow ppxf to fit for stellar kinematics
    # else:
    #     nstmom =2
    #     stellar_kinematics =np.array([0, 100]*nbins).reshape(nbins, 2)
    #     fixed_st_kin=False


#    tpl_comp, moments, start, bounds, tied, gas_comp =\ # I don't think I need the below now???
#        util_templates.define_emission_line_input_for_ppxf(config, nstpl, ngastpl, eml_tying,emldb,
#        stellar_kinematics[0,:], fixed_st_kin=fixed_st_kin)

    n_gas_comp = 3 #len(np.unique(tpl_comp[gas_comp]))
    n_gas_templates =  ngastpl #len(tpl_comp[gas_comp]) ngastpl defined above

    # Array to store results of ppxf - come back to this, I don' think it's necc what I want?
    gas_kinematics         = np.zeros((nbins, n_gas_comp, 2))+np.nan # the 2 was configs['GAS_MOMENTS'], but I think we just want 2 for now
    gas_kinematics_err     = np.zeros((nbins, n_gas_comp, 2))  +np.nan
    chi2                   = np.zeros((nbins))
    gas_flux               = np.zeros((nbins,n_gas_templates))
    gas_flux_error         = np.zeros((nbins,n_gas_templates))
    gas_named              = []
    bestfit                = np.zeros((nbins,npix))
    gas_bestfit            = np.zeros((nbins,npix))
    stkin                  = np.zeros((nbins,nstmom))
    stkin_err              = np.zeros((nbins,nstmom))

    # ====================
    # Run PPXF
    start_time = time.time()

    if config['GENERAL']['PARALLEL'] == True:
        printStatus.running("Running PPXF for emission lines analysis in parallel mode")
        logging.info("Running PPXF for emission lines analysis in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
                for _ in range(config['GENERAL']['NCPU'])]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue
        for i in range(nbins):
            # this changes the stellar kinematics starting guess for each bin
            #start2 = np.copy(start)
            #start2[0]=stellar_kinematics[i, :]
            inQueue.put( ( i, templates, spectra[:,i], error[:,i], velscale,\
                start[i], goodPixels_sfh, tpl_comp, moments, offset, emi_mpol_deg, fixed[i], velscale_ratio,\
                tied, gas_comp,gas_names, nbins ) )

        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]

        # send stop signal to stop iteration
        for _ in range(config['GENERAL']['NCPU']):
            inQueue.put('STOP')

        # stop processes
        for p in ps:
            p.join()

        # Get output
        index = np.zeros(nbins)

        # i, sol, kin_err, chi2, gas_flux, gas_flux_err, gas_names, bestfit, gas_bestfit # old?
        # i, gas_sol, gas_error, chi2, gas_flux, gas_flux_error, gas_names, bestfit, gas_bestfit, star_sol, star_err  # new
        for i in range(0, nbins):
            index[i]                                    = ppxf_tmp[i][0]
            gas_kinematics[i,:, :]                      = ppxf_tmp[i][1]
            gas_kinematics_err[i,:, :]                  = ppxf_tmp[i][2]
            chi2[i]                                     = ppxf_tmp[i][3]
            gas_flux[i,:]                               = ppxf_tmp[i][4]
            gas_flux_error[i,:]                         = ppxf_tmp[i][5]
            gas_named.append(ppxf_tmp[i][6])                            # Let's see if this works!
            bestfit[i,:]                                = ppxf_tmp[i][7]
            gas_bestfit[i,:]                            = ppxf_tmp[i][8]
            stkin[i,:]                                  = ppxf_tmp[i][9]
            stkin_err[i, :]                             = ppxf_tmp[i][10]

        # Sort output
        argidx = np.argsort( index )
        gas_kinematics         = gas_kinematics[argidx,:, :]
        gas_kinematics_err     = gas_kinematics_err[argidx,:, :]
        chi2                   = chi2[argidx]
        gas_flux               = gas_flux[argidx,:]
        gas_flux_error         = gas_flux_error[argidx,:]
        gas_named              = gas_names[argidx]
        bestfit                = bestfit[argidx,:]
        gas_bestfit            = gas_bestfit[argidx,:]
        stkin                  =stkin[argidx, :]
        stkin_err              =stkin_err[argidx, :]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)


    elif config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            #start[0]=stellar_kinematics[i, :]
            start[0]=start[i] # Added this in. Check if ok.

            gas_kinematics[i,:, :], kinematics_all_err[i,:, :],\
                chi2[i], gas_flux[i,:],gas_flux_error[i,:], _ ,\
                bestfit[i,:], gas_bestfit[i,:] , stkin[i,:], stkin_err[i,:]  = \
                run_ppxf(templates, spectra[:,i], error[:,i], velscale, \
                start[i], goodPixels_sfh, tpl_comp, moments, offset, emi_mpol_deg, \
                fixed[i], velscale_ratio, tied, gas_comp, gas_names, i, nbins)


        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print("             Running PPXF on %s spectra took %.2fs" % (nbins, time.time() - start_time))
    #print("")
    logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))

    # Check if there was a problem with a spectra: NOT DONE

    # # add back the part of the spectrum that was truncated because of lack of templates - needed?
    # bestfit_1 = np.zeros((nbins,npix_in))
    # gas_bestfit_1 = np.zeros((nbins,npix_in))
    # bestfit_1[:, wav_cov_templates]=bestfit
    # gas_bestfit_1[:, wav_cov_templates]=gas_bestfit

    # tidy up the ppXF output so it matches the order to the original line-list
    linesfitted, fluxes_final, fluxes_err_final, vel_final,vel_err_final, \
        sigma_final,sigma_err_final, extra= tidy_up_fluxes_and_kinematics(gas_kinematics,
        gas_kinematics_err,gas_flux, gas_flux_error,emldb, eml_tying, config['GAS']['MOM'])

    # get fluxes in the correct units, see Westfall et al. 2019 eq 16
    gas_flux_in_units = fluxes_final*(velscale/C)*\
        linesfitted['lambda']*(1+config['GENERAL']['REDSHIFT']/C )
    gas_err_flux_in_units = fluxes_err_final*(velscale/C)*\
        linesfitted['lambda']*(1+config['GENERAL']['REDSHIFT']/C )
    # divide by the number of spaxels per bin to make the flux per spaxel
    for i in range(gas_flux_in_units.shape[0]):
        gas_flux_in_units[i, :]= gas_flux_in_units[i, :]/n_spaxels_per_bin[i]
        gas_err_flux_in_units[i, :]= gas_err_flux_in_units[i, :]/n_spaxels_per_bin[i]

    # add back the template LSF
    eml_fwhm_angstr = LSF_Templates(linesfitted['lambda'])
    templates_sigma = eml_fwhm_angstr/\
        linesfitted['lambda']*C/2.355

   # templates_sigma = np.zeros(sigma_final.shape)+templates_sigma
    sigma_final_measured  = (sigma_final**2 + templates_sigma**2)**(0.5)

    # save results to file
    save_ppxf_emlines(config, config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'], config['GAS']['LEVEL'], linesfitted,
        gas_flux_in_units, gas_err_flux_in_units,vel_final, vel_err_final,
        sigma_final_measured, sigma_err_final, chi2, templates_sigma, bestfit, gas_bestfit, stkin, spectra, error, logLam_galaxy, ubins, npix, extra)


    printStatus.updateDone("Emission line fitting done")
   #print("")
    logging.info("Emission Line Fitting done\n")
