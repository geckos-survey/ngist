from astropy.io import fits
from astropy.table import Table
import yaml
import pandas as pd
import numpy as np
import os
import h5py



def table2pandas(table, addid=True):
    df = table.to_pandas()
    if addid:
        df['BIN_ID'] = df.index
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(3)
    return df

def load_config(directory):
    with open(directory+'/CONFIG', "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    conf_df = pd.DataFrame(columns=['Module', 'Configs', 'Values'])
    for module in conf.keys():
        config_i = 0
        for config in conf[module].keys():
            if config_i == 0:
                new = pd.DataFrame(columns=conf_df.columns, data=[[module, config, conf[module][config]]])
                conf_df = pd.concat([conf_df, new], axis=0)
            else:
                new = pd.DataFrame(columns=conf_df.columns, data=[['', config, conf[module][config]]])
                conf_df = pd.concat([conf_df, new], axis=0)
            config_i += 1
    return conf_df



class gistDataBase():
    """ Load all available data to make it accessible for plotting. """


    def __init__(self, settings=None):

        # Some default settings
        self.initialization(settings)

        # Setup the rest


    def dialogRunSelection(self, tmp0):
        """
        Select the output directory of the run to be displayed. Basic checks
        and corrections are applied.
        """
        # tmp0 = '/home/zwan0382/Documents/projects/Mapviewer_web/resultsRevisedREr5/'
        if len( tmp0 ) > 0:
            if tmp0.split('/')[-1] == 'maps':
                tmp0 = tmp0[:-5]
            tmp1 = tmp0.split('/')[-1]
            self.dirprefix = os.path.join(tmp0,tmp1)
            self.directory = tmp0+'/'

            # self.loadData()
            # self.createWindow()


    def loadData(self, path_gist_run):
        self.dialogRunSelection(path_gist_run)
        self.CONFIG_df = load_config(path_gist_run)


       # Check if *_table.fits is available. This file is required!
        # if os.path.isfile(self.dirprefix+'_table.fits') == False:
        #     print("You did not select a valid GIST output directory.")
        #     self.dialogRunSelection()

        # Determine which data is available
        self.MASK = os.path.isfile(self.dirprefix+'_mask.fits')

        self.KIN = os.path.isfile(self.dirprefix+'_kin.fits')

        # Determine weather its gistPipeline or nGIST


        if (os.path.isfile(self.dirprefix+'_gas_BIN.fits') == False  and os.path.isfile(self.dirprefix+'_gas_SPAXEL.fits') == False) or \
           (os.path.isfile(self.dirprefix+'_gas_BIN.fits') == False  and os.path.isfile(self.dirprefix+'_gas_SPAXEL.fits') == False):
            self.GAS = False
            self.gasLevelAvailable = []
        if (os.path.isfile(self.dirprefix+'_gas_BIN.fits') == True   and  os.path.isfile(self.dirprefix+'_gas_SPAXEL.fits') == False) or \
           (os.path.isfile(self.dirprefix+'_gas_bin.fits') == True   and  os.path.isfile(self.dirprefix+'_gas_spaxel.fits') == False):
            self.GAS = True
            self.gasLevelAvailable = ['BIN']
        if (os.path.isfile(self.dirprefix+'_gas_BIN.fits') == False  and  os.path.isfile(self.dirprefix+'_gas_SPAXEL.fits') == True) or \
           (os.path.isfile(self.dirprefix+'_gas_bin.fits') == False  and  os.path.isfile(self.dirprefix+'_gas_spaxel.fits') == True):
            self.GAS = True
            self.gasLevelAvailable = ['SPAXEL']
        if (os.path.isfile(self.dirprefix+'_gas_BIN.fits') == True   and  os.path.isfile(self.dirprefix+'_gas_SPAXEL.fits') == True) or \
           (os.path.isfile(self.dirprefix+'_gas_bin.fits') == True   and  os.path.isfile(self.dirprefix+'_gas_spaxel.fits') == True):
            self.GAS = True
            self.gasLevelAvailable = ['BIN', 'SPAXEL']

        if len( self.gasLevelAvailable ) == 1:
            self.gasLevel = self.gasLevelAvailable[0]
            self.gasLevel_onlyBIN = True
        else:
            self.gasLevel = self.gasLevelSelected
            self.gasLevel_onlyBIN = False

        self.SFH           = os.path.isfile(self.dirprefix+'_sfh.fits')

        if (os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == False) or \
           (os.path.isfile(self.dirprefix+'_ls_orig_res.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_adap_res.fits') == False):
            self.LS = False
            self.LsLevelAvailable = []
        if (os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == False) or \
           (os.path.isfile(self.dirprefix+'_ls_orig_res.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_adap_res.fits') == False):
            self.LS = True
            self.LsLevelAvailable = ['ORIGINAL']
        if (os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == True) or \
           (os.path.isfile(self.dirprefix+'_ls_orig_res.fits') == False  and  os.path.isfile(self.dirprefix+'_ls_adap_res.fits') == True):
            self.LS = True
            self.LsLevelAvailable = ['ADAPTED']
        if (os.path.isfile(self.dirprefix+'_ls_OrigRes.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_AdapRes.fits') == True) or \
           (os.path.isfile(self.dirprefix+'_ls_orig_res.fits') == True  and  os.path.isfile(self.dirprefix+'_ls_adap_res.fits') == True):
            self.LS = True
            self.LsLevelAvailable = ['ORIGINAL', 'ADAPTED']

        if len( self.LsLevelAvailable ) == 1:
            self.LsLevel = self.LsLevelAvailable[0]
            self.LsLevel_onlyORIGINAL = True
        else:
            self.LsLevel = self.LsLevelSelected
            self.LsLevel_onlyORIGINAL = False



        # ======================================================== #
        #                    R E A D   D A T A                     #
        # ======================================================== #

        # Read table and get transformation array
        self.table     = fits.open(self.dirprefix+'_table.fits')[1].data
        self.table_df  = table2pandas(Table(self.table), addid=False)
        self.pixelsize = fits.open(self.dirprefix+'_table.fits')[0].header['PIXSIZE']
        _, idxConvertShortToLong = np.unique( np.abs(self.table.BIN_ID), return_inverse=True )


        # Read spectra
        try:
            with h5py.File(self.dirprefix+'_bin_spectra.hdf5', 'r') as f:
                self.Spectra = f['SPEC'][:].T
                self.Lambda = f['LOGLAM'][:]
        except:
            with h5py.File(self.dirprefix+'_BinSpectra.hdf5', 'r') as f:
                self.Spectra = f['SPEC'][:].T
                self.Lambda = f['LOGLAM'][:]
        self.LambdaLIN  = np.exp(self.Lambda)
        nbins           = self.Spectra.shape[0]
        if self.gasLevel == 'SPAXEL':
            self.AllSpectra = fits.open(self.dirprefix+'_AllSpectra.fits')[1].data.SPEC


        # Read mask
        if self.MASK == True:
            self.Mask = fits.open(self.dirprefix+'_mask.fits')[1].data
            self.Mask_df  = table2pandas(Table(self.Mask), addid=False)


        # Read stellar kinematics
        if self.KIN == True:
            self.kinResults_Vorbin = fits.open(self.dirprefix+'_kin.fits')[1].data
            self.kinResults_Vorbin_df = table2pandas(Table(self.kinResults_Vorbin))
            self.kinResults = self.kinResults_Vorbin[idxConvertShortToLong]
            try:
                self.kinBestfit = fits.open(self.dirprefix+'_kin-bestfit.fits')[1].data.BESTFIT
                self.kinLambda  = fits.open(self.dirprefix+'_kin-bestfit.fits')[2].data.LOGLAM
                self.kinLambdaLIN  = np.exp(self.kinLambda)
                self.kinGoodpix = fits.open(self.dirprefix+'_kin-bestfit.fits')[3].data.GOODPIX
            except:
                self.kinBestfit = fits.open(self.dirprefix+'_kin_bestfit.fits')[1].data.BESTFIT
                self.kinLambda  = fits.open(self.dirprefix+'_kin_bestfit.fits')[2].data.LOGLAM
                self.kinLambdaLIN  = np.exp(self.kinLambda)
                self.kinGoodpix = fits.open(self.dirprefix+'_kin_bestfit.fits')[3].data.GOODPIX

            # median_V_stellar   = np.nanmedian( self.kinResults.V[np.where( self.table.BIN_ID >= 0 )[0]] )
            # self.kinResults.V = self.kinResults.V - median_V_stellar
        else:
            self.kinResults_Vorbin = None
            self.kinResults_Vorbin_df = None
            self.kinResults = None
            self.kinBestfit = None
            self.kinLambda  = None
            self.kinLambdaLIN  = None
            self.kinGoodpix = None


        # Read emissionLines results
        if self.GAS == True:
            if os.path.isfile(self.dirprefix+"_gas-cleaned_BIN.fits") == True:
                self.EmissionSubtractedSpectraBIN    = np.array( fits.open(self.dirprefix+"_gas-cleaned_BIN.fits")[1].data.SPEC )
            elif os.path.isfile(self.dirprefix+"_gas_cleaned_bin.fits") == True:
                self.EmissionSubtractedSpectraBIN    = np.array( fits.open(self.dirprefix+"_gas_cleaned_bin.fits")[1].data.SPEC )
            if os.path.isfile(self.dirprefix+"_gas-cleaned_SPAXEL.fits") == True:
                self.EmissionSubtractedSpectraSPAXEL = np.array( fits.open(self.dirprefix+"_gas-cleaned_SPAXEL.fits")[1].data.SPEC )
            elif os.path.isfile(self.dirprefix+"_gas_cleaned_spaxel.fits") == True:
                self.EmissionSubtractedSpectraSPAXEL = np.array( fits.open(self.dirprefix+"_gas_cleaned_spaxel.fits")[1].data.SPEC )

            try:
                gas              = fits.open(self.dirprefix+'_gas_'+self.gasLevel+'.fits')[1].data
                for i in range(len(gas.columns)):
                    gas.columns[i].name = gas.columns[i].name.replace('.', '_')
                self.gasBestfit  = fits.open(self.dirprefix+'_gas-bestfit_'+self.gasLevel+'.fits')[1].data.BESTFIT
                self.gasLambda   = fits.open(self.dirprefix+'_gas-bestfit_'+self.gasLevel+'.fits')[2].data.LOGLAM
                self.gasLambdaLIN  = np.exp(self.gasLambda)
                self.gasGoodpix  = fits.open(self.dirprefix+'_gas-bestfit_'+self.gasLevel+'.fits')[3].data.GOODPIX
            except:
                gas              = fits.open(self.dirprefix+'_gas_'+self.gasLevel.lower()+'.fits')[1].data
                for i in range(len(gas.columns)):
                    gas.columns[i].name = gas.columns[i].name.replace('.', '_')
                self.gasBestfit  = fits.open(self.dirprefix+'_gas_bestfit_'+self.gasLevel.lower()+'.fits')[1].data.BESTFIT
                self.gasLambda   = fits.open(self.dirprefix+'_gas_bestfit_'+self.gasLevel.lower()+'.fits')[2].data.LOGLAM
                self.gasLambdaLIN  = np.exp(self.gasLambda)
                self.gasGoodpix  = fits.open(self.dirprefix+'_gas_bestfit_'+self.gasLevel.lower()+'.fits')[3].data.GOODPIX

            if self.gasLevel == 'BIN':
                self.gasResults_Vorbin = gas
                self.gasResults_Vorbin_df = table2pandas(Table(self.gasResults_Vorbin))
                self.gasResults = gas[idxConvertShortToLong]
            if self.gasLevel == 'SPAXEL':
                self.gasResults_Vorbin = None
                self.gasResults_Vorbin_df = None
                self.gasResults = gas
            self.gasResults_Vorbin_df.columns = self.gasResults_Vorbin_df.columns.str.replace(".", "_")

            # for name in self.gasResults.names:
            #     if name.split('_')[-1] == 'V':
            #         self.gasResults[name] = self.gasResults[name] - median_V_stellar
        else:
            self.gasResults_Vorbin = None
            self.gasResults_Vorbin_df = None
            self.gasResults = None
            self.gasBestfit = None
            self.gasLambda  = None
            self.gasLambdaLIN = None
            self.gasGoodpix = None


        # Read starFormatioHistories results
        if self.SFH == True:
            self.sfhResults_Vorbin = fits.open(self.dirprefix+'_sfh.fits')[1].data
            self.sfhResults_Vorbin_df = table2pandas(Table(self.sfhResults_Vorbin))
            self.sfhResults = self.sfhResults_Vorbin[idxConvertShortToLong]
            try:
                self.sfhBestfit = fits.open(self.dirprefix+'_sfh-bestfit.fits')[1].data.BESTFIT
                self.sfhLambda  = fits.open(self.dirprefix+'_sfh-bestfit.fits')[2].data.LOGLAM
                self.sfhLambdaLIN  = np.exp(self.sfhLambda)
                self.sfhGoodpix = fits.open(self.dirprefix+'_sfh-bestfit.fits')[3].data.GOODPIX
            except:
                self.sfhBestfit = fits.open(self.dirprefix+'_sfh_bestfit.fits')[1].data.BESTFIT
                self.sfhLambda  = fits.open(self.dirprefix+'_sfh_bestfit.fits')[2].data.LOGLAM
                self.sfhLambdaLIN  = np.exp(self.sfhLambda)
                self.sfhGoodpix = fits.open(self.dirprefix+'_sfh_bestfit.fits')[3].data.GOODPIX

            # if 'V' in self.sfhResults.names:
            #     self.sfhResults.V = self.sfhResults.V - median_V_stellar

            # Read the age, metallicity and [Mg/Fe] grid
            try:
                grid        = fits.open(self.dirprefix+'_sfh-weights.fits')[2].data
            except:
                grid        = fits.open(self.dirprefix+'_sfh_weights.fits')[2].data
            self.metals = np.unique(grid.METAL)
            self.age    = np.power(10, np.unique(grid.LOGAGE))

            # Read weights
            try:
                hdu_weights  = fits.open(self.dirprefix+'_sfh-weights.fits')
            except:
                hdu_weights  = fits.open(self.dirprefix+'_sfh_weights.fits')
            nAges        = hdu_weights[0].header['NAGES']
            nMetal       = hdu_weights[0].header['NMETAL']
            nAlpha       = hdu_weights[0].header['NALPHA']
            self.Weights = np.reshape(np.array(hdu_weights[1].data.WEIGHTS), (nbins,nAges,nMetal,nAlpha))
            self.Weights = np.transpose(self.Weights, (0,2,1,3))
        else:
            self.sfhResults_Vorbin = None
            self.sfhResults_Vorbin_df = None
            self.sfhResults = None
            self.sfhBestfit = None
            self.sfhLambda  = None
            self.sfhLambdaLIN = None
            self.sfhGoodpix = None
            self.metals     = None
            self.age        = None
            self.Weights    = None


        # Read lineStrengths results
        if self.LS == True:
            if self.LsLevel == "ORIGINAL":
                try:
                    ls = fits.open(self.dirprefix+'_ls_OrigRes.fits')[1].data
                except:
                    ls = fits.open(self.dirprefix+'_ls_orig_res.fits')[1].data
            elif self.LsLevel == "ADAPTED":
                try:
                    ls = fits.open(self.dirprefix+'_ls_AdapRes.fits')[1].data
                except:
                    ls = fits.open(self.dirprefix+'_ls_adap_res.fits')[1].data
            self.lsResults_Vorbin = ls
            self.lsResults_Vorbin_df = table2pandas(Table(self.lsResults_Vorbin))
            self.lsResults = ls[idxConvertShortToLong]
        else:
            self.lsResults_Vorbin = None
            self.lsResults_Vorbin_df = None
            self.lsResults = None


    def reset(self, settings):
        keys = list(self.__dict__.keys())
        if len(keys) > 0:
            for key in keys:
                delattr(self, key)

        self.initialization(settings)

    def initialization(self, settings=None):
        if settings == None:
            self.restrict2voronoi = 2
            self.gasLevelSelected = 'BIN'
            self.LsLevelSelected  = 'ADAPTED'
            self.AoNThreshold     = 3
        else:
            self.restrict2voronoi = settings['restrict2voronoi']
            self.gasLevelSelected = settings['gasLevelSelected']
            self.LsLevelSelected  = settings['LsLevelSelected']
            self.AoNThreshold     = settings['AoNThreshold']
