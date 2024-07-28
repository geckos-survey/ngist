from scipy import special, fftpack
from astropy.modeling import FittableModel, Parameter
import numpy as np
# gas_templates.py
# A collection of classes and functions to create some Gaussian emission line templates
# New for gist-geckos
# A mix of the ppxf_example_population_gas_sdss.py and the PHANGS DAP module
# Mainly taken from:
# https://gitlab.com/francbelf/ifu-pipeline/-/blob/master/TardisPipeline/utilities/util_ppxf_emlines.py
# https://gitlab.com/francbelf/ifu-pipeline/-/blob/master/TardisPipeline/utilities/util_lineprofiles.py
# emissionLines.config has been updated to https://gitlab.com/francbelf/ifu-pipeline/-/blob/master/TardisPipeline/Templates/configurationTemplates/emission_lines.setup

# Physical constants
C = 299792.458 # speed of light in km/s

class GaussianLSF:
    r"""
    Define a Gaussian line profile, *sampled* over the width of the
    sampling step, parameterized by its integral (:math:`F`), center
    (:math:`\mu`), and standard deviation (:math:`\sigma`). I.e:

    .. math::

        \mathcal{N}(x|f,\mu,\sigma) = \frac{f}{\sqrt{2\pi}\sigma}
        \exp\left(\frac{-\Delta^2}{2\sigma^2}\right)

    where :math:`\Delta = x-\mu`.  The coordinate vector :math:`x` does
    not need to be uniformly sampled.

    Args:
        p (array-like): (**Optional**) Input parameters ordered as the
            total integral of the profile, the profile center, and the
            profile standard deviation.  Assumed to be (1.0, 0.0, 1.0)
            by default.

    Attributes:
        p (numpy.ndarray): Most recently used parameters

    Raises:
        ValueError: Raised if the provided parameter vector is not 3
            elements long.
    """
    def __init__(self, p=None):
        self.set_par(p)


    def __call__(self, x, p):
        """Calculate the profile.

        Args:
            x (array-like): Independent variable.
            p (array-like): LSF parameters.
        """
        self.set_par(p)
        return self.sample(x)


    @staticmethod
    def npar():
        return 3


    def set_par(self, p):
        """
        Set the internal parameters to the provided set.

        Args:
            p (array-like): LSF parameters.

        Raises:
            ValueError: Raised if the provided parameter vector is not 3
                elements long.
        """
        if p is None:
            self.p = np.array([1.0, 0.0, 1.0])
            return
        if len(p) != GaussianLSF.npar():
            raise ValueError('Must provide {0} parameters.'.format(GaussianLSF.npar()))
        self.p = np.asarray(p)


    def sample(self, x):
        """
        Sample the profile.

        Args:
            x (array-like): Independent variable.
        """
        return self.p[0] * np.exp(-np.square((x-self.p[1])/self.p[2])/2.) \
                    / np.sqrt(2.0*np.pi) / self.p[2]


    def parameters_from_moments(self, mom0, mom1, mom2):
        """
        Provided the 0th, 1st, and 2nd moments, produce a set of
        parameters for the profile.
        """
        return np.array([mom0, mom1, mom2])



def _find_primary(emldb, i, primary=False):
        """
        Return the index of the line to which this one is tied.

        The result may be that this line is tied to one that is also
        tied to a second line.  If that's the case, the ``primary``
        keyword can be use to trace the parameter tying back to the
        independent line.

        Arg:
            i (int): The index of the line in the database.
            primary (bool): (**Optional**) Trace the line tying all the
                way back to the independent (primary) line.

        Return:
            int: The index of the line to which this one is tied.

        Raises:
            ValueError: Raised if the primary option is selected and the
                line does not trace back to a primary line.  This
                represents a poorly constructed emission-line database
                and should be avoided!
        """
        db_rows = np.arange(len(emldb))
        indx = db_rows[emldb['index'] == int(emldb['mode'][i][1:])][0]
        if not primary:
            return indx
        max_iter = 100
        j = 0
        while emldb['mode'][indx] != 'f' and j < max_iter:
            indx = db_rows[emldb['index'] == int(emldb['mode'][indx][1:])][0]
            j+=1
        if j == max_iter:
            raise ValueError('Line {0} (index={1}) does not trace back to a primary line!'.format(
                                i, emldb['index'][i]))
        return indx

class FFTGaussianLSF(GaussianLSF):
    r"""

    Define a Gaussian line profile by first constructing the analytic
    FFT of the profile and then returning the inverse real FFT.  See
    ppxf_util.emline by M. Cappellari.  The sampling *must* be uniform
    in :math:`x`.

    Args:
        p (array-like): (**Optional**) Input parameters ordered as the
            total integral of the profile, the profile center, and the
            profile standard deviation.  Assumed to be (1.0, 0.0, 1.0)
            by default.
        dx (float): (**Optional**) Sampling width.  Default is 1.
        pixel (bool): (**Optional**) Flag to produce profile integrated
            over the sampling width.

    Attributes:
        p (numpy.ndarray): Most recently used parameters
        dx (float): Assumed sampling.
        pixel (bool): Flag to produce profile integrated over the
            sampling width.

    Raises:
        ValueError: Raised if the provided parameter vector is not 3
            elements long.
    """
    def __init__(self, p=None, dx=None, pixel=True):
        self.set_par(p)
        self.dx = 1.0 if dx is None else dx
        self.pixel = pixel


    def sample(self, x):
        """
        Sample the profile.

        .. warning::
            Does **not** check if the provided :math:`x` values are
            sampled at :attr:`dx`.

        Args:
            x (array-like): Independent variable.
        """
        xsig = self.p[2]/self.dx
        x0 = (self.p[1]-x[0])/self.dx
        npad = fftpack.next_fast_len(x.size)
        w = np.linspace(0,np.pi,npad//2+1)
        rfft = self.p[0]*np.exp(-0.5*np.square(w*xsig) - 1j*w*x0)
        if self.pixel:
            rfft *= np.sinc(w/(2*np.pi))
        lsf = np.fft.irfft(rfft, n=npad)[:x.size]
        return lsf if self.pixel else lsf/self.dx

def generate_emission_lines_templates(emldb, LamRange, config, logLam, eml_fwhm_angstr):

    wave = np.exp(logLam)

    # ignore lines that fall outside the wavelength range covered by the data (in rest-frame)
    # note that the stellar templates extend further becauxe of the 150 km/s buffer on either side of the data
    extra_ignore = (emldb['lambda'] > LamRange[1]/(1+ config['GENERAL']['REDSHIFT']/C)   ) | (emldb['lambda']< LamRange[0]/(1+ config['GENERAL']['REDSHIFT']/C) )
    emldb['action'][extra_ignore] = 'i'

    # Get the list of lines to ignore
    ignore_line = emldb['action'] != 'f'

    # The total number of templates to construct is the number of
    # lines in the database minus the number of lines with mode=aN
    tied_all = np.array([m[0] == 'a' for m in emldb['mode']])
    nlinesdb=len(emldb)
    ntpl = nlinesdb - np.sum(ignore_line) - np.sum(tied_all)

    # Initialize the components
    comp = np.zeros(ntpl, dtype=int)-1
    vgrp = np.zeros(ntpl, dtype=int)-1
    sgrp = np.zeros(ntpl, dtype=int)-1

    # All the primary lines go into individual templates, kinematic
    # components, velocity groups, and sigma groups
    tpli = np.zeros(len(emldb), dtype=int)-1
    primary_line = (emldb['mode'] == 'f') & np.invert(ignore_line)

    nprimary = np.sum(primary_line)
    tpli[primary_line] = np.arange(nprimary)
    comp[:nprimary] = np.arange(nprimary)
    vgrp[:nprimary] = np.arange(nprimary)
    sgrp[:nprimary] = np.arange(nprimary)

    # some lines are not primary nor are being ignored
    # this means they are tied in some way
    finished = primary_line | ignore_line
    while np.sum(finished) != nlinesdb:
        # Find the indices of lines that are tied to finished lines
        start_sum = np.sum(finished)

        for i in range(nlinesdb):

            if finished[i]:
                continue
            indx = _find_primary(emldb, i)
            if not finished[indx]:
                continue

            finished[i] = True

            # Mode=a: Line is part of an existing template
            if emldb['mode'][i][0] == 'a':
                tpli[i] = tpli[indx]
            # Mode=k: Line is part of a different template but an
            # existing kinematic component
            if emldb['mode'][i][0] == 'k':
                tpli[i] = np.amax(tpli)+1
                comp[tpli[i]] = comp[tpli[indx]]
                vgrp[tpli[i]] = vgrp[tpli[indx]]
                sgrp[tpli[i]] = sgrp[tpli[indx]]
            # Mode=v: Line is part of a different template and
            # kinematic component with an untied sigma, but tied to
            # an existing velocity group
            if emldb['mode'][i][0] == 'v':
                tpli[i] = np.amax(tpli)+1
                comp[tpli[i]] = np.amax(comp)+1
                sgrp[tpli[i]] = np.amax(sgrp)+1
                vgrp[tpli[i]] = vgrp[tpli[indx]]
            # Mode=s: Line is part of a different template and
            # kinematic component with an untied velocity, but tied
            # to an existing sigma group
            if emldb['mode'][i][0] == 's':
               tpli[i] = np.amax(tpli)+1
               comp[tpli[i]] = np.amax(comp)+1
               vgrp[tpli[i]] = np.amax(vgrp)+1
               sgrp[tpli[i]] = sgrp[tpli[indx]]

        # If the loop ends up with the same number of parsed lines
        # that it started with, there must be an error in the
        # construction of the input database.
        if start_sum == np.sum(finished):
            raise ValueError('Unable to parse the input database.  Check tying parameters.')

    # Debug:
    if np.any(comp < 0) or np.any(vgrp < 0) or np.any(sgrp < 0):
        raise ValueError('Templates without an assigned component.  Check the input database.')

    #put everything in a dictionary
    eml_tying = {'tpli':tpli, 'comp':comp, 'vgrp':vgrp, 'sgrp':sgrp }

    _dw = np.diff(logLam)[0]
    _restwave  = np.log(emldb['lambda'])

    # Rest wavelength in pixel units
    _restwave = (_restwave - logLam[0])/_dw

    # Flux to pixel units; less accurate when spectrum is
    # logarithmically binned
    dl = emldb['lambda']*(np.exp(_dw/2)-np.exp(-_dw/2))
    _flux = emldb['A_i']

    # Dispersion in pixel units
    _sigma = eml_fwhm_angstr/dl/2.355

    # Construct the templates
    pix = np.arange(wave.size)
    flux = np.zeros((ntpl,wave.size), dtype=float)
    gas_names = []
    line_wave=[]
    for i in range(ntpl):
        # Find all the lines associated with this template:
        wtemp=tpli == i
        index = np.arange(nlinesdb)[wtemp]
        gas_names.append(emldb['name'][wtemp][0])
        line_wave.append(emldb['lambda'][wtemp][0])
        # Add each line to the template
        for j in index:
            # Declare an instance of the desired profile
            profile = FFTGaussianLSF()
            # Use the first three moments of the line to set the
            # parameters
            p = profile.parameters_from_moments(_flux[j], _restwave[j], _sigma[j])
            # Add the line to the flux in this template
            flux[i,:] += profile(pix, p)


    gas_templates, gas_names, line_wave = flux, np.array(gas_names), \
        np.array(line_wave)
    return(gas_templates.T, gas_names, line_wave, eml_tying)
