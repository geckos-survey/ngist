import copy
import glob
import math as mt
import os
import pdb
import sys
from os import path
from pprint import pprint

import astropy.convolution as ap_c
import astropy.io.fits as fits
import astropy.table as table
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp_s
from matplotlib.font_manager import FontProperties
from numpy.polynomial import hermite, legendre
from ppxf.ppxf import robust_sigma
from scipy import fftpack, linalg, ndimage, optimize
from scipy.interpolate import interp1d

from ngistPipeline.emissionLines.magpiGandalf.cap_mpfit import mpfit

# ---------------------------------------------------------------------------- #
# This version has been modified from the original to compute uncertainties on the
# derived emission line parameters using Monte Carlo realisations of the best fit data
# last edited by JTM on 2021-05-25
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Version 7 of pPXF does not contain the _bvls_solve and nnls_flags functions
# anymore. However, the current implementation of pyGandALF depends on these
# functions. In order to assure the compatibility of the GIST framework with
# the most recent version of pPXF, we include these two functions from pPXF
# version 6 below. We are grateful to Michele Cappellari for his permission to
# include these functions.


def nnls_flags(A, b, npoly):
    """
    Solves min||A*x - b|| with
    x[j] >= 0 for j >= npoly
    x[j] free for j < npoly
    where A[m, n], b[m], x[n], flag[n]

    """
    m, n = A.shape
    AA = np.hstack([A, -A[:, :npoly]])
    x = optimize.nnls(AA, b)[0]
    x[:npoly] -= x[n:]
    return x[:n]


def _bvls_solve(A, b, npoly):
    # No need to enforce positivity constraints if fitting one single template:
    # use faster linear least-squares solution instead of NNLS.
    m, n = A.shape
    if m == 1:  # A is a vector, not an array
        soluz = A.dot(b) / A.dot(A)
    elif n == npoly + 1:  # Fitting a single template
        soluz = linalg.lstsq(A, b)[0]
    else:  # Fitting multiple templates
        soluz = nnls_flags(A, b, npoly)
    return soluz


# ---------------------------------------------------------------------------- #

#################################################################
# global
C = np.float64(299792.458)


#################################################################
# Class to store Emission Setup
class EmissionSetup:
    def __init__(self, i, name, _lambda, action, kind, a, v, s, fit, aon):
        self.i = int(i)
        self.name = name
        self._lambda = np.float64(_lambda)
        self.action = action
        self.kind = kind
        self.a = np.float64(a)
        self.v = np.float64(v)
        self.s = np.float64(s)
        self.fit = fit
        self.aon = aon

    def __repr__(self):
        return (
            "\n{i:%2d  name:%8s  lambda:%6.3f  action:%2s  kind:%3s  a:%4.6f  v:%4.2f  s:%4.2f  fit:%3s  aon:%.1f}"
            % (
                self.i,
                self.name,
                self._lambda,
                self.action,
                self.kind,
                self.a,
                self.v,
                self.s,
                self.fit,
                self.aon,
            )
        )


#################################################################
def fullprint(*args, **kwargs):
    # Print the FULL content of an array in a "pretty" way
    from pprint import pprint

    opt = np.get_printoptions()
    np.set_printoptions(threshold="nan")
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)


#################################################################
def where_eq(data, attr, value):
    # Return an array of indexes where attribute == value in data
    index = []
    for i, elem in enumerate(data):
        if getattr(elem, attr) == value:
            index.append(i)
    return index


#################################################################
def err_msg_exit(txt):
    # Print and error message and exit (abort execution)
    print("ERROR: " + txt)
    sys.exit(-1)


#################################################################
def load_emission_setup(data):
    # Load data of emission setup from a flat array and return an
    # array of classes
    out = []
    for elem in data:
        out.append(EmissionSetup(*elem))
    return out


################################################################################
#
def rebin(x, factor):
    """
    Rebin a vector, or the first dimension of an array,
    by averaging within groups of "factor" adjacent values.
    """
    if factor == 1:
        xx = x
    else:
        xx = x.reshape(len(x) // factor, factor, -1).mean(1).squeeze()
    #
    return xx


###############################################################################
def show_fit(
    galaxy,
    bestfit,
    emission,
    best_pars,
    sol,
    goodpixels,
    mdegree,
    reddening,
    l0_gal,
    lstep_gal,
    l0_templ,
    log10,
    kinstars,
    nlines,
):
    # in progress...
    plt.close()
    plt.ion()
    # plot final data-model comparison if required.
    # the colors below were chosen for the sauron colormap.
    npix = len(galaxy)
    # Configure plot
    mn = min(galaxy)
    mx = max(galaxy)
    mn -= 0.1 * (mx - mn)
    resid = mn + galaxy - bestfit
    y1 = min(resid[goodpixels])
    y2 = mx
    diff = y2 - y1
    y1 = y1 - 0.15 * diff
    y2 = y2 + 0.1 * diff
    ax = plt.gca()
    try:
        ax.set_facecolor("black")
    except:
        # set_axis_bcgolor is deprecated since version 2.0
        # Use it only if set_facecolor is not available...
        ax.set_axis_bgcolor("black")
    ax.set_xlim([-0.02 * npix, 1.02 * npix])
    ax.set_ylim([y1, y2])
    ax.grid(color="w", linestyle="--", linewidth=0.5)
    plt.ylabel("counts")
    plt.xlabel("pixels")
    # PLOT DATA!!
    plt.plot(galaxy, "white", lw=0.8, label="data")
    plt.plot(bestfit, "red", lw=1.5, label="fit")
    plt.plot(bestfit - emission, "r--", lw=0.8)
    plt.plot(emission + mn, "cyan", lw=0.7, label="emission-lines")
    plt.plot(goodpixels, resid[goodpixels], "g,", ms=0.8, label="residuals")
    n = len(goodpixels)
    plt.plot(goodpixels, ([mn] * n), "green")
    w = np.where((goodpixels - np.roll(goodpixels, 1)) > 1)[0]
    w = (
        [0, n - 1] if len(w) == 0 else np.concatenate([[0], w - 1, w, [n - 1]])
    )  # add first and last point
    for j in range(len(w)):
        plt.plot(
            [goodpixels[w[j]]] * 2, np.append(mn, galaxy[goodpixels[w[j]]]), color="g"
        )
    resid_noise = robust_sigma(resid[goodpixels] - mn, zero=True)
    plt.plot(emission * 0 + resid_noise + mn, "w--", lw=0.8, label="res.-noise")
    # plots the polinomial correction and the unadjusted templates
    x = np.linspace(-1.0, 1.0, num=npix)
    mpoly = 1.0  # the loop below can be null if mdegree < 1
    npars = len(best_pars) - mdegree
    if not reddening:
        for j in range(1, mdegree + 1):
            mpoly += sp_s.legendre(j)(x) * best_pars[npars + j - 1]
        plt.plot(
            (bestfit - emission) / mpoly, "y,", ms=0.3, label="unadjusted continuum"
        )
    else:
        vstar = (
            kinstars[0] + (l0_gal - l0_templ) * C
            if not log10
            else kinstars[0] + (l0_gal - l0_templ) * C * np.log(10.0)
        )
        reddening_attenuation = dust_calzetti(
            l0_gal, lstep_gal, npix, sol[nlines * 4], vstar, log10
        )
        int_reddening_attenuation = (
            1.0
            if len(reddening) != 2
            else dust_calzetti(
                l0_gal, lstep_gal, npix, sol[nlines * 4 + 1], vstar, log10
            )
        )
        plt.plot(
            ((bestfit - emission) / reddening_attenuation),
            "y,",
            ms=0.3,
            label="unadjusted continuum",
        )
    # Legend
    fontP = FontProperties()
    fontP.set_size("x-small")
    plt.legend(prop=fontP)
    plt.draw()
    plt.pause(3)


###############################################################################
def _display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.
    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin) / pixelsize) + 1)
    ny = int(round((ymax - ymin) / pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin) / pixelsize).astype(int)
    k = np.round((y - ymin) / pixelsize).astype(int)
    img[j, k] = counts
    plt.imshow(
        np.rot90(img),
        interpolation="nearest",
        cmap="prism",
        extent=[
            xmin - pixelsize / 2,
            xmax + pixelsize / 2,
            ymin - pixelsize / 2,
            ymax + pixelsize / 2,
        ],
    )
    plt.draw()


#############################################################################
def load_emission_setup_new(data):
    # Load data of emission setup from a flat array and return an
    # array of classes
    out = []
    for i in np.arange(0, np.shape(data)[1]):
        elem = [
            data[0][i],
            data[1][i],
            np.float64(data[2][i]),
            data[3][i],
            data[4][i],
            np.float64(data[5][i]),
            data[6][i],
            data[7][i],
            data[8][i],
            data[9][i],
        ]
        out.append(EmissionSetup(*elem))
    return out


#################################################################
def remouve_detected_emission(
    galaxy, bestfit, emission_templates, sol_gas_A, AoN_thresholds, goodpixel
):
    # ; Given the galaxy spectrum, the best fit, the emission-line
    # ; amplitudes, a vector with the A/N threshold for each line, and the
    # ; array containing the spectra of each best-fitting emission-line
    # ; templates, this function simply compute the residual-noise lvl,
    # ; compares it the amplitude of each lines, and remouves from the
    # ; galaxy spectrum only the best-matching emission-line templates for
    # ; which the correspoding A/N exceed the input threshold.  This is a
    # ; necessary step prior to measurements of the strength of the stellar
    # ; absorption line features
    # ;
    # ; A list of goodpixels may be optionally input, for instance if any
    # ; pixel was excluded by sigma-clipping during the continuum and
    # ; emission-line fitting, or by excluding pixels on either ends of the
    # ; spectra
    # ;
    # ; Also optionally outputs the computed A/N ratios.
    # ; Get the Residual Noise lvl.
    resid = galaxy - bestfit
    if goodpixel != None:
        resid = resid[goodpixels]
    resid_noise = robust_sigma(resid)
    # ; A/N of each line
    AoN = sol_gas_A / resid_noise
    # ; Create neat spectrum, that is, a spectrum where only detected
    # ; emission has been remouved
    neat_galaxy = galaxy
    for i in range(0, len(sol_gas_A)):
        if AoN[i] >= AoN_thresholds[i]:
            neat_galaxy = neat_galaxy - emission_templates[:, i]
    return AoN, neat_galaxy


#################################################################
def return_arg(arg_out, values):
    # Clear an array (arg_out) and append the contents of other (values)
    # Used to copy arguments to return (try to simulate IDL's way to
    # return results using Python parameters)
    # Clean array
    while len(arg_out) > 0:
        # print arg_out[0]
        del arg_out[0]
    # Copy values
    for x in values:
        arg_out.append(x)


#################################################################
def mask_emission_lines(
    npix,
    Vsys,
    emission_setup,
    velscale,
    l0_gal,
    lstep_gal,
    sigm,
    l_rf_range,
    log10,
    sysRedshift,
):
    # Return a list of goodpixels to fit that excludes regions potentially
    # affected by gas emission and by sky lines. Unless the log10 keyword
    # is specified, wavelength values are assumed to be ln-rebinned, and
    # are defined by the l0_gal, lstep_gal, npix parameters. The position of
    # gas and sky emission lines is set by the input emission_setup
    # structure and the width of the mask by the sigma parameter. If a
    # sigma value is not passed than the width of each line is taken from
    # the emission_setup structure.
    #
    # The rest-frame fitting wavelength range can be manually restricted
    # using the l_rf_range keyword to pass min and max observed
    # wavelength. Typically used to exclude regions at either side of
    # spectra.
    # speed of light
    c = np.float64(299792.458)
    # define good pixels array
    goodpixels = np.arange(0, npix)
    # if set, exclude regions at either ends of the spectra using the keyword l_rf_range
    if l_rf_range != None:
        if (np.shape(l_rf_range))[
            0
        ] > 1:  # CHECKS ONLY FOR DIM NOT WHETHER IT EXISTS IS SET
            pix0 = int(
                mt.ceil((np.log(l_rf_range[0]) - l0_gal) / lstep_gal + Vsys / velscale)
            )
            pix1 = int(
                mt.ceil((np.log(l_rf_range[1]) - l0_gal) / lstep_gal + Vsys / velscale)
            )
            if log10 == 1:
                pix0 = int(
                    mt.ceil(
                        (np.log10(l_rf_range[0]) - l0_gal) / lstep_gal + Vsys / velscale
                    )
                )
                pix1 = int(
                    mt.ceil(
                        (np.log10(l_rf_range[1]) - l0_gal) / lstep_gal + Vsys / velscale
                    )
                )
            goodpixels = np.arange(np.max([pix0, 0]), np.min([pix1, npix]))
    tmppixels = goodpixels
    # looping over the listed emission-lines and mask those tagged with an
    # 'm' for mask. Mask sky lines at rest-frame wavelength
    for i in np.arange(0, len(emission_setup)):
        if (emission_setup[i]).action == "m":
            #        print('--> masking ' + (emission_setup[i]).name)
            if (emission_setup[i]).name != "sky":
                meml_cpix = mt.ceil(
                    (np.log((emission_setup[i])._lambda) - l0_gal) / lstep_gal
                    + Vsys / velscale
                )
            if ((emission_setup[i]).name != "sky") & (log10 == 1):
                meml_cpix = mt.ceil(
                    (np.log10((emission_setup[i])._lambda) - l0_gal) / lstep_gal
                    + Vsys / velscale
                )
            # sky lines are at rest-frame
            if (emission_setup[i]).name == "sky":
                meml_cpix = mt.ceil(
                    (np.log((emission_setup[i])._lambda / (1 + sysRedshift)) - l0_gal)
                    / lstep_gal
                )
            if ((emission_setup[i]).name == "sky") & (log10 == 1):
                meml_cpix = mt.ceil(
                    (np.log10((emission_setup[i])._lambda / (1 + sysRedshift)) - l0_gal)
                    / lstep_gal
                )
            # set the width of the mask in pixels using either
            # 3 times the sigma of each line in the emission-line setup
            # or the provided sigma value
            if sigm != None:
                msigma = 3 * sigm / velscale
            if sigm == None:
                msigma = 3 * ((emission_setup[i]).s) / velscale
            meml_bpix = meml_cpix - msigma
            meml_rpix = meml_cpix + msigma
            w = np.where((goodpixels >= meml_bpix) & (goodpixels <= meml_rpix))
            if np.size(w) != 0:
                tmppixels[w] = -1
            elif np.size(w) == 0:
                #          print('this line is outside your wavelength range. We shall ignore it')
                (emission_setup[i]).action = "i"
    w = np.where(tmppixels != -1)
    goodpixels = goodpixels[w]
    return goodpixels, emission_setup


#################################################################
def set_constraints(
    galaxy,
    noise,
    cstar,
    kinstars,
    velscale,
    degree,
    mdegree,
    goodpixels,
    emission_setup,
    start_pars,
    l0_gal,
    lstep_gal,
    int_disp,
    log10,
    reddening,
    l0_templ,
):
    # This subroutine sets up the constraints and boundaries for the
    # variables to be fitted, preparing and returning the PARINFO and
    # FUNCTARG structure for MPFIT
    # Total number of parameters that MPFIT will deal with, i.e. the
    # number of emission lines (counting multiplets only once) times two
    # (or three if we are fitting also the amplitudes to get the errors on
    # them), plus the order of the multiplicative polynomials and if
    # needed the reddening parameters
    #
    parinfo = []
    i_lines = where_eq(emission_setup, "kind", "l")
    nlines = len(i_lines)
    n_pars = nlines * 2 + mdegree

    if reddening != None:
        n_pars = n_pars + len(reddening)
    # Setup the PARINFO structure that will allow us to control the limits
    # of our parameters with MPFIT, decide whether we want to hold them at
    # the input values, or tie some of them to others.
    #  values_def = {'step':velscale/150., 'limits':[0.,0.], 'limited':[1,1], 'fixed':0, 'tied':' '}
    values_def = {
        "step": 0,
        "limits": [0.0, 0.0],
        "limited": [1, 1],
        "fixed": 0,
        "tied": " ",
    }
    for i in np.arange(n_pars):
        parinfo.append(copy.copy(values_def))
    # A) First of all, fix V_gas and S_gas to their input values for
    # the lines with a 'h' (for hold) as their fit-kind tag
    for i in np.arange(nlines):
        j = i_lines[i]
        if emission_setup[j].fit == "h":
            inndx = range(2 * i, 2 * i + 1, 1)
            parinfo[inndx]["fixed"] = 1
    # B) Second, set the limits
    # i) for V_gas and S_gas
    vlimit = 6e2  # custom for MAGPI
    slimit = 3e2  # custom for MAGPI

    for i in range(0, nlines * 2, 2):
        parinfo[i]["limits"] = [
            start_pars[i] - (vlimit / velscale),
            start_pars[i] + (vlimit / velscale),
        ]  # Limits for Vgas (from km/s to pixels)
        parinfo[i + 1]["limits"] = [
            1.0 / velscale,
            slimit / velscale,
        ]  # Limits for Sgas (from km/s to pixels)
        # to avoid problems near the previous boundaries
        if start_pars[i] <= parinfo[i]["limits"][0]:
            start_pars[i] = parinfo[i]["limits"][0] + 0.0001
        if start_pars[i] >= parinfo[i]["limits"][1]:
            start_pars[i] = parinfo[i]["limits"][1] - 0.0001
        if start_pars[i + 1] <= parinfo[i + 1]["limits"][0]:
            start_pars[i + 1] = parinfo[i + 1]["limits"][0] + 0.0001
        if start_pars[i + 1] >= parinfo[i + 1]["limits"][1]:
            start_pars[i + 1] = parinfo[i + 1]["limits"][1] - 0.0001

    # ii) for the mult. polynomial (if needed)
    if mdegree >= 1:
        for iindx in range(n_pars - mdegree, len(parinfo)):
            parinfo[iindx]["limits"] = [-1.0, 1.0]
            parinfo[iindx]["step"] = 1e-3
    # iii) and for the reddening parameters (if needed). These will follow the
    # emission-line parameters in the parameter array.
    if reddening is not None:
        for iindx in range(nlines * 2, nlines * 2 + len(reddening), 1):
            parinfo[iindx]["limits"] = [np.float64(0.0), np.float64(5.0)]
            parinfo[iindx]["step"] = 1e-3
    # C) Finally, find the lines for which the kinematics needs to be tied
    # to that of other lines. These have either 't#', 'v#' or 's#'
    # fit-kind tags, where # indicates the index of the line to which they
    # are tied. Notice that the # indexes are the ones listed in the .i
    # tag of the emission-lines setup structure
    #
    # If 't#' we tie both the position and width of the lines, if 'v#'
    # only the position, and if 's#' only the width.
    for i in np.arange(0, nlines):
        j = i_lines[i]
        fit_tag = emission_setup[j].fit[0]
        # check if we have a 't' tag
        if fit_tag == "t":
            # find the reference line reference index, as given in the
            # 'fit' tag of the emission setup structure
            k_refline = int(emission_setup[j].fit[1:])
            # which correspond to the following position in emission setup
            # structure that was passed to this function
            j_refline = where_eq(emission_setup, "i", k_refline)
            if len(j_refline) != 1:
                err_msg_exit(
                    "Hey, you tied "
                    + emission_setup[j].name
                    + " to a line you are not even fitting..."
                )
            j_refline = j_refline[0]
            # and to the following position in the list of the emission
            # lines to fit
            i_refline = where_eq([emission_setup[k] for k in i_lines], "i", k_refline)
            if len(i_refline) != 1:
                err_msg_exit(
                    "Hey, you tied "
                    + emission_setup[j].name
                    + " to a line that is not in the current subset (kind=l)..."
                )
            i_refline = i_refline[0]
            l_line = emission_setup[j]._lambda
            str_l_line = str(l_line)
            l_refline = emission_setup[j_refline]._lambda
            str_l_refline = str(l_refline)

            parinfo[2 * i]["tied"] = (
                "p["
                + str(2 * i_refline)
                + "]-numpy.log("
                + str_l_refline
                + "/"
                + str_l_line
                + ")/"
            ) + (str(lstep_gal).strip()).replace(" ", "")
            parinfo[2 * i + 1]["tied"] = ("p[" + str(2 * i_refline + 1) + "]").replace(
                " ", ""
            )
            # to deal with log10-lambda rebinned data, instead of ln-lambda
            if log10 == 1:
                parinfo[2 * i]["tied"] = (
                    "p["
                    + str(2 * i_refline)
                    + "]-numpy.log10("
                    + str_l_refline
                    + "/"
                    + str_l_line
                    + ")/"
                ) + (str(lstep_gal).strip()).replace(" ", "")

        # check if we have a 'v' tag
        if fit_tag == "v":
            # find the reference line reference index, as given in the
            # 'fit' tag of the emission setup structure
            k_refline = int(emission_setup[j].fit[1:])
            # which correspond to the following position in emission setup
            # structure that was passed to this function
            j_refline = where_eq(emission_setup, "i", k_refline)
            if len(j_refline) != 1:
                err_msg_exit(
                    "Hey, you tied "
                    + emission_setup[j].name
                    + " to a line you are not even fitting..."
                )
            j_refline = j_refline[0]
            # and to the following position in the list of the emission
            # lines to fit
            i_refline = where_eq([emission_setup[k] for k in i_lines], "i", k_refline)
            i_refline = i_refline[0]
            l_line = emission_setup[j]._lambda
            str_l_line = str(l_line)
            l_refline = emission_setup[j_refline]._lambda
            str_l_refline = str(l_refline)

            parinfo[2 * i]["tied"] = (
                "p["
                + str(2 * i_refline)
                + "]-numpy.log("
                + str_l_refline
                + "/"
                + str_l_line
                + ")/"
            ) + (str(lstep_gal).strip()).replace(" ", "")
            # to deal with log10-lambda rebinned data, instead of ln-lambda
            if log10 == 1:
                parinfo[2 * i]["tied"] = (
                    "p["
                    + str(2 * i_refline)
                    + "]-numpy.log10("
                    + str_l_refline
                    + "/"
                    + str_l_line
                    + ")/"
                ) + (str(lstep_gal).strip()).replace(" ", "")

        # check if we have a 's' tag
        if fit_tag == "s":
            # find the reference line reference index, as given in the
            # 'fit' tag of the emission setup structure
            k_refline = int((emission_setup[j].fit[1:]))
            # which correspond to the following position in emission setup
            # structure that was passed to this function
            j_refline = where_eq(emission_setup, "i", k_refline)
            if len(j_refline) != 1:
                err_msg_exit(
                    "Hey, you tied "
                    + emission_setup[j].name
                    + " to a line you are not even fitting..."
                )
            j_refline = j_refline[0]
            # and to the following position in the list of the emission
            # lines to fit
            i_refline = where_eq([emission_setup[k] for k in i_lines], "i", k_refline)
            i_refline = i_refline[0]
            l_line = emission_setup[j]._lambda
            str_l_line = str(l_line)
            l_refline = emission_setup[j_refline]._lambda
            str_l_refline = str(l_refline)

            parinfo[2 * i + 1]["tied"] = (
                "p[" + string(2 * i_refline + 1) + "]"
            ).replace(" ", "")

    functargs = {
        "cstar": cstar,
        "galaxy": galaxy,
        "noise": noise,
        "emission_setup": emission_setup,
        "kinstars": kinstars,
        "velscale": velscale,
        "degree": degree,
        "mdegree": mdegree,
        "goodpixels": goodpixels,
        "l0_gal": l0_gal,
        "lstep_gal": lstep_gal,
        "int_disp": int_disp,
        "log10": log10,
        "reddening": reddening,
        "l0_templ": l0_templ,
    }
    return parinfo, functargs


################################################################################
def shifta(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


################################################################################
def rebin(x, factor):
    """
    Rebin a vector, or the first dimension of an array,
    by averaging within groups of "factor" adjacent values.
    """
    if factor == 1:
        xx = x
    else:
        xx = x.reshape(len(x) // factor, factor, -1).mean(1).squeeze()
    return xx


###############################################################################
def create_templates(emission_setup, pars, npix, lstep_gal, int_disp_pix, log10):
    # Take the emission-setup structure and the input pars parameter array
    # to make emission-line single or multi-Gaussian templates.
    #
    # The input pars array should contiant only the Gaussian paramenter for the lines
    # that we are actually fitting, such as Hb or [OIII]5007, containing
    # only the V_gas and S_gas parameters, like [V_Hb, S_Hb, V_OIII5007, S_OIII5007, ...]
    #
    # On the other hand if we wish to fit also the amplitudes with MPFIT,
    # then the pars array should be [A_Hb, V_Hb, S_Hb, A_OIII5007, ...]
    # This happens when we are evaluating the errors in the Gaussian
    # parameters around the best solution.
    i_lines = where_eq(emission_setup, "kind", "l")
    nlines = len(i_lines)
    n_pars = 2 * nlines
    if n_pars != len(pars):
        print("Hey, this is not the right emission-line parameter array")
    # array that will contain the emission-line templates, including multiplets
    gaus = []
    # a) First create the emission-line templates corresponding to each
    # single line (like Hb) or to each main line of a multiplet (like
    # [OIII]5007)
    for i in range(nlines):
        # Create the emission-line templates.
        # Use the amplitude from the emission-line setup. If that is
        # set to unity, than the NNLS weight assigned to each template
        # will actually correspond to the emission-line
        # amplitude. Negative input values will produce absorption
        # lines.
        ampl_i = np.float64(emission_setup[i_lines[i]].a)
        gaus.append(
            create_gaussn(
                np.arange(npix, dtype=np.float64),
                np.append(ampl_i, pars[2 * i : 2 * i + 2]),
                int_disp_pix[i_lines[i]],
            )
        )
    #  gaus = np.copy(gaus)

    # b) Then find all the satellite lines belonging to multiplets
    # (like [OIII]4959), and add them to the main emission-line template
    # we just created in a)
    i_slines = []
    for i, em in enumerate(emission_setup):
        if em.kind[0] == "d":
            i_slines.append(i)
    n_slines = len(i_slines)
    if n_slines > 0:
        # loop over the satellite lines
        for i in range(n_slines):
            j = i_slines[i]
            # Current index in the emission-line setup structure for
            # the current satellite line (e.g. 1 for [OIII]4959)
            # Find the reference line index, as given in the "kind" tag of the
            # emission setup structure, which points to the main line in the
            # present multiplet (e.g. 2 for [OIII]5007 if kind was d2 for [OIII]4959)
            k_mline = int(emission_setup[j].kind[1:])
            # which correspond to the following position in the emission setup
            # structure that was passed to this function (e.g. still 2, if
            # the indices of the lines in the emission setup start at 0 and
            # increase by 1)
            j_mline = where_eq(emission_setup, "i", k_mline)[0]
            # Get the wavelengths of both satellite and main lines
            # and compute the offset (in pix) that we need to apply in order
            # to correctly place the satellite line.
            l_sline = emission_setup[j]._lambda
            l_mline = emission_setup[j_mline]._lambda
            # to deal with log10-lambda rebinned data, instead of ln-lambda
            if log10:
                offset = mt.log10(l_mline / l_sline) / lstep_gal
            else:
                offset = mt.log(l_mline / l_sline) / lstep_gal
            # Get the index in the array of the lines to fit, corresponding
            # to the main line of the present multiplet, so that we can add the
            # satellite emission-line template in the right place in the
            # gaussian templates array
            i_mline = where_eq([emission_setup[x] for x in i_lines], "i", k_mline)[0]
            # Finally, create the satellite template, and add it to that of
            # the corresponding main line of this multiplet.

            # Use the amplitudes given in the emission setup structure
            # and the wavelength the lines to compute the amplitude
            # that the satellite lines must have to obtain to the
            # desired relative strength, in terms of total flux, of
            # the satellite line w.r.t to that of their main line
            # (e.g.  F_Hb = 0.35 F_Ha without reddening).
            f_sat_to_main_ratio = emission_setup[j].a * emission_setup[j_mline].a
            lambda_main_to_sat_ratio = (
                emission_setup[j_mline]._lambda / emission_setup[j]._lambda
            )
            obs_sigma_pix_main_to_sat_ratio = np.sqrt(
                pars[i_mline * 2 + 1] ** 2 + int_disp_pix[j_mline] ** 2
            ) / np.sqrt(pars[i_mline * 2 + 1] ** 2 + int_disp_pix[j] ** 2)
            a_sline = (
                f_sat_to_main_ratio
                * lambda_main_to_sat_ratio
                * obs_sigma_pix_main_to_sat_ratio
            )

            gaus_sline = create_gaussn(
                np.arange(npix, dtype="float64"),
                [a_sline, pars[i_mline * 2] - offset, pars[i_mline * 2 + 1]],
                int_disp_pix[j],
            )
            gaus[i_mline] += gaus_sline
    return gaus


###############################################################################
def BVLSN_Solve_pxf(AA, bb, degree, nlines):
    # No need to enforce positivity constraints if fitting one
    # single template: use faster SVD solution instead of BVLS.
    #
    AA = np.array(AA, dtype=np.float64)
    bb = np.array(bb, dtype=np.float64)

    soluz = _bvls_solve(AA, bb, 0)
    return soluz


###############################################################################
def create_gaussn(x, xpars, int_disp_pix_line):
    # def create_gaussn(x, xpars, int_disp_pix2): !OLD version
    #  The instrumental resolution is supoly_iosed to be in sigma and in pixels
    # at this stage
    pars = xpars  # np.copy(xpars)
    npars = len(xpars)
    npix = len(x)
    y = np.zeros(npix, dtype=np.float64)
    for i in np.arange(0, npars - 2, 3):
        pars[i + 2] = mt.sqrt(pars[i + 2] ** 2 + (int_disp_pix_line) ** 2)
        w = (np.arange(0, npix, dtype=np.float64) - pars[i + 1]) / pars[i + 2]
        y += pars[i] * np.exp(-(w**2) / 2.0)
    return y


###############################################################################
def _losvd_rfft(pars, nspec, moments, nl):
    """
    Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
    Equation (38) of Cappellari M., 2017, MNRAS, 466, 798
    http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
    """
    vsyst = 0  # Def in ppxf
    factor = 1.0
    sigma_diff = 0.0
    ncomp = 1  # One component only
    nspec = 1
    losvd_rfft = np.empty((nl, ncomp, nspec), dtype=complex)
    p = 0
    for j, mom in enumerate(moments):  # loop over kinematic components
        for k in range(nspec):  # nspec=2 for two-sided fitting, otherwise nspec=1
            s = 1 if k == 0 else -1  # s=+1 for left spectrum, s=-1 for right one
            vel, sig = vsyst + s * pars[0 + p], pars[1 + p]
            a, b = [vel, sigma_diff] / sig
            w = np.linspace(0, np.pi * factor * sig, nl)
            losvd_rfft[:, j, k] = np.exp(1j * a * w - 0.5 * (1 + b**2) * w**2)
            if mom > 2:
                n = np.arange(3, mom + 1)
                nrm = np.sqrt(sp_s.factorial(n) * 2**n)  # vdMF93 Normalization
                coeff = np.append([1, 0, 0], (s * 1j) ** n * pars[p - 1 + n] / nrm)
                poly = hermite.hermval(w, coeff)
                losvd_rfft[:, j, k] *= poly
        p += mom
    return np.conj(losvd_rfft)


###############################################################################
def _losvd_rfft_old(pars, nspec, moments, nl, ncomp, vsyst, factor, sigma_diff):
    """
    Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
    Equation (38) of Cappellari M., 2017, MNRAS, 466, 798
    http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
    """
    losvd_rfft = np.empty((nl, ncomp, nspec), dtype=complex)
    p = 0
    for j, mom in enumerate(moments):  # loop over kinematic components
        for k in range(nspec):  # nspec=2 for two-sided fitting, otherwise nspec=1
            s = 1 if k == 0 else -1  # s=+1 for left spectrum, s=-1 for right one
            vel, sig = vsyst + s * pars[0 + p], pars[1 + p]
            a, b = [vel, sigma_diff] / sig
            w = np.linspace(0, np.pi * factor * sig, nl)
            losvd_rfft[:, j, k] = np.exp(1j * a * w - 0.5 * (1 + b**2) * w**2)
            #
            if mom > 2:
                n = np.arange(3, mom + 1)
                nrm = np.sqrt(sp_s.factorial(n) * 2**n)  # vdMF93 Normalization
                coeff = np.append([1, 0, 0], (s * 1j) ** n * pars[p - 1 + n] / nrm)
                poly = hermite.hermval(w, coeff)
                losvd_rfft[:, j, k] *= poly
        p += mom
    #
    return np.conj(losvd_rfft)


###############################################################################
def convolve_templates(templates, kinstars, velscale, npix_gal):  # , velscale_ratio)
    # <<<< NO CONVOLUTION ANALOGOUS TO IDL SO DO AS IN PPXF-ish IN FOURIER SPACE >>>>
    #  if (velscale_ratio != 1): # Check for oversampling ... .
    #    vel   = kinstars[0]/(velscale*velscale_ratio)            # in pixels
    #    sigma = kinstars[1]/(velscale*velscale_ratio)                # in pixels
    #    dx = int(mt.ceil(abs(vel) + 4*sigma))       # Sample the Gaussian and GH at least to vel+4*sigma
    #    x  = np.arange(dx, -dx-1, -1)               # Evaluate the Gaussian using steps of 1 pixel.
    #  else:
    #    vel   = kinstars[0]/velscale                # in pixels
    #    sigma = kinstars[1]/velscale                # in pixels
    #    dx = int(mt.ceil(abs(vel) + 4*sigma))       # Sample the Gaussian and GH at least to vel+4*sigma
    #    x  = np.arange(dx, -dx-1, -1)               # Evaluate the Gaussian using steps of 1 pixel.
    #
    vel = kinstars[0] / velscale  # in pixels
    sigma = kinstars[1] / velscale  # in pixels
    dx = int(
        mt.ceil(abs(vel) + 4 * sigma)
    )  # Sample the Gaussian and GH at least to vel+4*sigma
    x = np.arange(dx, -dx - 1, -1)  # Evaluate the Gaussian using steps of 1 pixel.
    w = (x - vel) / sigma
    w2 = w**2
    losvd = np.exp(-0.5 * w2) / (
        mt.sqrt(2 * mt.pi) * sigma
    )  # Normalized total(Gaussian)=1
    poly = np.zeros(len(w))
    # Hermite polynomials as in van der Marel & Franx (1993).
    # Coefficients are given e.g. in Apoly_iendix C of Capoly_iellari et al. (2002)
    nkins = len(kinstars)
    if nkins > 2:
        # Pre-calculate constants
        sqrt3 = mt.sqrt(3)
        sqrt24 = mt.sqrt(24)
        sqrt60 = mt.sqrt(60)
        sqrt720 = mt.sqrt(720)
        poly = (
            1
            + kinstars[2] / sqrt3 * (w * (2 * w2 - 3))  # H3
            + kinstars[3] / sqrt24 * (w2 * (4 * w2 - 12) + 3)
        )  # H4
        if nkins == 6:
            poly += kinstars[4] / sqrt60 * (
                w * (w2 * (4 * w2 - 20) + 15)
            ) + kinstars[  # H5
                5
            ] / sqrt720 * (
                w2 * (w2 * (8 * w2 - 60) + 90) - 15
            )  # H6
        losvd *= poly
    s = np.shape(templates)
    ctemplates = np.zeros_like(templates, dtype=np.float64)
    ctemplates = np.transpose(ctemplates)
    # MEET HALFWAY LOSVD NOT FOURIER TRANSFORMED BEFORE AND TEMPLATES HANDLED WITH PADDING
    # TAKEN FROM IDL VERSION OF PPXF-ish ...
    #
    #                        f        k
    #  ppxf_convol_fft(star[*,j],losvd[*,component[j],k])
    nf = np.shape(templates)[1]
    nk = len(losvd)
    nn = long(2 ** (mt.ceil(np.log(nf + nk / 2) / np.log(2))))
    for j in np.arange(np.shape(templates)[0]):  # Loop over templates
        f1 = np.zeros(nn, dtype=np.float64)
        k1 = np.zeros(nn, dtype=np.float64)
        f1[0:nf] = templates[j, :]
        k1[0:nk] = np.flip(losvd, 0)
        k1 = np.roll(k1, int(-(nk - 1) / 2))
        A = np.fft.fft(f1)
        B = np.fft.fft(k1)
        con = np.real((np.fft.ifft(A * B)))[0:nf]
        ctemplates[:, j] = con
    return np.array(ctemplates)


###############################################################################
def convolve_templates_new(
    templates, kinstars, velscale, npix_gal, velscale_ratio, vsyst
):
    #
    # Pre-compute FFT of real input of all templates
    #
    npix_temp = len(templates[0, :])
    templates = templates.T
    npad = 2 ** int(np.ceil(np.log2(templates.shape[0])))
    templates_rfft = np.fft.rfft(templates, npad, axis=0)
    #
    pars = kinstars
    #
    pars[0:2] = pars[0:2] / velscale
    #
    nspec = 1  # not 2 sided by default
    moments = [len(kinstars)]  # fix to 1 comp for now
    nl = templates_rfft.shape[0]
    ncomp = 1  # Add 2 comp !
    factor = velscale_ratio
    sigma_diff = 0.0  # already broadened & convolved !
    #
    losvd_rfft = _losvd_rfft_old(
        pars, nspec, moments, nl, ncomp, vsyst / velscale, factor, sigma_diff
    )
    #
    pp = []
    tmp = np.empty((1, npix_temp))
    #
    for k in np.arange((np.shape(templates)[1])):
        template_rfft = templates_rfft[:, k]
        pr = template_rfft * losvd_rfft[:, 0, 0]
        tt = np.fft.irfft(pr, npad)
        pp.append(rebin(tt[: npix_temp * factor], factor)[:npix_gal].ravel())
    #
    return np.transpose(np.array(pp))


###############################################################################
def dust_calzetti(l0_gal, lstep_gal, npix, ebv, vstar, log10):
    # This procedure uses the dust model of Calzetti et al. (2000, ApJ,
    # 533, 682), and for a given E(B-V) value returns the flux attenuation
    # array, which can be used to get reddened templates. Here the spectra
    # are assumed to be binned on a ln-rebinned wavelentgh grid as defined
    # by input l0_gal,lstep_gal,npix parameters. The input receiding
    # velocity vstar, is used to derive the dust reddening in the galaxy
    # rest-frame.
    #
    # Can be used also to de-reddened the object spectra by the Milky-Way
    # dust extinction, using as E(B-V) the opposite of the Schlegel et
    # al. values found in NED and vstar = 0.
    #
    # Initial version kindly provided by S. Kaviray, Oxford, 2006.
    # reconstruct the wavelength array in Anstroms, and compute rest-frame
    # values
    vect = np.arange(npix, dtype="float64")
    _lambda = np.exp(vect * lstep_gal + l0_gal)
    if log10 == 1:
        _lambda = 10 ** (vect * lstep_gal + l0_gal)
    _lambda /= np.exp(vstar / np.float64(C))
    # array to hold k(lambda) values
    k = np.zeros(len(_lambda), dtype=np.float64)
    for i in range(len(_lambda)):
        # convert wavelength units from angstroms to micrometres
        l = _lambda[i] / np.float64(1e4)
        # assign k values
        if (l >= 0.63) and (l <= 2.2):
            k[i] = 2.659 * (-1.857 + 1.040 / l) + 4.05
        elif l < 0.63:
            k[i] = 2.659 * (-2.156 + 1.509 / l - 0.198 / l**2 + 0.011 / l**3) + 4.05
        if l > 2.2:
            k[i] = 0.0  # l > 2.2
    # this should be then multiplied by the spectrum flux array
    return np.array(10 ** (-0.4 * ebv * k), dtype=np.float64)


###############################################################################
def fitfunc_gas(pars, **kwargs):
    cstar = kwargs["cstar"]
    galaxy = kwargs["galaxy"]
    noise = kwargs["noise"]
    kinstars = kwargs["kinstars"]
    velscale = kwargs["velscale"]
    degree = kwargs["degree"]
    mdegree = kwargs["mdegree"]
    goodpixels = kwargs["goodpixels"]
    l0_gal = kwargs["l0_gal"]
    lstep_gal = kwargs["lstep_gal"]
    int_disp = kwargs["int_disp"]
    log10 = kwargs["log10"]
    reddening = kwargs["reddening"]
    l0_templ = kwargs["l0_templ"]
    emission_setup = kwargs["emission_setup"]
    npix = len(galaxy)
    x = np.linspace(
        -1.0, 1.0, num=npix
    )  # X needs to be within [-1,1] for Legendre Polynomials
    nlines = len(where_eq(emission_setup, "kind", "l"))

    npars = nlines * 2
    # append the reddening parameters if needed
    if reddening:
        npars = npars + len(reddening)
    # The zero order multiplicative term is already included in the
    # linear fit of the templates. The polinomial below has mean of 1.
    mpoly = 1.0  # The loop below can be null if mdegree < 1
    for j in range(1, mdegree + 1):
        mpoly += sp_s.legendre(j)(x) * pars[npars + j - 1]
    # Emission Lines as given by the values in pars
    # passing only the emission-line parameters
    s = cstar.ndim
    # passing only the emission-line parameters
    eml_pars = pars[0 : nlines * 2]  # np.copy(pars[0:nlines*2])
    int_disp_pix = int_disp / velscale
    gaus = create_templates(
        emission_setup, eml_pars, npix, lstep_gal, int_disp_pix, log10
    )
    # Stacking all the inputs together:
    #   1.- Legendre polinomials of order 'degree'
    #   2.- Convolved SSP models (pre-convolved by the best LOSVD in set_constraints)
    #       adjusted by a multiplicative polinomials - or by reddening
    #   3.- Emission Lines, also reddened
    if cstar.ndim == 2:  # Number of template spectra
        ntemp = cstar.shape[1]
    else:
        ntemp = 1
    ccc = np.zeros(
        shape=(npix, (degree + nlines + ntemp + 1))
    )  # This array is used for estimating predictions
    aaa = ccc  # This array is used for the actual solution of the system)
    for j in range(degree + 1):
        ccc[:, j] = sp_s.legendre(j)(x)
    if reddening == None:
        # for j in range(ntemp-1):
        for j in range(ntemp):
            ccc[:, degree + 1 + j] = (
                mpoly * cstar[0:npix, j]
            )  # Convolved templates x mult. polinomials
        for j in range(nlines):
            ccc[:, degree + ntemp + 1 + j] = gaus[
                j
            ]  # np.copy(gaus[j])           # Emission lines
    else:
        # redden both stellar and emission-line templates
        ebv = pars[nlines * 2]
        Vstar = kinstars[0] + (l0_gal - l0_templ) * np.float64(299792.458)
        if log10 == 1:
            Vstar = kinstars[0] + (l0_gal - l0_templ) * np.float64(299792.458) * np.log(
                np.float64(10)
            )
        reddening_attenuation = dust_calzetti(
            l0_gal, lstep_gal, npix, ebv, Vstar, log10
        )
        # but also include extra internal reddening if requested
        if len(reddening) == 2:
            int_ebv = pars[nlines * 2 + 1]
            int_reddening_attenuation = dust_calzetti(
                l0_gal, lstep_gal, npix, int_ebv, Vstar, log10
            )
        else:
            int_reddening_attenuation = np.float64(1.0)
        for j in range(ntemp):
            ccc[:, degree + 1 + j] = cstar[0:npix, j] * reddening_attenuation
        for j in range(nlines):
            ccc[:, degree + ntemp + 1 + j] = (
                gaus[j] * reddening_attenuation * int_reddening_attenuation
            )
    KK = np.copy(ccc)
    for j in range(degree + ntemp + nlines + 1):
        aaa[:, j] = ccc[:, j] / noise  # Weight all columns with errors
    solll = BVLSN_Solve_pxf(
        aaa[goodpixels, :], galaxy[goodpixels] / noise[goodpixels], degree, nlines
    )
    bestfit = np.matmul(KK, solll)  # IDL: c # sol
    err = (galaxy[goodpixels] - bestfit[goodpixels]) / noise[goodpixels]
    # output weights for the templates
    weights = np.array(solll[degree + 1 : len(solll)])
    # Make the array containing each of the best matching emission-line templates
    #
    # Array with the Gaussian templates weigths.
    # In case we have used the keyword FOR_ERRORS, these should all be 1
    # and gaus should already have the right amplitudes
    sol_gas = np.array(solll[degree + ntemp + 1 : degree + ntemp + nlines + 1])
    emission_templates = np.copy(gaus)
    if not reddening:
        for i in range(len(sol_gas)):
            emission_templates[i] = gaus[i] * sol_gas[i]
    else:
        # Provide the emission-line templates as observed, i.e. reddened
        for i in range(len(sol_gas)):
            emission_templates[i] = (
                gaus[i] * sol_gas[i] * reddening_attenuation * int_reddening_attenuation
            )
    # Return results using arguments...
    if "weights" in kwargs:
        return_arg(kwargs["weights"], weights)
    if "bestfit" in kwargs:
        return_arg(kwargs["bestfit"], bestfit)
    if "emission_templates" in kwargs:
        return_arg(kwargs["emission_templates"], emission_templates)
    return [0, err]


###############################################################################
def rearrange_results(
    res,
    weights,
    l0_gal,
    lstep_gal,
    velscale,
    emission_setup,
    int_disp,
    log10,
    reddening,
    err,
):
    # Given the input res array from the MPFIT fit (with V_gas and S_gas
    # best-fitting values - in pixels) and the weight from the BVLS fit
    # (which with the emission-line basic amplitudes to get A_gas),
    # construct the sol solution array, containing F_gas, A_gas, V_gas and
    # S_gas (the latter in km/s). Also rearrange the MPFIT formal
    # uncertainties in V_gas and S_gas, which should be consider only as
    # lower estimates.
    # If this routine is called after a second MPFIT fit for A_gas, V_gas
    # and S_gas then we also rearrange the corresponding MPFIT formal
    # uncertainties, which in this case will be the correct ones.
    # NOTE: at the moment this routine does not rearrange mult. polynomial
    # coefficient and corresponding uncertainties. This could be easily
    # implemented by adding a MDEGREE=mdegree keyword and then append to
    # the sol_final and esol_final array the corresponding numbers from
    # the res and err input arrays
    i_lines = where_eq(emission_setup, "kind", "l")
    nlines = len(i_lines)
    len_red = 0 if not reddening else len(reddening)
    lambda0 = [emission_setup[x]._lambda for x in i_lines]
    if log10:
        offset = (np.log10(lambda0) - l0_gal) / lstep_gal
    else:
        offset = (np.log(lambda0) - l0_gal) / lstep_gal
    N_FLUX = 100
    # make final output solution array
    sol_final = np.zeros(nlines * 4 + len_red)
    k = 0
    h = 0
    for i in range(nlines):
        # processing outputs from quickest fits, where only V_gas and S_gas
        # were solved with MPFIT whereas A_gas was left to BVLS
        ampl_i = emission_setup[i_lines[i]].a
        sol_final[k + 1] = (
            ampl_i * weights[len(weights) - nlines + i]
        )  # Final amplitude of the Emission line
        sol_final[k + 2] = (
            -(offset[i] - res[h]) * velscale
        )  # Radial velocity of the Emission line [km/s]
        sol_final[k + 3] = abs(
            res[h + 1] * velscale
        )  # Sigma (intrinsic!) of the Emission line [km/s]
        # Sigma (as observed!)
        sigma = mt.sqrt(sol_final[k + 3] ** 2.0 + int_disp[i_lines[i]] ** 2.0)
        # Flux of the Emission lines
        sol_final[k] = (
            sol_final[k + 1]
            * mt.sqrt(2 * mt.pi)
            * sigma
            * lambda0[i]
            * mt.exp(sol_final[k + 2] / C)
            / C
        )
        k += 4
        h = h + 2
    # Append reddening values to the final solution vector, after the
    # emission-line parameters
    if len_red > 0:  # EDITED BY ADRIAN, 06. MAY 2019
        #  if len(reddening) > 0:  # ORIGINAL CODE
        sol_final[nlines * 4 : nlines * 4 + len_red] = res[
            nlines * 2 : nlines * 2 + len_red
        ]
    if err is not None:
        esol_final = np.zeros(nlines * 4)
        # make room for errors in the reddening paramenter(s)
        if len_red > 0:  # EDITED BY ADRIAN, 06. MAY 2019
            #    if len(reddening) > 0:  # ORIGINAL CODE
            esol_final = np.zeros(nlines * 4 + len_red)
        k = 0
        h = 0
        # MPFIT errors only from V_gas and S_gas fit
        for i in range(nlines):
            esol_final[k] = 0.0
            esol_final[k + 1] = 0.0
            esol_final[k + 2] = (
                err[h] * velscale
            )  # these are almost certain lower limits for
            esol_final[k + 3] = (
                err[h + 1] * velscale
            )  # the real uncertainties on these parameters
            k += 4
            h += 2
        # Add reddening errors
        if len_red > 0:  # EDITED BY ADRIAN, 06. MAY 2019
            #   if len(reddening) > 0:  # ORIGINAL CODE
            esol_final[nlines * 4 : nlines * 4 + len_red] = err[
                nlines * 2 : nlines * 2 + len_red
            ]
    sol = np.copy(sol_final)
    if err is not None:
        esol = np.copy(esol_final)
    return sol, esol


###############################################################################
def gandalf(
    templates,
    galaxy,
    noise,
    velscale,
    sol,
    emission_setup,
    l0_gal,
    lstep_gal,
    goodpixels,
    degree,
    mdegree,
    int_disp,
    plot,
    quiet,
    log10,
    reddening,
    l0_templ,
    for_errors,
    velscale_ratio,
    vsyst,
):  #:, lsf_matrix):
    templates = np.transpose(templates)
    len_red = 0 if not reddening else len(reddening)
    # ------------------------------------
    # Do some initial input error checking
    if templates.ndim > 2 or galaxy.ndim != 1 or noise.ndim != 1:
        err_msg_exit("Wrong input dimensions")
    if len(galaxy) != len(noise):
        err_msg_exit("GALAXY and NOISE must have the same size")
    if templates.shape[1] < len(galaxy):
        err_msg_exit("STAR length cannot be smaller than GALAXY")
    if degree <= 0:
        degree = -1
    elif degree > 0:
        degree = degree
    if mdegree <= 0:
        mdegree = 0
    elif mdegree > 0:
        mdegree = mdegree
    goodpixels = range(len(galaxy)) if goodpixels is None else goodpixels
    if max(goodpixels) > (len(galaxy) - 1):
        err_msg_exit("GOODPIXELS are outside the data range")
    int_disp = 0.0 if int_disp is None else int_disp
    # do not allow use simultaneous use of reddening and polynomials.
    if reddening is not None:
        if isinstance(reddening, float) or isinstance(reddening, int):
            reddening = np.array([reddening])
        elif len(reddening) > 0 and (degree != -1 or mdegree != 0):
            err_msg_exit("Reddening & polynomial adjust. cannot be used together")
        elif len(reddening) > 2:
            err_msg_exit("Sorry, can only deal with two dust components...")
        elif len(reddening) == 0:
            reddening = None
    #
    # @@@@@@@@@@@@@@@@@@@@@@@  Check if lstep can be taken from templates @@@@@@@@@@@@@@@@@@@@@ !New
    #
    #  npix_temp = len(templates[0,:]
    #  if velscale_ratio != 1:
    #    assert isinstance(velscale_ratio, int), \
    #      "VELSCALE_RATIO must be an integer"
    #    npix_temp -= npix_temp % velscale_ratio
    #      # Make size multiple of velscale_ratio
    #    templates = templates[:,npix_temp]
    #      # This is the size after rebin()
    #    npix_temp //= velscale_ratio
    #    factor = velscale_ratio
    #
    #  if (velscale_ratio !=1):
    #    velscale_temp = velscale/velscale_ratio
    #    pdb.set_trace()
    #
    #
    # @@@@@@@@@@@@@@@@@@@@@@@  Check if lstep can be taken from templates @@@@@@@@@@@@@@@@@@@@@
    # ------------------------------------
    # First of all find the emission-lines which we are effectively going
    # to fit.  That is, exclude from the input structure the lines that
    # are either being masked or not fitted.
    i_f = where_eq(emission_setup, "action", "f")
    emission_setup_in = emission_setup
    # Count the number of single lines or the number of multiplets
    i_lines = where_eq(emission_setup, "kind", "l")
    nlines = len(i_lines)
    # ------------------------------------
    # Make sure that the input amplitudes of each single line or of the
    # main lines of each multiplsts are either 1 or -1.
    for i in i_lines:
        ampl_i = np.float64(emission_setup[i].a)
        if ampl_i > 0:
            emission_setup[i].a = np.float64(1.0)
        elif ampl_i < 0:
            emission_setup[i].a = np.float64(-1.0)
    # ------------------------------------
    # Declare and fill in the array with the starting guesses for the
    # parameter to fit with MPFIT, namely V_gas and S_gas for the emission
    # lines parameters, the coefficients of the mult. polynomials and, if
    # necessary the reddening parameters. The latter are placed
    # right after the emission-line parameters.
    if reddening:
        start_pars = np.zeros(2 * nlines + mdegree + len_red)
    else:
        start_pars = np.zeros(2 * nlines + mdegree)
    # Loop over the lines, and assign starting V_gas and S_gas as from the
    # emission-line setup structure. These are set as starting position
    # and width in pixel units.
    h = 0
    for j in i_lines:
        # current emission-line index in the input setup structure
        # to deal with log10-lambda rebinned data, instead of ln-lambda
        if log10:
            offset = (np.log10(emission_setup[j]._lambda) - l0_gal) / lstep_gal
        else:
            offset = (np.log(emission_setup[j]._lambda) - l0_gal) / lstep_gal
        start_pars[h + 0] = emission_setup[j].v / velscale + offset
        start_pars[h + 1] = emission_setup[j].s / velscale
        h = h + 2
    # add if necessary the starting E(B-V) values
    if reddening:
        start_pars[2 * nlines : 2 * nlines + len(reddening)] = reddening
    # ------------------------------------
    # Convolve the input stellar templates with the input stellar kinematics
    kinstars = sol[0:6]
    if vsyst == 0:
        npix_gal = len(galaxy)
        cstar = convolve_templates(
            templates, kinstars, velscale, npix_gal
        )  # , velscale_ratio)
    # !New (Allow oversample)
    else:
        npix_gal = len(galaxy)
        cstar = convolve_templates_new(
            templates, kinstars, velscale, npix_gal, velscale_ratio, vsyst
        )
    #
    #
    # !NEW version \added ......................................
    # int_disp_var
    # ; ------------------------------------
    # ; Check whether int_disp is single number or a 2-dimensional
    # ; [wavelenght, int_disp] array specifying the variation in wavelength
    # ; of the line-spread function. Then create a int_disp vector
    # ; specifying the instrumental resolution (in km/s and in sigma, as for
    # ; the single valued constant case) at the observed location of the
    # ; emission lines that we are going to use, interpolating the input
    # ; values in one case and replicating the input constant int_disp
    # ; values in the other.
    int_disp_in = int_disp
    if np.size(int_disp) > 1:
        c = np.float64(299792.4580)  # Speed of light in km/s
        if log10 != 1:
            Vstar = kinstars[0] + (l0_gal - l0_templ) * c
        if log10 == 1:
            Vstar = kinstars[0] + (l0_gal - l0_templ) * c * np.log(np.float64(10.000))
        interp_x = [x._lambda * (np.exp(Vstar / c)) for x in emission_setup]
        int_f = interp1d(
            np.ravel(int_disp[0, :]),
            np.ravel(int_disp[1, :]),
            "linear",
            fill_value="extrapolate",
        )
        int_disp = int_f(interp_x)
    else:
        int_disp = np.array((range(len(emission_setup)))) * 0.0 + int_disp
    # !NEW version \added ......................................
    # ------------------------------------
    # Set the limits and the appropriate inter-dependencies for fitting /FIRST CALL/
    # emission-line Gaussians and prepare the FUNCTARGS and PARINFO
    # arguments to be passed to MPFIT

    # step over starting guesses to stabilise output
    steps = [0.0, -3 * velscale, 3 * velscale]
    store_min_res = np.zeros(len(steps))
    store_pars = []
    for ii, step in enumerate(steps):
        # update starting parameters with small offset
        istart_pars = copy.deepcopy(start_pars)
        h = 0
        for j in i_lines:
            # current emission-line index in the input setup structure
            # to deal with log10-lambda rebinned data, instead of ln-lambda
            istart_pars[h + 0] += step
            h = h + 2

        parinfo_2, functargs_2 = set_constraints(
            galaxy,
            noise,
            cstar,
            kinstars,
            velscale,
            degree,
            mdegree,
            goodpixels,
            emission_setup,
            istart_pars,
            l0_gal,
            lstep_gal,
            int_disp,
            log10,
            reddening,
            l0_templ,
        )
        # ------------------------------------
        # This is where the GANDALF fit is actually performed. Call MPFIT to
        # find the best-fitting position and width of the emission lines while
        # using BVLS at each iteration to solve for the relative weights of
        # the stellar and emission-line templates. The latter weights
        # correspond to the emission-line amplitudes. Solve also for the
        # mdegree mult. polynomial coeffients and, if needed, also for the
        # best reddening parameters.
        #
        # Note that we evalutate also the errors on the best parameters, but
        # as regards the position and width of the lines these should only be
        # considered as lower estimates for the real uncertainties.
        mpfit_out = mpfit(
            fitfunc_gas,
            xall=istart_pars,
            functkw=functargs_2,
            parinfo=parinfo_2,
            ftol=1e-5,
            quiet=1,
        )
        status = mpfit_out.status
        ncalls = mpfit_out.nfev
        errors = mpfit_out.perror
        errmsg = mpfit_out.errmsg
        best_pars = mpfit_out.params
        # ------------------------------------
        # Call again FITFUNC_GAS with the best paramenters, not only to
        # compute the final fit residuals and hence assess the quality of the
        # fit, but also to retrieve:
        # the best fitting template weights   (WEIGHTS)
        # the best fitting overall model      (BESTFIT)
        # the best fitting emission templates (EMISSION_TEMPLATES)
        bestfit = []
        weights = []
        emission_templates = []
        st, resid = fitfunc_gas(
            best_pars,
            cstar=cstar,
            galaxy=galaxy,
            noise=noise,
            kinstars=kinstars,
            velscale=velscale,
            degree=degree,
            mdegree=mdegree,
            goodpixels=goodpixels,
            bestfit=bestfit,
            weights=weights,
            emission_setup=emission_setup,
            l0_gal=l0_gal,
            lstep_gal=lstep_gal,
            emission_templates=emission_templates,
            int_disp=int_disp,
            log10=log10,
            reddening=reddening,
            l0_templ=l0_templ,
        )
        bestfit = np.array(bestfit)
        weights = np.array(weights)
        emission_templates = np.array(emission_templates)
        chi2 = 0
        if sum(noise) == len(galaxy):
            # If you have input as errors on the fluxes an array of constant unity vales
            # compute Chi^2/DOF and use this instead of bestnorm/dof to rescale the formal uncertainties
            chi2 = robust_sigma(resid, zero=True) ** 2
            errors = errors * np.sqrt(chi2)
        # ------------------------------------
        # Add up the best-fitting emission templates to get the emission spectrum
        if len(emission_templates) == 1:
            emission = emission_templates[0]
        elif len(emission_templates) > 1:
            emission = np.sum(emission_templates, axis=0)
        else:
            err_msg_exit("Wrong size of emission templates")

        store_pars.append(
            (
                status,
                ncalls,
                errors,
                errmsg,
                best_pars,
                bestfit,
                weights,
                emission_templates,
            )
        )
        store_min_res[ii] = chi2
    best_res = np.argmin(store_min_res)

    (
        status,
        ncalls,
        errors,
        errmsg,
        best_pars,
        bestfit,
        weights,
        emission_templates,
    ) = store_pars[best_res]
    chi2 = store_min_res[best_res]

    # ------------------------------------
    #  Rearrange the final results (both best_pars and weights) in the
    # output array SOL, which includes also line fluxes. Fill in also the
    #  ESOL error array.
    sol, esol = rearrange_results(
        best_pars,
        weights,
        l0_gal,
        lstep_gal,
        velscale,
        emission_setup,
        int_disp,
        log10,
        reddening,
        errors,
    )

    # Appends to the best-fitting gas results also the mdegree polynomial
    # coefficients.
    if mdegree != 0:
        sol = np.append(sol, best_pars[len(best_pars) - mdegree : len(best_pars)])

    # this shouldn't be required, but make a deep copy of the solution to keep it safe.
    sol_safe = copy.deepcopy(sol)

    # Show the fit if requested
    if plot and not for_errors:
        show_fit(
            galaxy,
            bestfit,
            emission,
            best_pars,
            sol,
            goodpixels,
            mdegree,
            reddening,
            l0_gal,
            lstep_gal,
            l0_templ,
            log10,
            kinstars,
            nlines,
        )

    # ------------------------------------
    # JTM: this is where all the changes happen, fitting MC realisations of the best fit template
    # ------------------------------------
    # Properly compute error estimates on all emission-line parameters, by
    # solving non-linearly also for the line amplitudes with MPFIT, not
    # only for the line positions and widths, as done previously. BVLS
    # will now deal only with the weight of the stellar templates. We will
    # start such new fit from the previous solution.

    if for_errors == 1:
        nmc = 50
        eout_all = np.zeros((nmc, len(sol) - mdegree), dtype=np.float32)

        # -----------------
        # Set up the starting guesses for the new fit, including now the amplitudes.
        if len_red != 0:  # EDITED BY ADRIAN, 06. MAY 2019
            start_pars = np.zeros(2 * nlines + mdegree + len_red)
        else:
            start_pars = np.zeros(2 * nlines + mdegree)

        # Populate the starting parameter array based on sol which
        # lists, in the order:
        # (F_gas, A_gas, V_gas, S_gas) for each line
        # Reddening E(B-V) value(s)     - if any
        # Mult. polynomial coefficients - if any
        h = 0
        for i in range(nlines):
            # pull best-fit emission-line parameters for this line
            sol_i = sol[i * 4 + 2 : i * 4 + 4]  # pulls only v and s

            # current emission-line index in the input setup structure
            # to deal with log10-lambda rebinned data, instead of ln-lambda
            j = i_lines[i]
            if log10:
                offset = (np.log10(emission_setup[j]._lambda) - l0_gal) / lstep_gal
            else:
                offset = (np.log(emission_setup[j]._lambda) - l0_gal) / lstep_gal
            start_pars[h + 0] = sol_i[0] / velscale + offset
            start_pars[h + 1] = sol_i[1] / velscale
            h += 2

        # If needed, add the starting reddening guesses
        if len_red != 0:  # EDITED BY ADRIAN, 06. MAY 2019
            start_pars[2 * nlines : 2 * nlines + len_red] = sol[
                4 * nlines : 4 * nlines + len_red
            ]

        # If needed, add the starting mult. polynomial coefficients
        # which are at the end of the sol solution array
        if mdegree != 0:
            start_pars[len(start_pars) - mdegree : len(start_pars)] = sol[
                len(sol) - mdegree : len(sol)
            ]

        # -----------------
        # re-initilize the fit constrains and appropriate inter-dependencies
        # for the parameters to be fitted. Only need to do this once
        iparinfo, ifunctargs = set_constraints(
            galaxy,
            noise,
            cstar,
            kinstars,
            velscale,
            degree,
            mdegree,
            goodpixels,
            emission_setup,
            start_pars,
            l0_gal,
            lstep_gal,
            int_disp,
            log10,
            reddening,
            l0_templ,
        )

        # for each MC iteration, generate a new galaxy spectrum based on the bestfit and shuffled residuals
        base_resid = copy.deepcopy(resid)  # NB: these residuals are error normalised!
        for mc_iter in range(nmc):
            np.random.shuffle(base_resid)  # shuffle the residuals!

            # generate a mock galaxy with residual properties matching the observed ata
            igalaxy = copy.deepcopy(bestfit)
            igalaxy[goodpixels] += (
                base_resid * noise[goodpixels]
            )  # rescale residuals by the (unshuffled) noise.

            # paranoid nan/inf catching
            igalaxy[np.isnan(igalaxy) | ~np.isfinite(igalaxy)] = 0.0

            # update the galaxy spectrum in functargs. No need to redo the full setup, just replace in dict.
            ifunctargs["galaxy"] = igalaxy

            # -----------------
            # Re-run MPFIT starting from previous solution and using now the
            # FOR_ERRORS keyword to specify that we solve non-linearly also for
            # the amplitudes, and not only for the line position and width.

            mpfit_out = mpfit(
                fitfunc_gas,
                xall=start_pars,
                functkw=ifunctargs,
                parinfo=iparinfo,
                ftol=1e-5,
                quiet=1,
            )
            status_2 = mpfit_out.status
            ncalls_2 = mpfit_out.nfev
            errors_2 = mpfit_out.perror
            best_pars_2 = mpfit_out.params
            # -----------------
            # Re-evaluate the fit residuals to re-assess the fit quality and
            # rescale the errors. The last MPFIT fit should have always
            # converged to the input best solution Also, the weights here are
            # set to unity for the emission-line templates, as their amplitude
            # is determined by MPFIT.
            bestfit_2 = []
            weights_2 = []
            emission_templates_2 = []

            st, resid_2 = fitfunc_gas(
                best_pars_2,
                cstar=cstar,
                galaxy=igalaxy,
                noise=noise,
                kinstars=kinstars,
                velscale=velscale,
                degree=degree,
                mdegree=mdegree,
                goodpixels=goodpixels,
                bestfit=bestfit_2,
                weights=weights_2,
                emission_setup=emission_setup,
                l0_gal=l0_gal,
                lstep_gal=lstep_gal,
                emission_templates=emission_templates_2,
                int_disp=int_disp,
                log10=log10,
                reddening=reddening,
                l0_templ=l0_templ,
            )
            bestfit_2 = np.array(bestfit_2)
            weights_2 = np.array(weights_2)
            emission_templates_2 = np.array(emission_templates)

            # -----------------
            # Rearrange the final results in the output array SOL, which
            # includes also line fluxes. This time evaluate also the errors on
            # these last values, using for now a simple MC error propagation

            sol_2, _ = rearrange_results(
                best_pars_2,
                weights_2,
                l0_gal,
                lstep_gal,
                velscale,
                emission_setup,
                int_disp,
                log10,
                reddening,
                errors_2,
            )
            eout_all[mc_iter, :] = sol_2

        esol_2 = np.nanstd(eout_all, axis=0)

        # -----------------
        # Rewrite on the final solution array
        esol = esol_2
        # best_pars = best_pars_2
        # bestfit = bestfit_2
        # emission = emission_2
        # emission_templates = emission_templates_2
        # weights = weights_2
        # -----------------
        # Show the fit if requested
        if plot:
            show_fit(
                galaxy,
                bestfit,
                emission,
                best_pars,
                sol,
                goodpixels,
                mdegree,
                reddening,
                l0_gal,
                lstep_gal,
                l0_templ,
                log10,
                kinstars,
                nlines,
            )
    # ------------------------------------
    # If we used reddening, recover the reddened amplitudes.  In other
    # words, make sure we output the emission-line amplitudes as observed.
    # This is the right thing to later compute the amplitude-over-noise
    # ratio of the lines and decide whether they are detected
    if reddening:
        # make the spectrum wavelength array
        if not log10:
            ob_lambda = np.exp(
                np.arange(len(galaxy), dtype="float") * lstep_gal + l0_gal
            )
            Vstar = kinstars[0] + (l0_gal - l0_templ) * C
        else:
            ob_lambda = 10 ** (
                np.arange(len(galaxy), dtype="float") * lstep_gal + l0_gal
            )
            Vstar = kinstars[0] + (l0_gal - l0_templ) * C * mt.log(10.0)
        # receding velocity
        # total reddening attenuation that was applied to the emission lines
        # in FITFUNC_GAS
        reddening_attenuation = dust_calzetti(
            l0_gal, lstep_gal, len(galaxy), sol[nlines * 4], Vstar, log10
        )
        if len(reddening) == 2:
            int_reddening_attenuation = dust_calzetti(
                l0_gal, lstep_gal, len(galaxy), sol[nlines * 4 + 1], Vstar, log10
            )
        else:
            int_reddening_attenuation = 1.0
        # get the reddening attenuation at the line wavelength
        interp_x = ob_lambda / np.exp(Vstar / C)
        interp_y = reddening_attenuation * int_reddening_attenuation
        interp_xout = [emission_setup[x]._lambda for x in i_lines]
        red_f = interp1d(interp_x, interp_y, kind="linear", fill_value="extrapolate")
        reddening_attenuation_emission = red_f(interp_xout)
        for l in range(nlines):
            # Finally, attenuate the output amplitude
            sol[l * 4 + 1] = sol[l * 4 + 1] * reddening_attenuation_emission[l]
            # and corresponding errors on the amplitudes
            if for_errors:
                esol[l * 4 + 1] = esol[l * 4 + 1] * reddening_attenuation_emission[l]
    # ------------------------------------
    if not quiet:
        resid = galaxy - bestfit
        resid_noise = robust_sigma(resid[goodpixels], zero=True)
        flux_tag = "Flux" if not reddening else "Deredd.Flux"
        print(
            "\n%8s %11s %13s %8s %14s %12s"
            % ("Line", flux_tag, "Ampl.", "V", "sig", "A/N")
        )
        print(
            " -------------------------------------------------------------------------"
        )
        for l, i in enumerate(i_lines):
            print(
                "%8s %12.4f %12.4f %12.4f %12.4f %12.4f"
                % (
                    emission_setup[i].name,
                    sol[l * 4],
                    sol[l * 4 + 1],
                    sol[l * 4 + 2],
                    sol[l * 4 + 3],
                    sol[l * 4 + 1] / resid_noise,
                )
            )

        if reddening:
            # make the spectrum wavelength array
            if not log10:
                ob_lambda = np.exp(
                    np.arange(len(galaxy), dtype="float") * lstep_gal + l0_gal
                )
                Vstar = kinstars[0] + (l0_gal - l0_templ) * C
            else:
                ob_lambda = 10 ** (
                    np.arange(len(galaxy), dtype="float") * lstep_gal + l0_gal
                )
                Vstar = kinstars[0] + (l0_gal - l0_templ) * C * mt.log(10.0)
            # receding velocity
            # total reddening attenuation that was applied to the emission lines
            # in FITFUNC_GAS
            reddening_attenuation = dust_calzetti(
                l0_gal, lstep_gal, len(galaxy), sol[nlines * 4], Vstar, log10
            )
            if len(reddening) == 2:
                print("E(B-V)_int = " + str(esol[nlines * 4 + 1]))
        print("\nRoN = " + str(resid_noise))
        print("\nfeval = " + str(ncalls))
        print("\n")
    # ------------------------------------
    # Restore the input emission-line setup structure
    emission_setup = emission_setup_in  # dummy
    int_disp = int_disp_in
    return weights, emission_templates, bestfit, sol_safe, esol
