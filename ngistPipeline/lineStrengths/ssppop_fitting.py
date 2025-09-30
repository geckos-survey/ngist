#!/usr/bin/env python
import optparse
import os
import sys
import warnings

import emcee
# from   joblib              import Parallel, delayed
import matplotlib.pyplot as plt
# import h5py
import numpy
import scipy.spatial.qhull as qhull
from astropy.io import fits


# ===============================================================================
#
# RUN_SSPPOP_FITTING
# This function runs a specific case in the configuration file
#
# Jesus Falcon-Barroso, IAC, June 2016
# ===============================================================================
def load_data(datafile, index_names):
    ## Loading the FITS table with observed values
    hdu = fits.open(datafile)
    data = hdu[1].data

    ## Finding unique Voronoi bin values
    u, ibins = numpy.unique(data.field("BIN_ID"), return_index=True)
    u, ispax = numpy.unique(data.field("BIN_ID"), return_inverse=True)
    nspax = len(ispax)
    nbins = len(ibins)
    nindex = len(index_names)

    ## Creating array of observed indices
    obs_indices = numpy.zeros([nbins, nindex])
    err_indices = numpy.zeros([nbins, nindex])
    for i in range(nindex):
        tmp = data.field(index_names[i])
        obs_indices[:, i] = tmp[ibins]
        tmp = data.field("D" + index_names[i])
        err_indices[:, i] = tmp[ibins]

    ## Creating structure with extra info
    struct = {
        "X": data.field("X"),
        "Y": data.field("Y"),
        "XBIN": data.field("XBIN"),
        "YBIN": data.field("YBIN"),
        "BIN_ID": data.field("BIN_ID"),
        "IBINS": ibins,
        "ISPAX": ispax,
        "NBINS": nbins,
        "NSPAX": nspax,
    }

    return obs_indices, err_indices, struct


# ===============================================================================
def load_models(modelfile, index_names):
    ## Loading the FITS table with observed values
    hdu = fits.open(modelfile)
    model = hdu[1].data
    nmodels = len(model.field(index_names[0]))
    nindex = len(index_names)

    ## Creating array of observed indices
    model_indices = numpy.zeros([nmodels, nindex])
    for i in range(nindex):
        tmp = model.field(index_names[i])
        model_indices[:, i] = tmp

    ## Creating the triangulation array
    params = numpy.empty((nmodels, 3))
    params[:, 0] = model.field("AGE")
    params[:, 1] = model.field("MET")
    params[:, 2] = model.field("ALPHA")
    tri = qhull.Delaunay(params, qhull_options="QJ")
    labels = ["AGE", "METAL", "ALPHA"]

    return model_indices, params, tri, labels


# ==============================================================================
def printProgress(
    iteration, total, prefix="", suffix="", decimals=2, barLength=80, color="g"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        color       - Optional  : color identifier (Str)
    """
    if color == "y":
        color = "\033[43m"
    elif color == "k":
        color = "\033[40m"
    elif color == "r":
        color = "\033[41m"
    elif color == "g":
        color = "\033[42m"
    elif color == "b":
        color = "\033[44m"
    elif color == "m":
        color = "\033[45m"
    elif color == "c":
        color = "\033[46m"

    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = color + " " * filledLength + "\033[49m" + " " * (barLength - filledLength - 1)
    sys.stdout.write("\r%s |%s| %s%s %s\r" % (prefix, bar, percents, "%", suffix)),
    sys.stdout.flush()


# ==============================================================================
def interp_weights(xyz, uvw, tri):
    # Creates a Delaunay triangulation and finds the vertices and weights of
    # points around a given location in parameter space

    d = len(uvw[0, :])
    simplex = tri.find_simplex(uvw)
    vertices = numpy.take(tri.simplices, simplex, axis=0)
    temp = numpy.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = numpy.einsum("njk,nk->nj", temp[:, :d, :], delta)

    return vertices, numpy.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


# ==============================================================================
def interpolate(values, vtx, wts):
    return numpy.einsum("nj,nj->n", numpy.take(values, vtx), wts)


# ==============================================================================
def lnprior(par, model_pars):
    # par = [Age, Met, Alpha] in no particular order
    if (
        # Rejecting solutions outside some boundary limits
        (par[0] >= numpy.amin(model_pars[:, 0]))
        and (par[0] <= numpy.amax(model_pars[:, 0]))
        and (par[1] >= numpy.amin(model_pars[:, 1]))
        and (par[1] <= numpy.amax(model_pars[:, 1]))
        and (par[2] >= numpy.amin(model_pars[:, 2]))
        and (par[2] <= numpy.amax(model_pars[:, 2]))
    ):
        return 0.0

    return -numpy.inf


# ==============================================================================
def compute_indices(par, data, model_indices, params, tri):
    input_pt = numpy.array(par, ndmin=2)
    vtx, wts = interp_weights(params, input_pt, tri)
    outindices = numpy.zeros(len(data))
    for i in range(len(model_indices[0, :])):
        outindices[i] = interpolate(model_indices[:, i], vtx, wts)

    return outindices


# ==============================================================================
def lnprob(par, data, error, model_indices, params, tri):
    # Checking the priors
    lp = lnprior(par, params)
    if not numpy.isfinite(lp):
        return -numpy.inf

    # Interpolating the model grid indices at desired point
    out_indices = compute_indices(par, data, model_indices, params, tri)

    # Computing the likelyhood for a given set of params
    bad = error <= 0.0
    if numpy.any(bad):
        error = error * 0.0 + 1e10
    inv_sigma2 = 1.0 / (error**2)
    lnlike = -0.5 * numpy.sum(
        (data - out_indices) ** 2 * inv_sigma2 - numpy.log(inv_sigma2)
    )

    # Safety check. If lnlike is not finite then return -numpy.inf
    if not numpy.isfinite(lnlike):
        return -numpy.inf

    return lp + lnlike


# ==============================================================================
def ssppop_fitting(
    data,
    error,
    model_indices,
    params,
    tri,
    labels,
    nwalkers,
    nchain,
    plot,
    verbose,
    progress,
    ncases,
    outdir,
):
    # Print progressbar
    # printProgress(progress, ncases, prefix = ' Progress:', suffix = 'Complete', barLength = 50)

    ## Defining some parameters of the fitting
    ndim = len(params[0, :])

    # Defining an initial set of walkers
    zpt = numpy.mean(params, axis=0)
    kick = [0.05, 0.05, 0.05]
    p0 = [zpt + kick * numpy.random.randn(ndim) for i in range(nwalkers)]

    # Setting up the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(data, error, model_indices, params, tri)
    )

    # Running the Markov chain for NCHAIN iterations
    sampler.reset()
    #    print("")
    for counter, result in enumerate(sampler.sample(p0, iterations=nchain)):
        if verbose == 1:
            printProgress(
                counter + 1,
                nchain,
                prefix=" Progress:",
                suffix="Complete",
                barLength=50,
            )

    #    print("Mean acceptance fraction: {0:.3f}".format(numpy.mean(sampler.acceptance_fraction)))
    #    print("Autocorrelation time:", sampler.get_autocorr_time())

    samples = sampler.chain
    shape = samples.shape
    nparams = shape[2]
    samples_flat = samples[:, :, :].reshape(
        (-1, nparams)
    )  # Reshaping the output to [nchain,ndim]
    nsamples = numpy.tile(numpy.linspace(0, shape[1] - 1, shape[1]), shape[0])
    w = nsamples >= 0.5 * shape[1]
    good_samples = samples_flat[w, :]

    # If desired, plot MCMC chain and corner plot
    if plot == True:
        for i in range(ndim):
            kk = sampler.chain[:, :, i]
            idx = int(str(ndim) + "1" + str(i + 1))
            plt.subplot(idx)
            plt.plot(kk.T)
            plt.ylabel(labels[i])
        plt.savefig(outdir + "maps/MCMC/Chain_BINID" + str(progress) + ".pdf", dpi=30)
        corner.corner(
            good_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            verbose=True,
            plot_datapoints=False,
        )
        plt.savefig(outdir + "maps/MCMC/Corner_BINID" + str(progress) + ".pdf", dpi=30)

    # Storing results
    outpars = numpy.zeros(3 * ndim + 2)  # Params (3*ndim), LnP, flag
    for i in range(ndim):
        outpars[3 * i] = numpy.percentile(good_samples[:, i], 50)
        outpars[3 * i + 1] = numpy.percentile(good_samples[:, i], 16) - outpars[3 * i]
        outpars[3 * i + 2] = numpy.percentile(good_samples[:, i], 84) - outpars[3 * i]

    # Computing the chi2
    out_indices = compute_indices(
        [outpars[0], outpars[3], outpars[6]], data, model_indices, params, tri
    )
    bad = error <= 0.0
    if numpy.any(bad):
        error = error * 0.0 + 1e10
    inv_sigma2 = 1.0 / error**2
    outpars[3 * ndim] = -0.5 * numpy.sum(
        (data - out_indices) ** 2 * inv_sigma2 - numpy.log(inv_sigma2)
    )

    # Flag cases where solution is close to a boundary of parameter space
    outpars[3 * ndim + 1] = 1
    for i in range(ndim - 1):
        dlo = numpy.abs(outpars[3 * i] - numpy.amin(params[:, i]))
        dhi = numpy.abs(outpars[3 * i] - numpy.amax(params[:, i]))
        if (dlo < 0.01) or (dhi < 0.01):
            outpars[3 * ndim + 1] = 0

    return outpars, good_samples


# ===============================================================================
if __name__ == "__main__":
    os.system("clear")
    warnings.filterwarnings("ignore")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -c cfile -nc ncores")
    parser.add_option(
        "-d",
        "--data",
        dest="datafile",
        type="string",
        default="../results/NGC5746.MIUSCATssp.SN100_results_v2.fits",
        help="FITS table with observed indices results",
    )
    parser.add_option(
        "-m",
        "--models",
        dest="modelfile",
        type="string",
        default="MILES_KB_LIS8.4.fits",
        help="FITS table with the model predictions",
    )
    parser.add_option(
        "-i",
        "--indices",
        dest="indices",
        type="string",
        default="Hbetao,Fe5015,Mgb,Fe5270,Fe5335",
        help="Set of indices to use",
    )
    parser.add_option(
        "-n",
        "--ncores",
        dest="ncores",
        type="int",
        default=1,
        help="number of cores to use",
    )
    parser.add_option(
        "-w",
        "--nwalkers",
        dest="nwalkers",
        type="int",
        default=100,
        help="number of walkers to use",
    )
    parser.add_option(
        "-k",
        "--nchain",
        dest="nchain",
        type="long",
        default=1000,
        help="number of iterations to run",
    )
    parser.add_option(
        "-p",
        "--plot",
        dest="plot",
        type="int",
        default=0,
        help="Switch to plot (or not the chain and corner plot",
    )
    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        type="int",
        default=0,
        help="See progress bar foreach chain",
    )

    (options, args) = parser.parse_args()
    datafile = options.datafile
    modelfile = options.modelfile
    indices = options.indices
    ncores = options.ncores
    nwalkers = options.nwalkers
    nchain = options.nchain
    plot = options.plot
    verbose = options.verbose

    ## Identifying the indices to fit
    index_names = indices.split(",")
    nindex = len(index_names)

    ## Loading the data
    print("# Loading data: " + datafile)
    data, error, datastruct = load_data(datafile, index_names)
    ncases = datastruct["NBINS"]
    nspax = datastruct["NSPAX"]
    nbins = datastruct["NBINS"]
    ispax = datastruct["ISPAX"]
    #    ncases = 3
    print("- Ncases:", ncases)

    ## Loading model predictions
    print("# Loading model predictions: " + modelfile)
    model_indices, params, tri, labels = load_models(modelfile, index_names)

    ## Running (in parallel) the minimization for each case in data
    print("# Running the Markov Chain...")
    print("- Nwalkers:", nwalkers)
    print("- Nchain:", nchain)
    tmp = Parallel(n_jobs=ncores, verbose=50)(
        delayed(ssppop_fitting)(
            data[i, :],
            error[i, :],
            model_indices,
            params,
            tri,
            labels,
            nwalkers,
            nchain,
            plot,
            verbose,
        )
        for i in range(ncases)
    )
    vals, chains = zip(*tmp)
    vals = numpy.array(vals)
    chains = numpy.array(chains)

    ## Saving the chain results in a FITS table
    tmpname = os.path.basename(datafile)
    rootname = os.path.splitext(tmpname)[0]
    outfits = "../results/" + rootname + "_ssppop_v3.fits"
    if os.path.exists(outfits):
        os.remove(outfits)
    print("")
    print("# Results will be stored in the FITS table: " + outfits)
    print("")
    cols = []
    cols.append(fits.Column(name="X", format="D", array=datastruct["X"]))
    cols.append(fits.Column(name="Y", format="D", array=datastruct["Y"]))
    cols.append(fits.Column(name="XBIN", format="D", array=datastruct["XBIN"]))
    cols.append(fits.Column(name="YBIN", format="D", array=datastruct["YBIN"]))
    cols.append(fits.Column(name="BIN_ID", format="D", array=datastruct["BIN_ID"]))
    ndim = len(params[0, :])
    for i in range(ndim):
        cols.append(fits.Column(name=labels[i], format="D", array=vals[ispax, 3 * i]))
        cols.append(
            fits.Column(
                name="D" + labels[i] + "_LO", format="D", array=vals[ispax, 3 * i + 1]
            )
        )
        cols.append(
            fits.Column(
                name="D" + labels[i] + "_HI", format="D", array=vals[ispax, 3 * i + 2]
            )
        )
    cols.append(fits.Column(name="lnP", format="D", array=vals[ispax, -2]))
    cols.append(fits.Column(name="Flag", format="D", array=vals[ispax, -1]))
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.writeto(outfits)

    outhdf5 = "../results/" + rootname + "_ssppop_chains_v3.hdf5"
    print("# MCMC chains will be stored in the HDF5 file: " + outhdf5)
    if os.path.exists(outhdf5):
        os.remove(outhdf5)
    print("")
    f = h5py.File(outhdf5, "w")
    dset = f.create_dataset("chains", data=chains, compression="gzip")
    f.close()

    print("# Done!")
