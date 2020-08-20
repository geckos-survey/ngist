#!/usr/bin/env python
import optparse
import os
import sys
import warnings
import numpy
import matplotlib.pyplot as plt
from   astropy.io        import ascii
from   astropy.io        import fits
#===============================================================================
#
# LSINDEX_SPEC
#
#  This function computes the Lick indices of a set of input spectra
#
#  NOTE: Input spectra is assumed to be in Angstroms.
#
# Jesus Falcon-Barroso, IAC, August 2016
#===============================================================================
#===============================================================================
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")

#==============================================================================
def load_inputlist(inlist):

    # Reading inputlist
    data = ascii.read(inlist,comment='\s*#')
    names = data['col1']
    redshift = data['col2']
    err_redshift = data['col3']

    return names, redshift, err_redshift
#==============================================================================
#
# FUNCTION: sum_counts()
#
def sum_counts(ll, c, b1, b2):

    # Central full pixel range
    dw = ll[1]-ll[0] # linear step size
    w  = ((ll >= b1+dw/2.) & (ll <= b2-dw/2.))
    s  = numpy.sum(c[w])

#    print( len( w[w==True] ) )
#    normalise = len( w[w==True] )
   
    # First fractional pixel
    pixb = ((ll < b1+dw/2.) & (ll > b1-dw/2.))
    if numpy.any(pixb):
       fracb = ((ll[pixb]+dw/2.)-b1)/dw
       s     = s+c[pixb]*fracb

    # Last fractional pixel
    pixr = ((ll < b2+dw/2.) & (ll > b2-dw/2.))
    if numpy.any(pixr):
       fracr = (b2-(ll[pixr]-dw/2.))/dw
       s     = s+c[pixr]*fracr

#    s = s / (normalise + fracr + fracb )
#    print(normalise, fracr, fracb, normalise + fracr + fracb)

    return s
#==============================================================================
#
# FUNCTION: calc_index()
#
def calc_index(bands,name,ll,counts,plot):

   cb = sum_counts(ll, counts, bands[0], bands[1])
   cr = sum_counts(ll, counts, bands[4], bands[5])
   s  = sum_counts(ll, counts, bands[2], bands[3])

   lb = (bands[0]+bands[1])/2.0
   lr = (bands[4]+bands[5])/2.0
   cb = cb / (bands[1]-bands[0])
   cr = cr / (bands[5]-bands[4])
   m  = (cr-cb) / (lr-lb)
   c1 = (m*(bands[2]-lb))+cb
   c2 = (m*(bands[3]-lb))+cb
   cont = 0.5*(c1+c2)*(bands[3]-bands[2])

#   print( cb, cr, s )
#   plot = 1

   if bands[6] == 1.:
     # atomic index
     ind = (1.0 - (s/cont))*(bands[3]-bands[2])
   elif bands[6] == 2.:
     # molecular index
     ind = -2.5*numpy.log10(s/cont)
#   print( ind )

   if plot > 0:
      minx = bands[0]-0.05*(bands[5]-bands[0])
      maxx = bands[5]+0.05*(bands[5]-bands[0])
      miny = numpy.amin(counts)-0.05*(numpy.amax(counts)-numpy.amin(counts))
      maxy = numpy.amax(counts)+0.05*(numpy.amax(counts)-numpy.amin(counts))
      plt.figure()
#      plt.plot(ll,counts,'k')
      plt.scatter(ll,counts,color='k')
      plt.xlabel("Wavelength ($\AA$)")
      plt.ylabel("Counts")
      plt.title(name)
      plt.xlim([minx,maxx])
      plt.ylim([miny,maxy])
      dw = ll[1]-ll[0]
      plt.plot([lb,lr],[c1*dw,c2*dw],'r')
      good = ((ll >= bands[2]) & (ll <= bands[3]))
      ynew = numpy.interp(ll,[lb,lr],[c1[0]*dw,c2[0]*dw])
      plt.fill_between(ll[good], counts[good], ynew[good],facecolor='green')
      for i in range(len(bands)):
         plt.plot([bands[i],bands[i]],[miny,maxy],'k--')
      plt.show()

   return ind
#==============================================================================
# purpose : Measure line-strength indices
#
# input : ll    - wavelength vector; assumed to be in *linear steps*
#         flux  - counts as a function of wavelength
#         noise - noise spectrum
#         z, z_err - redshift and error (in km/s)
#         lickfile - file listing the index definitions
#
# keywords  debug  - more than 0 gives some basic info
#           plot   - plot spectra
#           sims   - number of simulations for the errors (default: 100)
#
# output : names       - index names
#          index       - index values
#          index_error - index error values
#
# author : J. Falcon-Barroso
#
# version : 1.0  IAC (08/07/16) A re-coding of H. Kuntschner's IDL routine into python
#==============================================================================
def lsindex(ll, flux_in, noise, z, lickfile, plot=0, sims=0, z_err=0):

    # Deredshift spectrum to rest wavelength
    dll =(ll)/(z+1.)

    # Rebin to linear step
    flux = flux_in

    # Read index definition table
    tab   = ascii.read(lickfile, comment='\s*#')
    names = tab['names']
    bands = numpy.zeros((7,len(names)))
    bands[0,:] = tab['b1']
    bands[1,:] = tab['b2']
    bands[2,:] = tab['b3']
    bands[3,:] = tab['b4']
    bands[4,:] = tab['b5']
    bands[5,:] = tab['b6']
    bands[6,:] = tab['b7']

    # Measure line indices
    num_ind = len(bands[0,:])
    index   = numpy.zeros(num_ind)
    for k in range(num_ind):   # loop through all indices
        # check wether the wavelength range is o.k.
        if ((dll[0] <= bands[0,k]) and (dll[len(dll)-1] >= bands[5,k])):
           # calculate index value
           index0   = calc_index(bands[:,k], names[k], dll, flux, plot)
           index[k] = index0[0]
        else:
           # index outside wavelegth range
           index[k]= numpy.nan

    # Calculate errors
    index_error = numpy.zeros(num_ind,dtype='D'); index_error[:] = numpy.nan
    index_noise = numpy.zeros([num_ind,sims], dtype='D')

    if sims > 0:

        # Create redshift and sigma errors
        dz = numpy.random.randn(sims)*z_err

        # Loop through the simulations
        for i in range(sims):
            # resample spectrum according to noise
            ran   = numpy.random.normal(0.0,1.0,len(dll))
            flux_n = flux+ran*noise

            # loop through all indices
            for k in range(num_ind):
               # shift bands according to redshift error
               sz  = z + dz[i]
               dll = ll/(sz+1.)
               bands2 = bands[:,k]
               if ((dll[0] <= bands2[0]) and (dll[len(dll)-1] >= bands2[5])):
                  tmp = calc_index(bands2, names[k], dll, flux_n, 0)
                  index_noise[k,i]=tmp
               else:
                  # index outside wavelength range
                  index_noise[k,i]=numpy.nan

        # Get STD of distribution (index error)
        index_error = numpy.std(index_noise, axis=1)

    return names, index, index_error
#==============================================================================
if __name__ == "__main__":

    os.system('clear')
    warnings.filterwarnings("ignore")
    print("========================")
    print("= Running LSINDEX_SPEC =")
    print("========================")
    print("")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -i inputlist -l lickfile -o outfits -n nsims")
    parser.add_option("-i", "--inputlist", dest="inputlist", type="string", default="../config_files/miles_ku.inputlist",  help="List of input spectra, redshift and err_redshift")
    parser.add_option("-l", "--lickfile",  dest="lickfile",  type="string", default="../config_files/lick_bands.conf",     help="Lick file with index definitions")
    parser.add_option("-o", "--outfits",   dest="outfits",   type="string", default="../results/lick_indices.fits",        help="Name of output FITS table with results")
    parser.add_option("-n", "--nsims",     dest="nsims",     type="int",    default="0",                                   help="Number of MC simulations for errors")
    parser.add_option("-p", "--plot",      dest="plot",      type="int",    default="0",                                   help="Plotting or not [0/1]")

    (options, args) = parser.parse_args()
    inputlist = options.inputlist
    lickfile  = options.lickfile
    outfits   = options.outfits
    nsims     = options.nsims
    plot_flag = options.plot

    # Getting the list of FITS files to process
    print("# Loading inputlist: "+inputlist)
    inlist, redshift, err_redshift = load_inputlist(inputlist)
    nfiles = len(inlist)
    print("- "+str(nfiles)+" files found")
    print("")

    # Computing the magnitudes for each input FITS file
    print("# Computing indices...")
    root    = []
    for i in range(nfiles):

        # Opening the FITS file
        hdu     = fits.open(inlist[i])
        flux    = hdu[0].data
        npix    = len(flux)
        crpix   = hdu[0].header['CRPIX1']
        crval   = hdu[0].header['CRVAL1']
        cdelt   = hdu[0].header['CDELT1']
        wave    = ((numpy.arange(npix) + 1.0) - crpix) * cdelt + crval
        root    = numpy.append(root,os.path.basename(inlist[i]))

        # Computing the indices
        names, indices, errors = lsindex(wave, flux, flux*0.1, redshift[i], lickfile, plot=plot_flag, sims=nsims, z_err=err_redshift[i] )

        if i == 0:
            outls     = numpy.zeros((len(names),nfiles))
            outls_err = numpy.zeros((len(names),nfiles))

        outls[:,i] = indices
        outls_err[:,i] = errors

        printProgress(i+1, nfiles, prefix = ' ', suffix = 'Complete', barLength = 50)


    # Saving the results to a FITS table
    if os.path.exists(outfits):
       os.remove(outfits)
    print("# Results will be stored in the FITS table: "+outfits)
    print("")
    cols = []
    cols.append(fits.Column('Files', format='100A', array=root))
    ndim  = len(names)
    for i in range(ndim):
       cols.append(fits.Column(name=names[i],        format='D', array=outls[i,:]))
       cols.append(fits.Column(name="ERR_"+names[i], format='D', array=outls_err[i,:]))
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.writeto(outfits)

    print("# DONE!")
