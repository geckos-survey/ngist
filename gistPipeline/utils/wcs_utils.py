#!/usr/bin/env python

import numpy as np
from astropy.wcs import WCS


def strip_wcs_from_header(header):
    """
    Given a header with WCS information, remove ALL WCS information from that
    header
    """

    hwcs = WCS(header)
    wcsh = hwcs.to_header()

    keys_to_keep = [k for k in header if (k and k not in wcsh and "NAXIS" not in k)]

    newheader = header.copy()

    # Strip blanks first.  They appear to cause serious problems, like not
    # deleting things they should!
    if "" in newheader:
        del newheader[""]

    for kw in list(newheader.keys()):
        if kw not in keys_to_keep:
            del newheader[kw]

    for kw in (
        "CRPIX{ii}",
        "CRVAL{ii}",
        "CDELT{ii}",
        "CUNIT{ii}",
        "CTYPE{ii}",
        "PC0{ii}_0{jj}",
        "CD{ii}_{jj}",
        "CROTA{ii}",
        "PC{ii}_{jj}",
        "PC{ii:03d}{jj:03d}",
        "PV0{ii}_0{jj}",
        "PV{ii}_{jj}",
    ):
        for ii in range(5):
            for jj in range(5):
                k = kw.format(ii=ii, jj=jj)
                if k in newheader.keys():
                    del newheader[k]

    return newheader


def diagonal_wcs_to_cdelt(mywcs):
    """
    If a WCS has only diagonal pixel scale matrix elements (which are composed
    from cdelt*pc), use them to reform the wcs as a CDELT-style wcs with no pc
    or cd elements

    """
    offdiag = ~np.eye(mywcs.pixel_scale_matrix.shape[0], dtype="bool")
    if not any(mywcs.pixel_scale_matrix[offdiag]):
        cdelt = mywcs.pixel_scale_matrix.diagonal()
        del mywcs.wcs.pc
        del mywcs.wcs.cd
        mywcs.wcs.cdelt = cdelt
    return mywcs
