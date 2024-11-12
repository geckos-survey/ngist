   The nGIST Pipeline: A galaxy IFS analysis pipeline for modern IFS data
===============================================================================
This is the nGIST pipeline, an actively-developed and updated version of the GIST pipeline.
Useful for all galaxy IFS data, but specially developed and extensively tested with MUSE,
nGIST provides numerous updates and improvements over the GIST pipeline. 

Documentation
-------------
For a detailed documentation of the nGIST pipeline, including instructions on installation and configuration, 
please see [https://geckos-survey.github.io/gist-documentation/](https://geckos-survey.github.io/gist-documentation/)

Usage 
-------------

In its default implementation, nGIST extracts stellar kinematics, creates continuum-only and line-only cubes, performs an 
emission-line analysis, derives star formation histories and stellar population properties from full spectral fitting 
as well as via the measurement of absorption line-strength indices. Outputs are easy-to-read 2D maps .fits files of 
various derived parameters, along with best fit spectra for those that want to dive further into the data. 
The handy, quick-look Mapviewer tool is also included with this distribution; a method for visualising your data products 
on the fly. 

Citing GIST and the analysis routines
-------------------------------------
If you use this software framework for any publication, please cite Fraser-McKelvie et al. (http://arxiv.org/abs/2411.03430).
Also consider citing the original GIST pipeline, the code for which the nGIST pipeline is based:
Bittner et al. 2019 (A&A, 628, A117; https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B) and include its ASCL entry 
(http://ascl.net/1907.025) in a footnote. 

We remind the user to also cite the papers of the underlying analysis techniques and models, if these are used in the
analysis. In the default GIST implementation, these are the adaptive Voronoi tesselation routine (Cappellari & Copin
2003), the penalised pixel-fitting method (pPXF; Cappellari & Emsellem 2004; Cappellari 2017, Cappellari 2023), 
the line-strength measurement routines (Kuntschner et al. 2006; Martin-Navarro et al. 2018), and the MILES models 
included in the tutorial (Vazdekis et al. 2010). 


Disclaimer
----------
Although we provide this software as a convenient, all-in-one framework for the analysis of integral-field spectroscopic
data, it is of fundamental importance that the user understands exactly how the involved analysis methods work. We warn
that the improper use of any of these analysis methods, whether executed within the framework of the nGIST or not, will
likely result in spurious or erroneous results and their proper use is solely the responsibility of the user. Likewise,
the user should be fully aware of the properties of the input data before intending to derive high-level data products.
Therefore, this software framework should not be simply adopted as a black-box. To this extend, we urge any user to get
familiar with both the input data and analysis methods, as well as their implementation.





