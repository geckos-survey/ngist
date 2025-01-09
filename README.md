   The nGIST Pipeline: A galaxy IFS analysis pipeline for modern IFS data
===============================================================================
This is the nGIST pipeline, an actively-developed and updated version of the GIST pipeline.
Useful for all galaxy IFS data, but specially developed and extensively tested with MUSE,
nGIST provides numerous updates and improvements over the GIST pipeline. 

Lead Developers
-------------
Amelia Fraser-McKelvie & Jesse van de Sande

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
Bittner et al. 2019 (https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B) and include its ASCL entry 
(http://ascl.net/1907.025) in a footnote. 

nGIST builds on pre-existing software and is indebted to the work of several teams. We ask the user to also cite the papers of the underlying analysis techniques and models, if these are used in their work. In the default nGIST implementation, this includes the adaptive Voronoi tesselation routine of Cappellari & Copin 2003 (https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C).

For the science modules:

- If you use the 'ppxf' routine of the KIN module, please cite the penalised pixel-fitting method (pPXF): Cappellari & Emsellem 2004 (https://ui.adsabs.harvard.edu/abs/2004PASP..116..138C); Cappellari 2017 
(https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C), 
Cappellari 2023 (https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C), and the analysis improvements made by 
van de Sande et al. 2017 (https://ui.adsabs.harvard.edu/abs/2017ApJ...835..104V).

- If you use the 'ppxf' routine of the CONT module, please cite the above pPXF references.

- If you use the 'ppxf' routine of the GAS module, please cite the above pPXF references.
If you use the 'gandalf' routine of the GAS module, please cite Sarzi et al. 2006 (https://ui.adsabs.harvard.edu/abs/2006MNRAS.366.1151S) (ASCL: https://ascl.net/1708.012)
If you use the 'magpi_gandalf' routine of the GAS module, please cite Battisti et al., (in prep).

- If you use the 'ppxf' routine of the SFH module, please cite the same references as for the KIN module (if not cited already).

- If you use the 'default' routine of the LS module, please cite the LIS measurement definitions of Kuntschner et al. 2006 (https://ui.adsabs.harvard.edu/abs/2006MNRAS.369..497K), and the implemntation algorithm of routine of Martin-Navarro et al. 2018 (https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.3700M).

Finally, don't forget to attribute the stellar templates used in your analysis. Included in this distribution are the MILES models of Vazdekis et al. 2010 (https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1639V).

Disclaimer
----------
Although we provide this software as a convenient, all-in-one framework for the analysis of integral-field spectroscopic
data, it is of fundamental importance that the user understands exactly how the involved analysis methods work. We warn
that the improper use of any of these analysis methods, whether executed within the framework of the nGIST or not, will
likely result in spurious or erroneous results and their proper use is solely the responsibility of the user. Likewise,
the user should be fully aware of the properties of the input data before intending to derive high-level data products.
Therefore, this software framework should not be simply adopted as a black-box. To this extend, we urge any user to get
familiar with both the input data and analysis methods, as well as their implementation.





