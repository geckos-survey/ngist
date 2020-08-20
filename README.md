
   The GIST Framework: A multi-purpose tool for the analysis and visualisation of (integral-field) spectroscopic data
========================================================================================================================

The GIST is a convenient, all-in-one framework for the scientific analysis of fully reduced, (integral-field)
spectroscopic data. It is entirely written in Python3 and conducts all steps from the preparation of input data, over
the scientific analysis to the production of publication-quality plots.

In its default implementation, it extracts stellar kinematics, performs an emission-line analysis, derives star
formation histories and stellar population properties from full spectral fitting as well as via the measurement of
absorption line-strength indices. To this end, the GIST is exploiting the well-known pPXF and GandALF routines. In
addition, the framework is not specific to any instrument or analysis technique and provides easy means of modification
and further development, as of its modular code architecture. In fact, it is not only a neat combination of already
existing fitting routines, but a fully modular framework for the analysis of spectroscopic data in the context of a
variety of scientific objectives. 

The software further features the dedicated visualisation routine Mapviewer which has a sophisticated graphical user
interface. This allows the easy, fully-interactive plotting of all measurements, in particular maps, observed spectra,
fits, residuals, as well as star formation histories and the weight distribution of the models. 

An elaborate, Python-native parallelisation is implemented and tested on various machines from laptop to cluster scales. 

To date, the GIST framework has successfully been applied to both low and high-redshift data from MUSE, PPAK (CALIFA),
SINFONI, KCWI, and MaNGA, as well as to simulated data for HARMONI, WEAVE, and other artificial observations. 


Documentation
-------------
For a detailed documentation of the GIST framework, including instructions on installation, configuration, and a
tutorial, please see https://abittner.gitlab.io/thegistpipeline


Citing GIST and the analysis routines
-------------------------------------
If you use this software framework for any publication, please cite Bittner et al. 2019 (A&A, 628, A117;
https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B) and include its ASCL entry (http://ascl.net/1907.025) in a
footnote. 

We remind the user to also cite the papers of the underlying analysis techniques and models, if these are used in the
analysis. In the default GIST implementation, these are the adaptive Voronoi tesselation routine (Cappellari & Copin
2003), the penalised pixel-fitting method (pPXF; Cappellari & Emsellem 2004; Cappellari 2017), the pyGandALF routine
(Sarzi et al. 2006; Falcon-Barroso et al. 2006; Bittner et al. 2019), the line-strength measurement routines (Kuntschner
et al. 2006; Martin-Navarro et al. 2018), and the MILES models included in the tutorial (Vazdekis et al. 2010). 


Disclaimer
----------
Although we provide this software as a convenient, all-in-one framework for the analysis of integral-field spectroscopic
data, it is of fundamental importance that the user understands exactly how the involved analysis methods work. We warn
that the improper use of any of these analysis methods, whether executed within the framework of the GIST or not, will
likely result in spurious or erroneous results and their proper use is solely the responsibility of the user. Likewise,
the user should be fully aware of the properties of the input data before intending to derive high-level data products.
Therefore, this software framework should not be simply adopted as a black-box. To this extend, we urge any user to get
familiar with both the input data and analysis methods, as well as their implementation.


Acknowledgements
----------------
We thank Harald Kuntschner and Michele Cappellari for their permission to distribute their codes together with this
software package. We further thank Alexandre Vazdekis for permission to include the MILES library. The framework makes
use of Astropy, a community-developed core Python package for Astronomy (Astropy Collaboration et al. 2013, 2018), as
well as NumPy, SciPy and Matplotlib.


License
------------
This software is provided as is without any warranty whatsoever. Permission to use, for non-commercial purposes is
granted. Permission to modify for personal or internal use is granted, provided this copyright and disclaimer are
included in all copies of the software. Redistribution of the code, modified or not, is not allowed. All other rights
are reserved.
