from setuptools import setup
from setuptools import find_packages
import os

def readme():
    with open('README.md') as file:
        return(file.read())

def versionNumber():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngistPipeline/_version.py')) as versionFile:
        return(versionFile.readlines()[-1].split()[-1].strip("\"'"))

setup(name='ngistPipeline',
      version=versionNumber(),
      description='The nGIST Pipeline: An updated multi-purpose tool for the analysis and visualisation of (integral-field) spectroscopic data',
      long_description_content_type="text/markdown",
      long_description=readme(),
      classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Development Status :: 5 - Production/Stable',
        'License :: MIT',
      ],
      url='https://geckos-survey.github.io/gist-documentation/',
      author='Amelia Fraser-McKelvie & Jesse van de Sande',
      author_email='a.fraser-mckelvie@eso.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'astropy',
        'numpy',
        'scipy',
        'matplotlib',
        'spectral-cube',
        'extinction',
        'ipython',
        'jupyter',
        'h5py',
        'joblib',
        'tqdm',
        'pip',
        'ppxf',
        'vorbin',
        'plotbin',
      ],
      python_requires='>=3.6',
      entry_points={
        'console_scripts': [
            'ngistPipeline        = ngistPipeline.MainPipeline:main'
            ],
        'gui_scripts': [
            'Mapviewer           = ngistPipeline.Mapviewer:main',
            'gistPlot_kin        = ngistPipeline.plotting.gistPlot_kin:main',
            'gistPlot_gas        = ngistPipeline.plotting.gistPlot_gas:main',
            'gistPlot_sfh        = ngistPipeline.plotting.gistPlot_sfh:main',
            'gistPlot_ls         = ngistPipeline.plotting.gistPlot_ls:main'
         ],
      },
      include_package_data=True,
      zip_safe=False)
