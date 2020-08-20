from setuptools import setup
from setuptools import find_packages
import os

def readme():
    with open('README.md') as file:
        return(file.read())

def versionNumber():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gistPipeline/_version.py')) as versionFile:
        return(versionFile.readlines()[-1].split()[-1].strip("\"'"))

setup(name='gistPipeline',
      version=versionNumber(),
      description='A Multi-Purpose IFS Analysis Pipeline',
      long_description_content_type="text/markdown",
      long_description=readme(),
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://abittner.gitlab.io/thegistpipeline/',
      author='Adrian Bittner',
      author_email='adrian.bittner@eso.org',
      license='Other/Proprietary License',
      packages=find_packages(),
      install_requires=[
        'astropy==3.1',
        'emcee==2.2.1',
        'matplotlib==3.1',
        'numpy==1.16.4',
        'PyQt5==5.10',
        'scipy==1.3.0',
        'pyyaml>5.3',
        'vorbin==3.1.3',
        'ppxf==6.7.15',
        'plotbin==3.1.3',
        'printStatus>=1.0',
      ],
      python_requires='==3.6.*',
      entry_points={
        'console_scripts': [
            'gistPipeline        = gistPipeline.MainPipeline:main'
            ],
        'gui_scripts': [
            'Mapviewer           = gistPipeline.Mapviewer:main',
            'gistPlot_kin        = gistPipeline.plotting.gistPlot_kin:main',
            'gistPlot_lambdar    = gistPipeline.plotting.gistPlot_lambdar:main',
            'gistPlot_gas        = gistPipeline.plotting.gistPlot_gas:main',
            'gistPlot_sfh        = gistPipeline.plotting.gistPlot_sfh:main',
            'gistPlot_ls         = gistPipeline.plotting.gistPlot_ls:main'
         ],
      },
      include_package_data=True,
      zip_safe=False)
