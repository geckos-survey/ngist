from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README') as file:
        return(file.read())

setup(name='gistPipeline',
      version='2.1',
      description='A Multi-Purpose IFS Analysis Pipeline',
      long_description=readme(),
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='abittner.gitlab.io/thegistpipeline/',
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
        'vorbin==3.1.3',
        'ppxf==6.7.15',
        'plotbin==3.1.3',
      ],
      python_requires='==3.6',
      entry_points={
        'console_scripts': [
            'gistPipeline        = gistPipeline.MainPipeline:main'
            ],
        'gui_scripts': [
            'Mapviewer           = gistPipeline.Mapviewer:main',
            'gistPlot_kinematics = gistPipeline.gistModules.gistPlot_kinematics:main',
            'gistPlot_lambdar    = gistPipeline.gistModules.gistPlot_lambdar:main',
            'gistPlot_gandalf    = gistPipeline.gistModules.gistPlot_gandalf:main',
            'gistPlot_spp        = gistPipeline.gistModules.gistPlot_spp:main',
            'gistPlot_ls         = gistPipeline.gistModules.gistPlot_ls:main'
         ],
      },
      include_package_data=True,
      zip_safe=False)
