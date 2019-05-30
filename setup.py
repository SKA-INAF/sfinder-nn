#! /usr/bin/env python
"""
Setup for sfinder-nn
"""
import os
import sys
from setuptools import setup


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import sfindernn
	return sfindernn.__version__


reqs = ['numpy>=1.10',
        'astropy>=2.0, <3',
        'matplotlib',
        'keras>=2.0',
        'tensorflow>=1.13',
        'scikit-learn',
        'future']


#if sys.version_info < (2, 7):
#	reqs.append('lmfit==0.9.1')
#else:
#	reqs.append('lmfit>=0.9.2')

data_dir = 'data'

setup(
	name="sfindernn",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Source finder based on Neural Networks",
	license = "GPL3",
	url="https://github.com/SKA-INAF/sfinder-nn",
	long_description=read('README.md'),
	packages=['sfindernn'],
	install_requires=reqs,
	scripts=['scripts/generate_data.py','scripts/train_nn.py','test/test_train.sh','test/test_gendata.sh'],
	#data_files=[('sfindernn', [os.path.join(data_dir, 'MOC.fits')]) ],
	#setup_requires=['pytest-runner'],
	#tests_require=['pytest', 'nose']
)
