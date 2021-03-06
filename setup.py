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


#reqs = ['numpy>=1.10',
#        'astropy>=2.0, <3',
#        'keras>=2.0',
#        'tensorflow>=1.13']

PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy>=1.10')
reqs.append('astropy>=2.0, <3')


if PY_MAJOR_VERSION<=2:
	print("PYTHON 2 detected")
	reqs.append('future')
	reqs.append('scipy<=1.2.1')
	reqs.append('scikit-learn<=0.20')
	reqs.append('pyparsing>=2.0.1')
	reqs.append('matplotlib<=2.2.4')
else:
	print("PYTHON 3 detected")
	reqs.append('scipy')
	reqs.append('scikit-learn')
	reqs.append('pyparsing')
	reqs.append('matplotlib')

reqs.append('keras>=2.0')
reqs.append('tensorflow>=1.13')


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
	scripts=['scripts/generate_train_data.py','scripts/train_nn.py','scripts/test_nn.py','scripts/create_filelist.sh','scripts/apply_maxpool.py','test/test_train.sh','test/test_gendata.sh','macros/DrawNNLossVSEpochs.C','macros/DrawNNResults.C'],
	#data_files=[('sfindernn', [os.path.join(data_dir, 'MOC.fits')]) ],
	#setup_requires=['pytest-runner'],
	#tests_require=['pytest', 'nose']
)
