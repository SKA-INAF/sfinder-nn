#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sfindernn import __version__, __date__
from sfindernn import logger
from sfindernn.data_provider import DataProvider
from sfindernn.network import Network


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	parser.add_argument('-filelist_bkg', '--filelist_bkg', dest='filelist_bkg', required=False, type=str,action='store',help='List of files with bkg train images')
	parser.add_argument('-filelist_source', '--filelist_source', dest='filelist_source', required=False, type=str,action='store',help='List of files with source train images')	
	parser.add_argument('-filelist_sourcepars', '--filelist_sourcepars', dest='filelist_sourcepars', required=False, type=str,action='store',help='List of files with source target pars')
	parser.add_argument('-nnarcfile', '--nnarcfile', dest='nnarcfile', required=False, type=str,action='store',help='Name of file with NN architecture')	
	
	parser.add_argument('-nmaxobjects', '--nmaxobjects', dest='nmaxobjects', required=False, type=int, default=5, action='store',help='Max number of predicted objects in target (default=5)')
	parser.add_argument('-ntargetpars', '--ntargetpars', dest='ntargetpars', required=False, type=int, default=6, action='store',help='Nmber of pars per objects in target (default=6)')	


	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   INITIALIZE LOGGER
	#===========================
	#logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
	#logger= logging.getLogger(__name__)
	#logger.setLevel(logging.INFO)
	#logger.info("This is sfindernn {0}-({1}) data_generator script ".format(__version__, __date__))


	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Input filelist
	filelist_bkg= args.filelist_bkg
	filelist_source= args.filelist_source
	filelist_sourcepars= args.filelist_sourcepars
	nnarcfile= args.nnarcfile

	# - Train options
	nmaxobjects= args.nmaxobjects
	ntargetpars= args.ntargetpars
	
	
	#===========================
	#==   CHECK ARGS
	#===========================
	# ...

	#===========================
	#==   READ DATA
	#===========================
	logger.info("Running data provider ...")
	dp= DataProvider(
		filelist_bkg=filelist_bkg,
		filelist_source=filelist_source,
		filelist_sourcepars=filelist_sourcepars,
		nobjs=nmaxobjects,
		npars=ntargetpars
	)
	
	status= dp.read_train_data()
	if status<0:
		logger.error("Failed to read training data!")
		return 1
	

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Running NN training ...")
	nn= Network(nnarcfile,dp)

	status= nn.train()
	if status<0:
		logger.error("NN training failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

