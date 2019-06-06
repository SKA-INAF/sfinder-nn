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
from sfindernn.network import NNTester


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

	# - Input options
	parser.add_argument('-model', '--model', dest='model', required=True, type=str,action='store',help='Input file with saved network model')
	parser.add_argument('-inputimg', '--inputimg', dest='inputimg', required=True, type=str,action='store',help='Input image filename')
	
	# - NN loss function
	parser.add_argument('--normalize_inputs', dest='normalize_inputs', action='store_true',help='Normalize input data before training')	
	parser.set_defaults(normalize_inputs=False)
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in (0,1) range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in (0,1) range (default=10 Jy/beam)')
	parser.add_argument('-nmaxobjects', '--nmaxobjects', dest='nmaxobjects', required=False, type=int, default=5, action='store',help='Max number of predicted objects in target (default=5)')
	parser.add_argument('-ntargetpars', '--ntargetpars', dest='ntargetpars', required=False, type=int, default=6, action='store',help='Nmber of pars per objects in target (default=6)')		
	parser.add_argument('-pars_loss_weight', '--pars_loss_weight', dest='pars_loss_weight', required=False, type=float, default=1, action='store',help='Loss weight to be given to source pars learning (default=1)')
	parser.add_argument('-labels_loss_weight', '--labels_loss_weight', dest='labels_loss_weight', required=False, type=float, default=1, action='store',help='Loss weight to be given to source labels learning (default=1)')
	parser.add_argument('--no-classification', dest='no_classification', action='store_true',help='Disable learning of input labels in training')	
	parser.set_defaults(no_classification=False)
	parser.add_argument('--no-regression', dest='no_regression', action='store_true',help='Disable learning of input pars in training')	
	parser.set_defaults(no_regression=False)


	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	model_file= args.model
	inputimg= args.inputimg
	nmaxobjects= args.nmaxobjects
	ntargetpars= args.ntargetpars
	pars_loss_weight= args.pars_loss_weight
	labels_loss_weight= args.labels_loss_weight

	normalize_inputs= args.normalize_inputs
	normdatamin= args.normdatamin
	normdatamax= args.normdatamax

	learn_labels= True
	learn_pars= True
	if args.no_classification:
		learn_labels= False
	if args.no_regression:
		learn_pars= False

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Testing NN on input image %s ..." % inputimg)
	nn= NNTester(
		model_filename=model_file,
		nobjs=nmaxobjects,
		pars=ntargetpars
	)

	# - Set pars
	nn.enable_labels_learning(learn_labels)
	nn.enable_pars_learning(learn_pars)
	nn.set_pars_loss_weight(pars_loss_weight)
	nn.set_labels_loss_weight(labels_loss_weight)
	nn.enable_inputs_normalization(normalize_inputs)
	nn.set_input_data_norm_range(normdatamin,normdatamax)
	

	status= nn.test(inputimg)
	if status<0:
		logger.error("NN testing failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())


