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
from sfindernn.network import NNTrainer


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
	parser.add_argument('-filelist_bkg', '--filelist_bkg', dest='filelist_bkg', required=False, type=str,action='store',help='List of files with bkg train images')
	parser.add_argument('-filelist_source', '--filelist_source', dest='filelist_source', required=False, type=str,action='store',help='List of files with source train images')	
	parser.add_argument('-filelist_sourcepars', '--filelist_sourcepars', dest='filelist_sourcepars', required=False, type=str,action='store',help='List of files with source target pars')
	parser.add_argument('-nnarcfile', '--nnarcfile', dest='nnarcfile', required=False, type=str,action='store',help='Name of file with NN architecture')	
	
	# - Training data
	parser.add_argument('--normalize_targets', dest='normalize_targets', action='store_true',help='Normalize target data before training')	
	parser.set_defaults(normalize_targets=False)
	parser.add_argument('--normalize_inputs', dest='normalize_inputs', action='store_true',help='Normalize input data before training')	
	parser.set_defaults(normalize_inputs=False)
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in (0,1) range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in (0,1) range (default=10 Jy/beam)')
	parser.add_argument('-nmaxobjects', '--nmaxobjects', dest='nmaxobjects', required=False, type=int, default=5, action='store',help='Max number of predicted objects in target (default=5)')
	parser.add_argument('-ntargetpars', '--ntargetpars', dest='ntargetpars', required=False, type=int, default=6, action='store',help='Nmber of pars per objects in target (default=6)')	
	parser.add_argument('-test_size', '--test_size', dest='test_size', required=False, type=float, default=0.2, action='store',help='Fraction of input data used for testing the network (default=0.2)')
	
	# - Network training options
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=1.e-4, action='store',help='Learning rate (default=1.e-4)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-pars_loss_weight', '--pars_loss_weight', dest='pars_loss_weight', required=False, type=float, default=1, action='store',help='Loss weight to be given to source pars learning (default=1)')
	parser.add_argument('-labels_loss_weight', '--labels_loss_weight', dest='labels_loss_weight', required=False, type=float, default=1, action='store',help='Loss weight to be given to source labels learning (default=1)')
	parser.add_argument('--flip_train', dest='flip_train', action='store_true',help='Flip object train input data during training according to best MSE match with targets')	
	parser.set_defaults(flip_train=False)
	parser.add_argument('--flip_test', dest='flip_test', action='store_true',help='Flip object test input data during training according to best MSE match with targets')	
	parser.set_defaults(flip_test=False)
	parser.add_argument('--no-classification', dest='no_classification', action='store_true',help='Disable learning of input labels in training')	
	parser.set_defaults(no_classification=False)
	parser.add_argument('--no-regression', dest='no_regression', action='store_true',help='Disable learning of input pars in training')	
	parser.set_defaults(no_regression=False)


	# - Output options
	parser.add_argument('-outfile_loss', '--outfile_loss', dest='outfile_loss', required=False, type=str, default='nn_loss.png', action='store',help='Name of NN loss plot file (default=nn_loss.png)')
	parser.add_argument('-outfile_accuracy', '--outfile_accuracy', dest='outfile_accuracy', required=False, type=str, default='nn_accuracy.png', action='store',help='Name of NN accuracy plot file (default=nn_accuracy.png)')
	parser.add_argument('-outfile_model', '--outfile_model', dest='outfile_model', required=False, type=str, default='nn_model.png', action='store',help='Name of NN model plot file (default=nn_model.png)')
	parser.add_argument('-outfile_posaccuracy', '--outfile_posaccuracy', dest='outfile_posaccuracy', required=False, type=str, default='nn_posaccuracy.png', action='store',help='Name of NN source position accuracy plot file (default=nn_posaccuracy.png)')
	parser.add_argument('-outfile_fluxaccuracy', '--outfile_fluxaccuracy', dest='outfile_fluxaccuracy', required=False, type=str, default='nn_fluxaccuracy.png', action='store',help='Name of NN source flux accuracy plot file (default=nn_fluxaccuracy.png)')
	parser.add_argument('-outfile_nnout_train', '--outfile_nnout_train', dest='outfile_nnout_train', required=False, type=str, default='train_nnout.dat', action='store',help='Name of output file with NN output for train data (default=train_nnout.dat)')
	parser.add_argument('-outfile_nnout_test', '--outfile_nnout_test', dest='outfile_nnout_test', required=False, type=str, default='test_nnout.dat', action='store',help='Name of output file with NN output for test data (default=test_nnout.dat)')
	parser.add_argument('-outfile_nnout_metrics', '--outfile_nnout_metrics', dest='outfile_nnout_metrics', required=False, type=str, default='nnout_metrics.dat', action='store',help='Name of output file with NN train metrics (default=nnout_metrics.dat)')


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

	# - Train data options
	normalize_targets= args.normalize_targets
	normalize_inputs= args.normalize_inputs
	normdatamin= args.normdatamin
	normdatamax= args.normdatamax
	test_size= args.test_size

	# - Train options
	nmaxobjects= args.nmaxobjects
	ntargetpars= args.ntargetpars
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	pars_loss_weight= args.pars_loss_weight
	labels_loss_weight= args.labels_loss_weight
	nepochs= args.nepochs
	flip_train= args.flip_train
	flip_test= args.flip_test
	learn_labels= True
	learn_pars= True
	if args.no_classification:
		learn_labels= False
	if args.no_regression:
		learn_pars= False
	
	# - Output file
	outfile_loss= args.outfile_loss
	outfile_accuracy= args.outfile_accuracy
	outfile_model= args.outfile_model
	outfile_posaccuracy= args.outfile_posaccuracy
	outfile_fluxaccuracy= args.outfile_fluxaccuracy
	outfile_nnout_train= args.outfile_nnout_train
	outfile_nnout_test= args.outfile_nnout_test
	outfile_nnout_metrics= args.outfile_nnout_metrics
	
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

	dp.enable_inputs_normalization(normalize_inputs)
	dp.set_input_data_norm_range(normdatamin,normdatamax)
	dp.enable_targets_normalization(normalize_targets)
	dp.set_test_sample_size(test_size)
	
	status= dp.read_train_data()
	if status<0:
		logger.error("Failed to read training data!")
		return 1
	

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Running NN training ...")
	nn= NNTrainer(nnarcfile,dp)

	nn.set_optimizer(optimizer)
	nn.set_learning_rate(learning_rate)	
	nn.set_batch_size(batch_size)
	nn.set_pars_loss_weight(pars_loss_weight)
	nn.set_labels_loss_weight(labels_loss_weight)
	nn.set_nepochs(nepochs)
	nn.enable_train_data_flip(flip_train)
	nn.enable_test_data_flip(flip_test)
	nn.enable_labels_learning(learn_labels)
	nn.enable_pars_learning(learn_pars)
	
	nn.set_outfile_loss(outfile_loss)
	nn.set_outfile_accuracy(outfile_accuracy)	
	nn.set_outfile_model(outfile_model)
	nn.set_outfile_posaccuracy(outfile_posaccuracy)
	nn.set_outfile_fluxaccuracy(outfile_fluxaccuracy)	
	nn.set_outfile_nnout_train(outfile_nnout_train)
	nn.set_outfile_nnout_test(outfile_nnout_test)
	nn.set_outfile_nnout_metrics(outfile_nnout_metrics)

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

