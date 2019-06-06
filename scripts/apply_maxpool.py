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

## GRAPHICS MODULES
import matplotlib.pyplot as plt

## KERAS MODULES
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

## MODULES
from sfindernn import __version__, __date__
from sfindernn import logger
from sfindernn.utils import Utils

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
	parser.add_argument('-inputimg', '--inputimg', dest='inputimg', required=True, type=str,action='store',help='Input image filename')
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in (0,1) range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in (0,1) range (default=10 Jy/beam)')	
	parser.add_argument('--normalize_inputs', dest='normalize_inputs', action='store_true')	
	parser.set_defaults(normalize_inputs=False)	

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

	
	# - Input data
	input_img= args.inputimg
	normmin= args.normdatamin
	normmax= args.normdatamax
	normalize_inputs= args.normalize_inputs

	#===========================
	#==   READ IMAGE
	#===========================
	logger.info("Reading input image %s ..." % input_img)
	data= None
	try:
		data, header= Utils.read_fits(input_img)
	except Exception as ex:
		errmsg= 'Failed to read input image data (err=' + str(ex) + ')'
		logger.error(errmsg)
		return -1
	
	imgsize= np.shape(data)
	nx= imgsize[1]
	ny= imgsize[0]	
			
	logger.info("Image has size (%d,%d)" % (nx,ny) )	

	#===========================
	#==   SET DATA
	#===========================
	# - Normalize data
	if normalize_inputs:
		logger.info("Normalizing inputs in range [%s,%s] ..." % (normmin,normmax))
		data= (data - normmin)/(normmax-normmin)

	# - Set input image tensor
	logger.info("Converting input data from numpy array to tensor ...")
	img_tensor = image.img_to_array(data)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	print(img_tensor.shape)

	#===========================
	#==   APPLY MAX POOL
	#===========================
	logger.info("Defining network arc ...")
	inputShape = (data.shape[0], data.shape[1], 1)
	inputs= Input(shape=inputShape,dtype='float', name='input')
	outputs= keras.layers.MaxPooling2D(pool_size=7, strides=None, padding='same', data_format=None)(inputs)

	model= Model(
		inputs=inputs,
		outputs=outputs,
		name="Net"
	)

	model.summary()

	logger.info("Compile network model ...")
	sgd= optimizers.SGD(lr=1.e-4)
	model.compile(loss='mean_squared_error', optimizer='sgd')
	

	logger.info("Applying max pool to input image ...")
	data_filt= model.predict(img_tensor)
	print("data_filt shape=", data_filt.shape)
	data_filt= data_filt.reshape(data_filt.shape[1],data_filt.shape[2])
	print("data_filt shape=", data_filt.shape)

	# - Draw input image and filt image
	(fig, ax) = plt.subplots(1,2, figsize=(10,10),squeeze=False)

	ax[0,0].set_title('Input image')
	ax[0,0].set_xlabel("x")
	ax[0,0].set_ylabel("y")
	ax[0,0].imshow(data)

	ax[0,1].set_title('Filtered image')
	ax[0,1].set_xlabel("x")
	ax[0,1].set_ylabel("y")
	ax[0,1].imshow(data_filt)

	plt.show()
	


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
