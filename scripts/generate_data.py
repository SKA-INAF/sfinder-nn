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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sfindernn import data_generator


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

	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=101, action='store',help='Image width in pixels (default=101)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=101, action='store',help='Image height in pixels (default=101)')	
	parser.add_argument('-nsamples_source', '--nsamples_source', dest='nsamples_source', required=False, type=int, default=-1, action='store',help='Number of train images extracted around sources from input maps (default=-1)')	
	parser.add_argument('-nsamples_bkg', '--nsamples_bkg', dest='nsamples_bkg', required=False, type=int, default=10, action='store',help='Number of train images for bkg extracted from input maps (default=10)')
	
	## - Bkg generation
	parser.add_argument('-inputimg', '--inputimg', dest='inputimg', required=False, type=str,action='store',help='Mosaic residual image from which to extract train data')
	parser.add_argument('--generate_bkg', dest='generate_bkg', action='store_true',help='Generate bkg instead of reading it from input image')	
	parser.set_defaults(generate_bkg=False)	
	parser.add_argument('-marginx', '--marginx', dest='marginx', required=False, type=int, default=0,action='store',help='Input image x margin in pixels used in source generation')
	parser.add_argument('-marginy', '--marginy', dest='marginy', required=False, type=int, default=0,action='store',help='Input image y margin in pixels used in source generation')
	parser.add_argument('-bkg_rms', '--bkg_rms', dest='bkg_rms', required=False, type=float, default=300.e-6, action='store',help='Generated bkg rms (default=300 muJy/beam)')
	parser.add_argument('-bkg_mean', '--bkg_mean', dest='bkg_mean', required=False, type=float, default=0, action='store',help='Generated bkg average (default=0 muJy/beam)')
	
	# - Source generation	
	parser.add_argument('-marginx_source', '--marginx_source', dest='marginx_source', required=False, type=int, default=2,action='store',help='Train image x margin in pixels used in source generation')
	parser.add_argument('-marginy_source', '--marginy_source', dest='marginy_source', required=False, type=int, default=2,action='store',help='Train image y margin in pixels used in source generation')
	parser.add_argument('-Smin', '--Smin', dest='Smin', required=False, type=float, default=1.e-6, action='store',help='Minimum source flux in Jy (default=1.e-6)')
	parser.add_argument('-Smax', '--Smax', dest='Smax', required=False, type=float, default=1, action='store',help='Maximum source flux in Jy (default=1)')
	parser.add_argument('-Smodel', '--Smodel', dest='Smodel', required=False, type=str, default='uniform', action='store',help='Source flux generation model (default=uniform)')
	parser.add_argument('-Sslope', '--Sslope', dest='Sslope', required=False, type=float, default=1.6, action='store',help='Slope par in expo source flux generation model (default=1.6)')
	parser.add_argument('-bmaj_min', '--bmaj_min', dest='bmaj_min', required=False, type=float, default=4, action='store',help='Gaussian components min bmaj in arcsec (default=4)')
	parser.add_argument('-bmaj_max', '--bmaj_max', dest='bmaj_max', required=False, type=float, default=10, action='store',help='Gaussian components max bmaj in arcsec (default=10)')
	parser.add_argument('-bmin_min', '--bmin_min', dest='bmin_min', required=False, type=float, default=4, action='store',help='Gaussian components  min bmin in arcsec (default=4)')
	parser.add_argument('-bmin_max', '--bmin_max', dest='bmin_max', required=False, type=float, default=10, action='store',help='Gaussian components  max bmin in arcsec (default=10)')
	parser.add_argument('-pa_min', '--pa_min', dest='pa_min', required=False, type=float, default=-90, action='store',help='Gaussian components  min position angle in deg (default=0)')
	parser.add_argument('-pa_max', '--pa_max', dest='pa_max', required=False, type=float, default=90, action='store',help='Gaussian components  max position angle in deg (default=180)')
	parser.add_argument('-nsources_max', '--nsources_max', dest='nsources_max', required=False, type=int, default=5, action='store',help='Maximum number of sources generated per crop image (default=5)')
	
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
	print("INFO: Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		print("ERROR: Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Data samples
	nx= args.nx
	ny= args.ny
	nsamples_bkg= args.nsamples_bkg
	nsamples_source= args.nsamples_source

	# - Bkg generation options
	inputimg= args.inputimg
	marginX= args.marginx
	marginY= args.marginy
	generate_bkg_from_img= True
	if args.generate_bkg:
		generate_bkg_from_img= False
	bkg_rms= args.bkg_rms
	bkg_mean= args.bkg_mean

	# - Source generation options
	marginX_source= args.marginx_source
	marginY_source= args.marginy_source
	Smin= args.Smin
	Smax= args.Smax
	Smodel= args.Smodel
	Sslope= args.Sslope
	nsources_max= args.nsources_max
	bmaj_min= args.bmaj_min
	bmaj_max= args.bmaj_max
	bmin_min= args.bmin_min
	bmin_max= args.bmin_max
	pa_min= args.pa_min
	pa_max= args.pa_max	
	
	
	#===========================
	#==   CHECK ARGS
	#===========================
	# - Check if input file is needed and not given
	if generate_bkg_from_img and not inputimg:
		logger.error("Missing input file argument (needed to generate train data)!")
		return 1

	#===========================
	#==   RUN DATA GENERATION
	#===========================
	logger.info("Creating and running data generator ...")
	dg= DataGenerator()

	dg.set_img_filename(inputimg)
	dg.set_margins(marginX,marginY)
	dg.set_source_margins(marginX_source,marginY_source)
	dg.set_nsources_max(nsources_max)
	dg.set_source_flux_range(Smin,Smax)
	dg.set_source_flux_rand_model(Smodel)
	dg.set_source_flux_rand_exp_slope(Sslope)
	dg.set_beam_bmaj_range(bmaj_min,bmaj_max)
	dg.set_beam_bmin_range(bmin_min,bmin_max)
	dg.set_beam_pa_range(pa_min,pa_max)
	dg.enable_bkg_generation_from_img(generate_bkg_from_img)
	dg.set_gen_bkg_rms(bkg_rms)
	dg.set_gen_bkg_mean(bkg_mean)
	dg.set_bkg_sample_size(nsamples_bkg)
	dg.set_source_sample_size(nsamples_source)
	dg.set_train_img_size(nx,ny)
	
	dg.generate_train_data()


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

