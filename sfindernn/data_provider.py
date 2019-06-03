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

## ADDON ML MODULES
from sklearn.model_selection import train_test_split

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class DataProvider(object):
	""" Class to read train data from disk and provide to network

			Arguments:
				- nobjs: Maximum number of source objects to be considered in train images
				- npars: Number of source parameters to be learnt in training
	"""
	
	
	def __init__(self,filelist_bkg,filelist_source,filelist_sourcepars,nobjs=5,npars=6):
		""" Return a DataProvider object """

		# - Input filelist
		self.img_bkg_filelist= filelist_bkg
		self.img_source_filelist= filelist_source
		self.sourcepars_filelist= filelist_sourcepars
		self.read_bkg= True
		if not self.img_bkg_filelist:
			self.read_bkg= False

		# - Input data 
		self.nx= 0
		self.ny= 0	
		self.inputs_bkg= None
		self.inputs_source= None
		self.inputs_train= None
		self.inputs_test= None		
		
		# - Target data and size
		self.nobjects= nobjs
		self.npars= npars
		self.test_size= 0.2
		self.outputs_bkg= None
		self.outputs_labels_bkg= None
		self.outputs_source= None
		self.outputs_labels_source= None	
		self.outputs_train= None
		self.outputs_test= None
		self.outputs_labels_train= None
		self.outputs_labels_test= None
		
		# - Input data normalization
		self.normalize_inputs= True
		self.normmin= 0.001
		self.normmax= 10
		
		# - Target data normalization
		self.normalize_targets= False
		self.theta_min= -90
		self.theta_max= 90
		self.sigma_min= 0
		self.sigma_max= 20
		self.normmin_pars= np.array([0,0,self.normmin,self.sigma_min,self.sigma_min,np.radians(self.theta_min)])
		self.normmax_pars= np.array([self.nx,self.ny,self.normmax,self.sigma_max,self.sigma_max,np.radians(self.theta_max)])

	#################################
	##     SETTERS/GETTERS
	#################################
	def enable_inputs_normalization(self,choice):
		""" Turn on/off inputs normalization """
		self.normalize_inputs= choice

	def set_input_data_norm_range(self,datamin,datamax):
		""" Set input data normalization range """
		self.normmin= datamin
		self.normmax= datamax

	def enable_targets_normalization(self,choice):
		""" Turn on/off target normalization """
		self.normalize_targets= choice
		
	def set_nobjects(self,n):
		""" Set maximum number of detected object in image """
		self.nobjects= n

	def set_npars(self,n):
		""" Set number of source parameters to be fitted in model """
		self.npars= n

	def set_test_sample_size(self,f):
		""" Set test sample proportion """
		self.test_size= f

	def get_img_size(self):
		""" Return the train image size """
		return self.nx, self.ny

	def get_input_data(self):
		""" Return the train & test input data """
		return self.inputs_train, self.inputs_test

	def get_target_data(self):
		""" Return the train & test target data """
		return self.outputs_train, self.outputs_test

	def get_target_label_data(self):
		""" Return the train & test target label data """
		return self.outputs_labels_train, self.outputs_labels_test

	def get_nobjects(self):
		""" Return the number of objects to be used in training """
		return self.nobjects

	def get_npars(self):
		""" Return the number of source pars to be used in training """
		return self.npars
	
	#################################
	##     READ BKG TRAIN DATA
	#################################
	def __read_bkg_train_data(self,filelist):
		""" Read background train data """
		
		# - Skip if filelist is empty or None
		if not filelist:
			logger.warn('Bkg filelist was not given as input.')
			return 0

		# - Init data
		input_data= []	
		output_size= self.nobjects*self.npars
		output_data= []  
		output_label_size= self.nobjects
		output_label_data= []
		nchannels= 1
		filelist_data= []

		# - Read list with files		
		try:
			filelist_data= Utils.read_ascii(filelist,['#'])
		except IOError:
			errmsg= 'Cannot read file: ' + filelist
			logger.error(errmsg)
			return -1

		# - Read image files in list	
		imgcounter= 0
		for item in filelist_data:
			imgcounter+= 1
			filename= item[0]
			logger.info("Reading file %s ..." % filename) 

			data= None
			try:
				data, header= Utils.read_fits(filename)
			except Exception as ex:
				errmsg= 'Failed to read bkg image data (err=' + str(ex) + ')'
				logger.error(errmsg)
				return -1
	
			imgsize= np.shape(data)
			nx= imgsize[1]
			ny= imgsize[0]	
			
			logger.debug("Bkg image no. %d has size (%d,%d)" % (imgcounter,nx,ny) )	

			# - Check bkg image size is equal in all bkg images
			#if imgcounter>1 and (nx!=self.nx or ny!=self.ny):
			#	errmsg= 'Bkg image no. ' + str(imgcounter) + ' has different size wrt previous bkg images!'
			#	logger.error(errmsg)
			#	return -1

			#self.nx= nx
			#self.ny= ny	

			# - Check bkg image size is equal to desired train image
			if nx!=self.nx or ny!=self.ny:
				errmsg= 'Bkg image no. ' + str(imgcounter) + ' has size different from source images!'
				logger.error(errmsg)
				return -1

			# - Set train data as a tensor of size [Nsamples,Nx,Ny,Nchan] Nchan=1
			data= data.reshape(imgsize[0],imgsize[1],nchannels)
			input_data.append(data)

			# - Set train target & labels
			output_data.append( np.zeros((1,output_size)) )
			output_label_data.append( np.zeros((1,output_label_size)) )

		#- Convert list to array
		self.inputs_bkg= np.array(input_data)
		self.inputs_bkg= self.inputs_bkg.astype('float32')
	
		self.outputs_bkg= np.array(output_data)
		self.outputs_bkg= self.outputs_bkg.astype('float32')

		outputs_shape= self.outputs_bkg.shape
		N= outputs_shape[0]
		self.outputs_bkg= self.outputs_bkg.reshape((N,output_size))

		self.outputs_labels_bkg= np.array(output_label_data)
		self.outputs_labels_bkg= self.outputs_labels_bkg.astype('float32')
		self.outputs_labels_bkg= self.outputs_labels_bkg.reshape((N,output_label_size))

		# - Normalize to [0,1]
		if self.normalize_inputs:
			logger.debug("inputs_bkg (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_bkg)),str(np.max(self.inputs_bkg))))
			self.inputs_bkg= (self.inputs_bkg - self.normmin)/(self.normmax-self.normmin)
			logger.debug("inputs_bkg (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_bkg)),str(np.max(self.inputs_bkg))))
		

		logger.debug("outputs_bkg: min/max=%s/%s" % (str(np.min(self.outputs_bkg)),str(np.max(self.outputs_bkg))))
		logger.debug("inputs_bkg size=", np.shape(self.inputs_bkg))
		logger.debug("outputs_bkg size=", np.shape(self.outputs_bkg))
		logger.debug("outputs_labels_bkg size=", np.shape(self.outputs_labels_bkg))
		logger.debug("outputs_bkg=",self.outputs_bkg)
		logger.debug("outputs_labels_bkg=",self.outputs_labels_bkg)

		return 0

	#################################
	##     READ SOURCE TRAIN DATA
	#################################
	def __read_source_train_data(self,filelist,filelist_pars):
		""" Read source train data """
				
		# - Init data
		input_data= []
		output_size= self.nobjects*self.npars
		output_data= []  
		output_label_size= self.nobjects
		output_label_data= []
		nchannels= 1
		filelist_data= []
		filelist_pars_data= []

		# - Set target normalization pars
		self.normmin_pars= np.array([0,0,self.normmin,self.sigma_min,self.sigma_min,np.radians(self.theta_min)])
		self.normmax_pars= np.array([self.nx,self.ny,self.normmax,self.sigma_max,self.sigma_max,np.radians(self.theta_max)])

		# - Read list with image files		
		try:
			filelist_data= Utils.read_ascii(filelist,['#'])
		except IOError:
			errmsg= 'Cannot read file: ' + filelist
			logger.error(errmsg)
			return -1

		# - Read list with source pars files		
		try:
			filelist_pars_data= Utils.read_ascii(filelist_pars,['#'])
		except IOError:
			errmsg= 'Cannot read file: ' + filelist_pars
			logger.error(errmsg)
			return -1

		# - Check lists have the same number of entries
		if len(filelist_data)!=len(filelist_pars_data):
			logger.error("Source img and pars filelist have different number of entries (%s!=%s)" % (len(filelist_data),len(filelist_pars_data)))
			return -1

		# - Read source images & pars
		imgcounter= 0

		for item, item_pars in zip(filelist_data,filelist_pars_data):
			imgcounter+= 1
			filename= item[0]
			filename_pars= item_pars[0]
			logger.info("Reading files: %s, %s ..." % (filename,filename_pars) )

			# - Read source img
			try:
				data, header= Utils.read_fits(filename)
			except Exception as ex:
				errmsg= 'Failed to read source image data (err=' + str(ex) + ')'
				logger.error(errmsg)
				return -1
	
			imgsize= np.shape(data)
			nx= imgsize[1]
			ny= imgsize[0]
			logger.debug("Source image no. %d has size (%d,%d)" % (imgcounter,nx,ny) )

			# - Check bkg image size is equal in all bkg images
			if imgcounter>1 and (nx!=self.nx or ny!=self.ny):
				errmsg= 'Source image no. ' + str(imgcounter) + ' has different size wrt previous source images!'
				logger.error(errmsg)
				return -1

			# - Check source image size is equal to desired train image
			#if nx!=self.nx or ny!=self.ny:
			#	errmsg= 'Source image no. ' + str(imgcounter) + ' has size different from bkg images!'
			#	logger.error(errmsg)
			#	return -1
		
			self.nx= nx
			self.ny= ny

			# - Set train data as a tensor of size [Nsamples,Nx,Ny,Nchan] Nchan=1
			data= data.reshape(imgsize[0],imgsize[1],nchannels)
			input_data.append(data)

			# - Read source pars
			source_pars= []
			skip_patterns= ['#']
			try:
				source_pars= Utils.read_ascii(filename_pars,skip_patterns)
			except IOError:
				errmsg= 'Cannot read file: ' + filename_spar
				logger.error(errmsg)
				return -1

			source_pars_size= np.shape(source_pars)
			nsources= source_pars_size[0]
			npars= source_pars_size[1]

			# - Check source pars number is >= desired pars
			if npars<self.npars:
				logger.error("Source pars read from file no. %s (%d) smaller than desired number of source pars (%d)" % (str(imgcounter),npars,self.npars) )
				return -1

			# - Set train targets
			targets= np.zeros((1,output_size))
			target_labels= np.zeros((1,output_label_size))
			par_counter= 0

			for k in range(nsources):
				target_labels[0,k]= 1
				for l in range(self.npars):			
					targets[0,par_counter+l]= source_pars[k][1+l]	
				par_counter+= self.npars

			output_data.append(targets)
			output_label_data.append(target_labels)


		#- Convert list to array
		self.inputs_source= np.array(input_data)
		self.inputs_source= self.inputs_source.astype('float32')

		self.outputs_source= np.array(output_data)
		self.outputs_source= self.outputs_source.astype('float32')

		outputs_shape= self.outputs_source.shape
		N= outputs_shape[0]
		self.outputs_source= self.outputs_source.reshape((N,output_size))

		self.outputs_labels_source= np.array(output_label_data)
		self.outputs_labels_source= self.outputs_labels_source.astype('float32')
		self.outputs_labels_source= self.outputs_labels_source.reshape((N,output_label_size))

		# - Normalize to [0,1]
		if self.normalize_inputs:
			logger.debug("inputs_source (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_source)),str(np.max(self.inputs_source))))
			self.inputs_source= (self.inputs_source - self.normmin)/(self.normmax-self.normmin)
			logger.debug("inputs_source (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_source)),str(np.max(self.inputs_source))))
			
		# - Normalize targets to [0,1]
		if self.normalize_targets:
			
			targets_normmin= np.zeros(self.nobjects*self.npars)
			targets_normmax= np.zeros(self.nobjects*self.npars)
			par_counter= 0
			for k in range(self.nobjects):
				for l in range(self.npars):
					targets_normmin[par_counter]= self.normmin_pars[l]
					targets_normmax[par_counter]= self.normmax_pars[l]
					par_counter+= 1

			logger.debug("targets_normmin=", targets_normmin)
			logger.debug("targets_normmax=", targets_normmax)
			logger.debug("outputs_source (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.outputs_source)),str(np.max(self.outputs_source))))
			self.outputs_source= (self.outputs_source - targets_normmin)/(targets_normmax-targets_normmin)
			logger.debug("outputs_source (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.outputs_source)),str(np.max(self.outputs_source))))

		logger.debug("inputs_source size=", np.shape(self.inputs_source))
		logger.debug("outputs_source size=", np.shape(self.outputs_source))
		logger.debug("outputs_labels_source size=", np.shape(self.outputs_labels_source))
		logger.debug("outputs_source=",self.outputs_source)
		logger.debug("outputs_labels_source=",self.outputs_labels_source)

		return 0

	#############################
	##     READ TRAIN DATA
	#############################
	def read_train_data(self):	
		""" Read train data from disk using input filelists """
				
		# - Read train data for source
		logger.info("Reading train data for source ...")
		status= self.__read_source_train_data(self.img_source_filelist,self.sourcepars_filelist)
		if status<0:
			return -1

		# - Read train data for bkg
		if self.read_bkg:
			logger.info("Reading train data for bkg ...")
			status= self.__read_bkg_train_data(self.img_bkg_filelist)
			if status<0:
				return -1
		else:
			logger.warn("No bkg data will be read as no filelist was given in input")

		# - Merge data for bkg & sources
		if self.read_bkg:
			logger.info("Merging train data for bkg & sources ...")
			inputs= np.concatenate((self.inputs_bkg,self.inputs_source))
			outputs= np.concatenate((self.outputs_bkg,self.outputs_source))
			outputs_labels= np.concatenate((self.outputs_labels_bkg,self.outputs_labels_source))
		else:
			logger.info("Setting train data to source data only ...")
			inputs= self.inputs_source
			outputs= self.outputs_source
			outputs_labels= self.outputs_labels_source

		# - Shuffle data before splitting in test & validation sample
		logger.info("Shuffling train data ...")
		indices= np.arange(inputs.shape[0])
		np.random.shuffle(indices)
		inputs= inputs[indices]
		outputs= outputs[indices]
		outputs_labels= outputs_labels[indices]
	
		logger.debug("inputs size=", np.shape(inputs))
		logger.debug("outputs size=", np.shape(outputs))
		logger.debug("outputs_labels size=", np.shape(outputs_labels))

		# - Partition the data into training and cross-validation splits
		logger.info("Splitting data into train & test samples ...")
		split= train_test_split(
			inputs,outputs,outputs_labels, 
			test_size=self.test_size, 
			random_state=None
		)
		(self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test, self.outputs_labels_train, self.outputs_labels_test) = split

		logger.debug("inputs_train size=", np.shape(self.inputs_train))
		logger.debug("inputs_test size=", np.shape(self.inputs_test))
		logger.debug("outputs_train size=", np.shape(self.outputs_train))
		logger.debug("outputs_test size=", np.shape(self.outputs_test))
		logger.debug("outputs_labels_train size=", np.shape(self.outputs_labels_train))
		logger.debug("outputs_labels_test size=", np.shape(self.outputs_labels_test))

		return 0

	def draw_train_inputs(self,nbins=30,logscale=True):
		""" Draw input train data """
		Utils.draw_histo(self.inputs_train,nbins,logscale)

	def draw_test_inputs(self,nbins=30,logscale=True):
		""" Draw input test data """
		Utils.draw_histo(self.inputs_test,nbins,logscale)

	def draw_train_outputs(self,nbins=30,logscale=True):
		""" Draw output train data """
		Utils.draw_histo(self.outputs_train,nbins,logscale)

	def draw_test_outputs(self,nbins=30,logscale=True):
		""" Draw output train data """
		Utils.draw_histo(self.outputs_test,nbins,logscale)
		

