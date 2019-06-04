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

## KERAS MODULES
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K
from keras.models import Model
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

## GRAPHICS MODULES
import matplotlib.pyplot as plt

## PACKAGE MODULES
from .utils import Utils
from .data_provider import DataProvider

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class NNTrainer(object):
	""" Class to create and train a neural net

			Arguments:
				- nnarc_filename: File with network architecture to be created
				- 
	"""
	
	def __init__(self,nnarc_filename,data_provider):
		""" Return a Network object """

		# - Input data
		self.nnarc_file= nnarc_filename
		self.dp= data_provider

		# - Train data	
		self.nsamples_train= 0
		self.nsamples_test= 0
		self.nx= 0
		self.ny= 0
		self.nchannels= 1
		self.nobjects= 0
		self.npars= 0
		self.npars_flip= 2
		self.flip_test_data= True
		self.flip_train_data= True
		self.inputs_train= None
		self.inputs_test= None
		self.outputs_train= None
		self.outputs_test= None
		self.outputs_labels_train= None
		self.outputs_labels_test= None

		# - Neural net architecture & train
		self.learn_labels= True
		self.learn_pars= True
		self.model= None
		self.fitsout= None
		self.optimizer= 'rmsprop'
		self.learning_rate= 1.e-4
		self.batch_size= 32
		self.nepochs= 10
		self.pars_loss_weight= 1
		self.labels_loss_weight= 1

		# - Neural Net results
		self.train_type_loss_vs_epoch= None
		self.train_pars_loss_vs_epoch= None
		self.train_loss_vs_epoch= None
		self.test_type_loss_vs_epoch= None
		self.test_pars_loss_vs_epoch= None
		self.test_loss_vs_epoch= None
		self.train_accuracy_vs_epoch= None
		self.test_accuracy_vs_epoch= None

		# - Output file options
		self.outfile_loss= 'nn_loss.png'
		self.outfile_accuracy= 'nn_accuracy.png'
		self.outfile_model= 'nn_model.png'
		self.outfile_posaccuracy= 'nn_posaccuracy.png'
		self.outfile_fluxaccuracy= 'nn_fluxaccuracy.png'
		self.outfile_nnout_train= 'train_nnout.dat'
		self.outfile_nnout_test= 'test_nnout.dat'
		self.outfile_nnout_metrics= 'nnout_metrics.dat'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_pars_loss_weight(self,w):
		""" Set source par loss weight """
		self.pars_loss_weight= w

	def set_labels_loss_weight(self,w):
		""" Set source labels loss weight """
		self.labels_loss_weight= w

	def set_nepochs(self,w):
		""" Set number of train epochs """
		self.nepochs= w

	def set_batch_size(self,bs):
		""" Set batch size """
		self.batch_size= bs

	def set_optimizer(self,opt):
		""" Set optimizer """
		self.optimizer= opt

	def set_learning_rate(self,lr):
		""" Set learning rate """
		self.learning_rate= lr

	def enable_labels_learning(self,choice):
		""" Turn on/off learning of labels during training """
		self.learn_labels= choice

	def enable_pars_learning(self,choice):
		""" Turn on/off learning of pars during training """
		self.learn_pars= choice

	def enable_train_data_flip(self,choice):
		""" Turn on/off flipping of train data during training """
		self.flip_train_data= choice

	def enable_test_data_flip(self,choice):
		""" Turn on/off flipping of test data during training """
		self.flip_test_data= choice

	def set_outfile_loss(self,filename):
		""" Set output file name for loss plot """
		self.outfile_loss= filename

	def set_outfile_accuracy(self,filename):
		""" Set output file name for accuracy plot """
		self.outfile_accuracy= filename

	def set_outfile_model(self,filename):
		""" Set output file name for model plot """
		self.outfile_model= filename

	def set_outfile_posaccuracy(self,filename):
		""" Set output file name for pos accuracy plot """
		self.outfile_posaccuracy= filename

	def set_outfile_fluxaccuracy(self,filename):
		""" Set output file name for flux accuracy plot """
		self.outfile_fluxaccuracy= filename

	def set_outfile_nnout_train(self,filename):	
		""" Set output file name where to store NN output for train data"""
		self.outfile_nnout_train= filename

	def set_outfile_nnout_test(self,filename):	
		""" Set output file name where to store NN output for test data"""	
		self.outfile_nnout_test= filename

	def set_outfile_nnout_metrics(self,filename):	
		""" Set output file name where to store NN output metrics"""	
		self.outfile_nnout_metrics= filename

	#####################################
	##     SET TRAIN/TEST DATA
	#####################################
	def __set_data(self):
		""" Set train/test data from provider """

		# - Retrieve input data info from provider
		self.inputs_train, self.inputs_test= self.dp.get_input_data()
		imgshape= self.inputs_train.shape
		imgshape_test= self.inputs_test.shape
			
		# - Retrieve output data info from provider
		self.outputs_train, self.outputs_test= self.dp.get_target_data()
		self.outputs_labels_train, self.outputs_labels_test= self.dp.get_target_label_data()
		self.nobjects= self.dp.get_nobjects() 
		self.npars= self.dp.get_npars()

		# - Check if data provider has data filled
		if self.inputs_train.ndim!=4 or self.inputs_test.ndim!=4:
			logger.error("Invalid number of dimensions in train/test data (4 expected) (hint: check if data was read in provider!)")
			return -1

		# - Set data
		self.nsamples_train= imgshape[0]
		self.nsamples_test= imgshape_test[0]
		self.nx= imgshape[2]
		self.ny= imgshape[1]
		self.nchannels= imgshape[3] 
		
		logger.info("Inputs size (nx,ny,nchan)=(%d,%d,%d)" % (self.nx,self.ny,self.nchannels))
		logger.info("Train/test sample size= %d/%d" % (self.nsamples_train,self.nsamples_test))
		logger.info("Train nobjects=%d" % self.nobjects)
		logger.info("Train npars=%d" % self.npars)


		return 0

	#####################################
	##     NN METRICS
	#####################################
	def __mse_metric(self,y_true,y_pred):
		""" Define MSE metric"""
		
		mse= 0

		if self.learn_labels:

			# - Extract tensors relative to pars
			y_true_type= y_true[:,:self.nobjects]
			y_pred_type= y_pred[:,:self.nobjects]
		
			# - Extract tensors relative to parameters
			y_true_pars= y_true[:,self.nobjects:]
			y_pred_pars= y_pred[:,self.nobjects:]

			# - Replicate true labels to have a tensor of size (N,nobjects*npars)
			#   This is multiplied by pars so that objects with label=0 are not entering in pars MSE computation (they sum zero)
			w= K.repeat_elements(y_true_type,self.npars,axis=1)
		
			# - Count number of objects
			N= tf.count_nonzero(y_true_type,dtype=tf.dtypes.float32)
		
			# - Compute MSE for pars relative to target objects (not background)
			mse= K.sum(K.square( w*(y_pred_pars - y_true_pars) ), axis=-1)/N

		else:
			mse= K.mean(K.square(y_pred - y_true), axis=-1)
		
		return mse

	
	def __classification_metric(self,y_true,y_pred):
		""" Define classification metric"""
		
		binaryCE= 0

		if self.learn_pars:
			# - Extract tensors relative to pars
			y_true_type= y_true[:,:self.nobjects]
			y_pred_type= y_pred[:,:self.nobjects]
		
			# - Compute binary crossentropy for labels
			binaryCE = keras.metrics.binary_crossentropy(y_true_type,y_pred_type)
			
		else:
			binaryCE = keras.metrics.binary_crossentropy(y_true,y_pred)
			
		return binaryCE

	#####################################
	##     NN LOSS FUNCTION FOR PARS
	#####################################
	def __tot_loss(self,y_true,y_pred):
		""" Definition of NN total loss """
		
		tot= 0

		if self.learn_labels:
			
			if self.learn_pars:
				# - Extract tensors relative to pars
				y_true_type= y_true[:,:self.nobjects]
				y_pred_type= y_pred[:,:self.nobjects]
		
				# - Compute binary crossentropy for labels
				binaryCE = keras.metrics.binary_crossentropy(y_true_type,y_pred_type)

				# - Extract tensors relative to parameters
				y_true_pars= y_true[:,self.nobjects:]
				y_pred_pars= y_pred[:,self.nobjects:]
		
				# - Replicate true labels to have a tensor of size (N,nobjects*npars)
				#   This is multiplied by pars so that objects with label=0 are not entering in pars MSE computation (they sum zero)
				w= K.repeat_elements(y_true_type,self.npars,axis=1)
		
				# - Count number of objects
				N= tf.count_nonzero(y_true_type,dtype=tf.dtypes.float32)
		
				# - Compute MSE for pars relative to target objects (not background)
				mse= K.sum(K.square( w*(y_pred_pars - y_true_pars) ), axis=-1)/N
		
				# - Compute total loss as weighted sum of MSE + binaryCE		
				tot= self.pars_loss_weight*mse + self.labels_loss_weight*binaryCE

			else:
				binaryCE = keras.metrics.binary_crossentropy(y_true,y_pred)
				tot= binaryCE

		else:
			mse= K.mean(K.square(y_pred - y_true), axis=-1)
			tot= mse
		
		return tot


	#####################################
	##     BUILD NETWORK FROM SPEC FILE
	#####################################
	def __build_network(self,filename):
		""" Building deep network taking architecture from file """

		# - Read NN architecture file
		nn_data= []
		skip_patterns= ['#']
		try:
			nn_data= Utils.read_ascii(filename,skip_patterns)
		except IOError:
			print("ERROR: Failed to read nn arc file %d!" % filename)
			return -1

		nlayers= np.shape(nn_data)[0]
		
		# - Input layer
		nchan= 1	
		inputShape = (self.ny, self.nx, self.nchannels)
		inputs= Input(shape=inputShape,dtype='float', name='input')
		x= inputs

		# - Parse NN architecture file and create intermediate layers
		for index in range(nlayers):
			layer_info= nn_data[index]
			logger.info("Layer no. %d: %s" % (index,layer_info))

			layer_type= layer_info[0]

			# - Add Conv2D layer?
			if layer_type=='Conv2D':
				nfields= len(layer_info)
				if nfields!=5:
					logger.error("Invalid number of fields (n=%d) given in Conv2D layer specification (5 expected)" % nfields)
					return -1
				nfilters= int(layer_info[1])
				kernSize= int(layer_info[2])
				activation= str(layer_info[3])
				padding= str(layer_info[4])
				x = layers.Conv2D(filters=nfilters, kernel_size=(kernSize,kernSize), activation=activation, padding=padding)(x)			
	
			# - Add MaxPooling2D layer?
			elif layer_type=='MaxPooling2D':
				nfields= len(layer_info)
				if nfields!=3:
					logger.error("Invalid number of fields (n=%d) given in MaxPooling2D layer specification (3 expected)" % nfields)
					return -1
				poolSize= int(layer_info[1])
				padding= str(layer_info[2])
				x = layers.MaxPooling2D(pool_size=(poolSize,poolSize),strides=None,padding=padding)(x)

			# - Add Dropout layer?
			elif layer_type=='Dropout':
				nfields= len(layer_info)
				if nfields!=2:
					logger.error("Invalid number of fields (n=%d) given in Dropout layer specification (2 expected)" % nfields)
					return -1
				dropout= float(layer_info[1])
				x = layers.Dropout(dropout)(x)

			# - Add BatchNormalization layer?
			elif layer_type=='BatchNormalization':
				x = layers.BatchNormalization()(x)
	
			# - Add Flatten layer?
			elif layer_type=='Flatten':
				x = layers.Flatten()(x)

			# - Add Dense layer?
			elif layer_type=='Dense':
				nfields= len(layer_info)
				if nfields!=3:
					logger.error("Invalid number of fields (n=%d) given in Dense layer specification (3 expected)" % nfields)
				nNeurons= int(layer_info[1])
				activation= str(layer_info[2])
				x = layers.Dense(nNeurons, activation=activation)(x)

			else:
				logger.error("Invalid/unknown layer type parsed (%s)!" % layer_type)
				return -1
			
		# - Output layers
		type_prediction = layers.Dense(self.nobjects, activation='sigmoid', name='type')(x)
		pars_prediction = layers.Dense(self.nobjects*self.npars, activation='linear', name='pars')(x)

		# - Concatenate output layers
		if self.learn_labels:
			if self.learn_pars:
				nn_prediction= layers.concatenate([type_prediction,pars_prediction],name='nnout')
			else:
				nn_prediction= layers.Dense(self.nobjects, activation='sigmoid', name='nnout')(x) 
		else:
			if self.learn_pars:
				nn_prediction= layers.Dense(self.nobjects*self.npars, activation='linear', name='nnout')(x)
			else:
				logger.error("You need to select learning of at least one between pars or labels!")
				return -1

		# - Create NN model
		self.model = Model(
				inputs=inputs,
				outputs=nn_prediction,
				name="SourceNet"
		)

		# - Print network architecture
		self.model.summary()

		#- Set optimizer & loss function per each output
		logger.info("Compiling network ...")
		if self.optimizer=='rmsprop':
			opt= optimizers.RMSprop(lr=self.learning_rate)
		elif self.optimizer=='sgd':
			#opt= optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
			opt= optimizers.SGD(lr=self.learning_rate, nesterov=False)
		elif self.optimizer=='sgdn':
			opt= optimizers.SGD(lr=self.learning_rate, nesterov=True)
		else:
			opt= optimizers.RMSprop(lr=self.learning_rate)
		
		#opt= Adam(lr=INIT_LR, decay=INIT_LR / nepochs)
	
		#losses = {
		#	"type": "binary_crossentropy",
		#	"pars": "mse"
		#}
		#lossWeights = {
		#	"type": self.labels_loss_weight,
		#	"pars": self.pars_loss_weight
		#}
		#self.model.compile(optimizer=opt,loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
		
		if self.learn_labels:
			if self.learn_pars:
				self.model.compile(optimizer=opt,loss=self.__tot_loss, metrics=['accuracy',self.__classification_metric,self.__mse_metric])
			else:
				self.model.compile(optimizer=opt,loss=self.__tot_loss, metrics=['accuracy',self.__classification_metric])
		else:
			if self.learn_pars:
				self.model.compile(optimizer=opt,loss=self.__tot_loss, metrics=['accuracy',self.__mse_metric])
			else:
				logger.error("You need to select learning of at least one between pars or labels!")
				return -1

		return 0
	
	###########################
	##     FLIP TRAIN DATA
	###########################
	def __do_train_data_flip(self):
		""" Flip train data during training """
		
		# - Get NN prediction to train data
		nnout_pred= self.model.predict(self.inputs_train)
		outputs_labels_pred= nnout_pred[:,:self.nobjects]
		outputs_pred= nnout_pred[:,self.nobjects:]

		nsamples= outputs_pred.shape[0]
				
		for sample in range(nsamples):
			
			mses= np.zeros((self.nobjects,self.nobjects))
			predout_mse= np.zeros((self.nobjects,self.npars_flip))
			expout_mse= np.zeros((self.nobjects,self.npars_flip))
			predout= np.zeros((self.nobjects,self.npars))
			expout= np.zeros((self.nobjects,self.npars))
			expout_labels= np.zeros((self.nobjects,1))
			
			for i in range(self.nobjects):
				expout_labels[i,0]= self.outputs_labels_train[sample,i]
				for j in range(self.npars):
					predout[i,j]= outputs_pred[sample,j+i*self.npars]
					expout[i,j]= self.outputs_train[sample,j+i*self.npars]
				for j in range(self.npars_flip):
					predout_mse[i,j]= outputs_pred[sample,j+i*self.npars]
					expout_mse[i,j]= self.outputs_train[sample,j+i*self.npars]
					
			for i in range(self.nobjects):
				for j in range(self.nobjects):
					mse= np.mean(np.square(expout_mse[i,:]-predout_mse[j,:]))	
					mses[i,j]= mse
	
			
			# - Find new ordering according to smallest MSE
			mses_copy= mses
			reorder_indexes= np.zeros(self.nobjects,dtype=int)
			for i in range(self.nobjects):
				ind_exp, ind_pred= np.unravel_index(mses.argmin(),mses.shape) # Find index of smallest mse
				mses[ind_exp]= np.Inf # Set mse to largest value so that it is not re-assigned anymore
				mses[:,ind_pred]= np.Inf 
				reorder_indexes[ind_pred]= ind_exp	
				
				# - Flip target
				self.outputs_train[sample]= expout[reorder_indexes].flatten() 
				self.outputs_labels_train[sample]= expout_labels[reorder_indexes].flatten() 
		
	###########################
	##     FLIP TEST DATA
	###########################
	def __do_test_data_flip(self):
		""" Flip test data during training """	
				
		#(outputs_labels_pred, outputs_pred)= self.model.predict(self.inputs_test)
		nnout_pred= self.model.predict(self.inputs_test)
		outputs_labels_pred= nnout_pred[:,:self.nobjects]
		outputs_pred= nnout_pred[:,self.nobjects:]
		nsamples= outputs_pred.shape[0]
		
		for sample in range(nsamples):
			
			mses= np.zeros((self.nobjects,self.nobjects))
			predout_mse= np.zeros((self.nobjects,self.npars_flip))
			expout_mse= np.zeros((self.nobjects,self.npars_flip))
			predout= np.zeros((self.nobjects,self.npars))
			expout= np.zeros((self.nobjects,self.npars))
			expout_labels= np.zeros((self.nobjects,1))
			
			for i in range(self.nobjects):
				expout_labels[i,0]= self.outputs_labels_test[sample,i]
				for j in range(self.npars):
					predout[i,j]= outputs_pred[sample,j+i*self.npars]
					expout[i,j]= self.outputs_test[sample,j+i*self.npars]
				for j in range(self.npars_flip):
					predout_mse[i,j]= outputs_pred[sample,j+i*self.npars]
					expout_mse[i,j]= self.outputs_test[sample,j+i*self.npars]
					
				for i in range(self.nobjects):
					for j in range(self.nobjects):
						mse= np.mean(np.square(expout_mse[i,:]-predout_mse[j,:]))	
						mses[i,j]= mse
	
				# - Find new ordering according to smallest MSE
				mses_copy= mses
				reorder_indexes= np.zeros(self.nobjects,dtype=int)
				for i in range(self.nobjects):
					ind_exp, ind_pred= np.unravel_index(mses.argmin(),mses.shape) # Find index of smallest mse
					mses[ind_exp]= np.Inf # Set mse to largest value so that it is not re-assigned anymore
					mses[:,ind_pred]= np.Inf 
					reorder_indexes[ind_pred]= ind_exp	
				
				#- Flip target
				self.outputs_test[sample]= expout[reorder_indexes].flatten() 
				self.outputs_labels_test[sample]= expout_labels[reorder_indexes].flatten() 


	###########################
	##     TRAIN NETWORK
	###########################
	def __train_network(self):
		""" Train deep network """
	
		logger.info("Start NN training ...")

		# - Initialize train/test loss vs epoch
		self.train_loss_vs_epoch= np.zeros((1,self.nepochs))	
		self.train_type_loss_vs_epoch= np.zeros((1,self.nepochs))	
		self.train_pars_loss_vs_epoch= np.zeros((1,self.nepochs))	
		self.test_loss_vs_epoch= np.zeros((1,self.nepochs))
		self.test_type_loss_vs_epoch= np.zeros((1,self.nepochs))
		self.test_pars_loss_vs_epoch= np.zeros((1,self.nepochs))
		self.train_accuracy_vs_epoch= np.zeros((1,self.nepochs))
		self.test_accuracy_vs_epoch= np.zeros((1,self.nepochs))
		deltaLoss_train= 0
		deltaLoss_test= 0
		deltaAcc_train= 0
		deltaAcc_test= 0

		# - Start training loop
		for epoch in range(self.nepochs):
		
			# - Train for 1 epoch
			type_loss_train= 0
			type_loss_test= 0
			pars_loss_train= 0
			pars_loss_test= 0

			if self.learn_labels:
				if self.learn_pars:

					self.fitout= self.model.fit(
						x=self.inputs_train, 
						y={"nnout": np.concatenate((self.outputs_labels_train,self.outputs_train),axis=1) },
						validation_data=(self.inputs_test,{"nnout": np.concatenate((self.outputs_labels_test,self.outputs_test),axis=1) }),
						epochs=1,
						batch_size=self.batch_size,
						verbose=1
					)
					type_loss_train= self.fitout.history['__classification_metric'][0]
					type_loss_test= self.fitout.history['val___classification_metric'][0]
					pars_loss_train= self.fitout.history['__mse_metric'][0]					
					pars_loss_test= self.fitout.history['val___mse_metric'][0]
					
				else:
					self.fitout= self.model.fit(
						x=self.inputs_train, 
						y={"nnout": self.outputs_labels_train },
						validation_data=(self.inputs_test,{"nnout": self.outputs_labels_test}),
						epochs=1,
						batch_size=self.batch_size,
						verbose=1
					)
					type_loss_train= self.fitout.history['__classification_metric'][0]
					type_loss_test= self.fitout.history['val___classification_metric'][0]
					
			else:
				if self.learn_pars:

					self.fitout= self.model.fit(
						x=self.inputs_train, 
						y={"nnout": self.outputs_train},
						validation_data=(self.inputs_test,{"nnout": self.outputs_test}),
						epochs=1,
						batch_size=self.batch_size,
						verbose=1
					)
					pars_loss_train= self.fitout.history['__mse_metric'][0]
					pars_loss_test= self.fitout.history['val___mse_metric'][0]
			
				else:
					logger.error("You need to select learning of at least one between pars or labels!")
					return -1

			# - Save epoch loss
			#print (self.fitout.history)
			
			accuracy_train= self.fitout.history['acc'][0]
			accuracy_test= self.fitout.history['val_acc'][0]
			loss_train= self.fitout.history['loss'][0]
			loss_test= self.fitout.history['val_loss'][0]
			
			if epoch>=1:
				deltaLoss_train= (loss_train/self.train_loss_vs_epoch[0,epoch-1]-1)*100.
				deltaLoss_test= (loss_test/self.test_loss_vs_epoch[0,epoch-1]-1)*100.
				deltaAcc_train= (accuracy_train/self.train_accuracy_vs_epoch[0,epoch-1]-1)*100.
				deltaAcc_test= (accuracy_test/self.test_accuracy_vs_epoch[0,epoch-1]-1)*100.

			self.train_loss_vs_epoch[0,epoch]= loss_train
			self.train_type_loss_vs_epoch[0,epoch]= type_loss_train
			self.train_pars_loss_vs_epoch[0,epoch]= pars_loss_train
			self.test_loss_vs_epoch[0,epoch]= loss_test
			self.test_type_loss_vs_epoch[0,epoch]= type_loss_test
			self.test_pars_loss_vs_epoch[0,epoch]= pars_loss_test
			self.train_accuracy_vs_epoch[0,epoch]= accuracy_train
			self.test_accuracy_vs_epoch[0,epoch]= accuracy_test
			
			logger.info("EPOCH %d: loss(train)=%s (dl=%s), loss(test)=%s (dl=%s), accuracy(train)=%s (da=%s), accuracy(test)=%s (da=%s)" % (epoch,loss_train,deltaLoss_train,loss_test,deltaLoss_test,accuracy_train,deltaAcc_train,accuracy_test,deltaAcc_test))
			

			# - Flip train data
			if self.flip_train_data and self.nobjects>1 and self.learn_labels and self.learn_pars:
				logger.info("Flipping train data at epoch %d ..." % epoch)
				self.__do_train_data_flip()	

			# - Get predictions for test sample and flip according to smallest MSE match
			if self.flip_test_data and self.nobjects>1 and self.learn_labels and self.learn_pars:
				logger.info("Flipping test data at epoch %d ..." % epoch)
				self.__do_train_data_flip()

		logger.info("NN training completed.")

		#===========================
		#==   SAVE NN
		#===========================
		#- Save the model weights
		logger.info("Saving NN weights ...")
		self.model.save_weights('model_weights.h5')

		# -Save the model architecture in json format
		logger.info("Saving NN architecture in json format ...")
		with open('model_architecture.json', 'w') as f:
			f.write(self.model.to_json())
		
		#- Save the model
		logger.info("Saving full NN model ...")
		self.model.save('model.h5')

		# - Save the networkarchitecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.model, to_file=self.outfile_model)

		return 0


	###########################
	##     EVALUATE NETWORK
	###########################
	def __evaluate_train_results(self):
		""" Evaluating network train results """

		#================================
		#==   SAVE TRAIN METRICS
		#================================
		logger.info("Saving train metrics (loss, accuracy) to file ...")
		N= self.train_loss_vs_epoch.shape[1]
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		metrics_data= np.concatenate(
			(epoch_ids,self.train_type_loss_vs_epoch.reshape(N,1),self.train_pars_loss_vs_epoch.reshape(N,1),self.train_loss_vs_epoch.reshape(N,1),self.test_type_loss_vs_epoch.reshape(N,1),self.test_pars_loss_vs_epoch.reshape(N,1),self.test_loss_vs_epoch.reshape(N,1),self.train_accuracy_vs_epoch.reshape(N,1),self.test_accuracy_vs_epoch.reshape(N,1)),
			axis=1
		)
			
		head= '# epoch - type loss - pars loss - tot loss - type loss (test) - pars loss (test) - tot loss (test) - accuracy - accuracy (test)'
		Utils.write_ascii(metrics_data,self.outfile_nnout_metrics,head)	

		
		#================================
		#==   EVALUATE NN ON TRAIN DATA
		#================================
		# - Computing true & false detections		
		logger.info("Computing NN performance metrics on train data ...")
		model_predictions= self.model.predict(self.inputs_train)

		detThr= 0.5
		nobjs_tot= 0
		nobjs_true= 0
		nobjs_rec= 0
		nobjs_rec_true= 0
		nobjs_rec_false= 0
		s_list= []
		xpull_list= []
		ypull_list= []
		spull_list= []
		nnout_train= []

		if self.learn_labels:
			predictions_labels_train= model_predictions[:,:self.nobjects]
			nsamples_train= self.outputs_labels_train.shape[0]
					
			if self.learn_pars:
				predictions_train= model_predictions[:,self.nobjects:]

				for i in range(nsamples_train):
					target= self.outputs_labels_train[i,:]
					pred= predictions_labels_train[i,:]	
					target_pars= self.outputs_train[i,:]
					pred_pars= predictions_train[i,:]


					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0

					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(n)
						obj_data.append(target[index])
						obj_data.append(pred[index])
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_train.append(obj_data)

					for index in true_obj_indexes:
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars) < target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						if pred[index]>detThr:
							ntrue+= 1
							x0_rec= pred_pars[0 + index*self.npars]
							y0_rec= pred_pars[1 + index*self.npars]
							S_rec= 1
							if (2 + index*self.npars)<pred_pars.size:
								S_rec= pred_pars[2 + index*self.npars]

							s_list.append(np.log10(S_true))
							spull_list.append(S_rec/S_true-1)
							xpull_list.append(x0_rec-x0_true)
							ypull_list.append(y0_rec-y0_true)

					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1
	
					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_train= 0
				reliability_train= 0
				if nobjs_tot>0:	
					completeness_train= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_train= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Train Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_train),nobjs_rec_true,nobjs_rec,str(reliability_train)))

			else:	

				for i in range(nsamples_train):
					target= self.outputs_labels_train[i,:]
					pred= predictions_labels_train[i,:]	
					
					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0

					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(n)
						obj_data.append(target[index])
						obj_data.append(pred[index])
			
						nnout_train.append(obj_data)
				
					for index in true_obj_indexes:	
						if pred[index]>detThr:
							ntrue+= 1
							
					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1
	
					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_train= 0
				reliability_train= 0
				if nobjs_tot>0:	
					completeness_train= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_train= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Train Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_train),nobjs_rec_true,nobjs_rec,str(reliability_train)))

		else:

			if self.learn_pars:

				nsamples_train= self.outputs_train.shape[0]
				predictions_train= model_predictions

				for i in range(nsamples_train):
					target_pars= self.outputs_train[i,:]
					pred_pars= predictions_train[i,:]

					for index in range(self.nobjects):
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars) < target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						x0_rec= pred_pars[0 + index*self.npars]
						y0_rec= pred_pars[1 + index*self.npars]
						S_rec= 1
						if (2 + index*self.npars)<pred_pars.size:
							S_rec= pred_pars[2 + index*self.npars]

						s_list.append(np.log10(S_true))
						spull_list.append(S_rec/S_true-1)
						xpull_list.append(x0_rec-x0_true)
						ypull_list.append(y0_rec-y0_true)

						obj_data= []
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_train.append(obj_data)

			else:
				logger.error("You need to select learning of at least one between pars or labels!")
				return -1
					
		# - Write ascii file with results
		logger.info("Writing ascii file with NN performances on train data ...")
		Utils.write_ascii(np.array(nnout_train),self.outfile_nnout_train,'# nobjects - target label - predicted label - target pars - predicted pars')	



		#predictions_labels_train= nnout_train[:,:self.nobjects]
		#predictions_train= nnout_train[:,self.nobjects:]

		#for i in range(predictions_labels_train.shape[0]):
		#	target= ','.join(map(str, self.outputs_labels_train[i,:]))	
		#	pred= ','.join(map(str, predictions_labels_train[i,:]))
		#	logger.debug("Train labels entry no. %d: target=[%s], pred=[%s]" % (i+1,target,pred) )

		#for i in range(predictions_train.shape[0]):
		#	target= ','.join(map(str, self.outputs_train[i,:]))
		#	pred= ','.join(map(str, predictions_train[i,:]))
		#	logger.debug("Train spars entry no. %d: target=[%s], pred=[%s]" % (i+1,target,pred) )


		# - Computing true & false detections
		#nsamples_train= self.outputs_labels_train.shape[0]
		#detThr= 0.5
		#nobjs_tot= 0
		#nobjs_true= 0
		#nobjs_rec= 0
		#nobjs_rec_true= 0
		#nobjs_rec_false= 0
		#s_list= []
		#xpull_list= []
		#ypull_list= []
		#spull_list= []
		#nnout_train= []
	
		#for i in range(nsamples_train):
		#	target= self.outputs_labels_train[i,:]
		#	pred= predictions_labels_train[i,:]	
		#	target_pars= self.outputs_train[i,:]
		#	pred_pars= predictions_train[i,:]

		#	true_obj_indexes= np.argwhere(target==1).flatten()
		#	rec_obj_indexes= np.argwhere(pred>detThr).flatten()
		#	n= len(true_obj_indexes)
		#	nrec= len(rec_obj_indexes)
		#	ntrue= 0
		#	nrec_true= 0
		#	nrec_false= 0

		#	for index in range(self.nobjects):
		#		obj_data= []
		#		obj_data.append(target[index])
		#		obj_data.append(pred[index])
		#		for k in range(self.npars):
		#			obj_data.append(target_pars[k+index*self.npars])
		#			obj_data.append(pred_pars[k+index*self.npars])
		#		
		#		nnout_train.append(obj_data)
			
		#	for index in true_obj_indexes:
		#		x0_true= target_pars[0 + index*self.npars]
		#		y0_true= target_pars[1 + index*self.npars]
		#		S_true= 1
		#		if (2 + index*self.npars) < target_pars.size:
		#			S_true= target_pars[2 + index*self.npars]
#
		#		if pred[index]>detThr:
		#			ntrue+= 1
		#			x0_rec= pred_pars[0 + index*self.npars]
		#			y0_rec= pred_pars[1 + index*self.npars]
		#			S_rec= 1
		#			if (2 + index*self.npars)<pred_pars.size:
		#				S_rec= pred_pars[2 + index*self.npars]

		#			s_list.append(np.log10(S_true))
		#			spull_list.append(S_rec/S_true-1)
		#			xpull_list.append(x0_rec-x0_true)
		#			ypull_list.append(y0_rec-y0_true)

		#	for index in rec_obj_indexes:
		#		if target[index]==1:
		#			nrec_true+= 1
		#		else:
		#			nrec_false+= 1
	
		#	nobjs_tot+= n
		#	nobjs_rec+= nrec
		#	nobjs_true+= ntrue
		#	nobjs_rec_true+= nrec_true 
		#	nobjs_rec_false+= nrec_false

		#completeness_train= 0
		#reliability_train= 0
		#if nobjs_tot>0:	
		#	completeness_train= float(nobjs_true)/float(nobjs_tot)
		#if nobjs_rec>0:
		#	reliability_train= float(nobjs_rec_true)/float(nobjs_rec)

		#logger.info("NN Train Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_train),nobjs_rec_true,nobjs_rec,str(reliability_train)))

		# - Write ascii file with results
		#logger.info("Writing ascii file with NN performances on train data ...")
		#Utils.write_ascii(np.array(nnout_train),self.outfile_nnout_train,'# target label - predicted label - target pars - predicted pars')	

		#================================
		#==   EVALUATE NN ON TEST DATA
		#================================
		logger.info("Computing NN performance metrics on test data ...")
		model_predictions= self.model.predict(self.inputs_test)


		detThr= 0.5
		nobjs_tot= 0
		nobjs_true= 0
		nobjs_rec= 0
		nobjs_rec_true= 0
		nobjs_rec_false= 0
		s_list_test= []
		xpull_list_test= []
		ypull_list_test= []
		spull_list_test= []
		nnout_test= []

		if self.learn_labels:

			predictions_labels_test= model_predictions[:,:self.nobjects]
			nsamples_test= self.outputs_labels_test.shape[0]
		
			if self.learn_pars:
				predictions_test= model_predictions[:,self.nobjects:]

				for i in range(nsamples_test):
					target= self.outputs_labels_test[i,:]
					pred= predictions_labels_test[i,:]	
					target_pars= self.outputs_test[i,:]
					pred_pars= predictions_test[i,:]

					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0
		
					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(n)
						obj_data.append(target[index])
						obj_data.append(pred[index])
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_test.append(obj_data)

					for index in true_obj_indexes:
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars)<target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						if pred[index]>detThr:
							ntrue+= 1
							x0_rec= pred_pars[0 + index*self.npars]
							y0_rec= pred_pars[1 + index*self.npars]
							S_rec= 1
							if (2 + index*self.npars)<pred_pars.size:			
								S_rec= pred_pars[2 + index*self.npars]

							s_list_test.append(np.log10(S_true))
							spull_list_test.append(S_rec/S_true-1)
							xpull_list_test.append(x0_rec-x0_true)
							ypull_list_test.append(y0_rec-y0_true)

					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1

					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_test= 0
				reliability_test= 0
				if nobjs_tot>0:
					completeness_test= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_test= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Test Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_test),nobjs_rec_true,nobjs_rec,str(reliability_test)))

			else:

				for i in range(nsamples_test):
					target= self.outputs_labels_test[i,:]
					pred= predictions_labels_test[i,:]	
					
					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0
		
					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(n)
						obj_data.append(target[index])
						obj_data.append(pred[index])
						
						nnout_test.append(obj_data)

					for index in true_obj_indexes:
						if pred[index]>detThr:
							ntrue+= 1
							
					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1

					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_test= 0
				reliability_test= 0
				if nobjs_tot>0:
					completeness_test= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_test= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Test Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_test),nobjs_rec_true,nobjs_rec,str(reliability_test)))

		else:
			if self.learn_pars:

				predictions_test= model_predictions
				nsamples_test= self.outputs_test.shape[0]
				
				for i in range(nsamples_test):
					target_pars= self.outputs_test[i,:]
					pred_pars= predictions_test[i,:]

					
					for index in range(self.nobjects):
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars)<target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						x0_rec= pred_pars[0 + index*self.npars]
						y0_rec= pred_pars[1 + index*self.npars]
						S_rec= 1
						if (2 + index*self.npars)<pred_pars.size:			
							S_rec= pred_pars[2 + index*self.npars]

						s_list_test.append(np.log10(S_true))
						spull_list_test.append(S_rec/S_true-1)
						xpull_list_test.append(x0_rec-x0_true)
						ypull_list_test.append(y0_rec-y0_true)

						obj_data= []
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_test.append(obj_data)

			else:
				logger.error("You need to select learning of at least one between pars or labels!")
				return -1


		# - Write ascii file with results
		logger.info("Write ascii file with NN performances on test data ...")
		Utils.write_ascii(np.array(nnout_test),self.outfile_nnout_test,'# nobjects - target label - predicted label - target pars - predicted pars')	



		#predictions_labels_test= nnout_test[:,:self.nobjects]
		#predictions_test= nnout_test[:,self.nobjects:]

		#nsamples_test= self.outputs_labels_test.shape[0]
		#detThr= 0.5
		#nobjs_tot= 0
		#nobjs_true= 0
		#nobjs_rec= 0
		#nobjs_rec_true= 0
		#nobjs_rec_false= 0
		#s_list_test= []
		#xpull_list_test= []
		#ypull_list_test= []
		#spull_list_test= []
		#nnout_test= []

		#for i in range(nsamples_test):
		#	target= self.outputs_labels_test[i,:]
		#	pred= predictions_labels_test[i,:]	
		#	target_pars= self.outputs_test[i,:]
		#	pred_pars= predictions_test[i,:]

		#	true_obj_indexes= np.argwhere(target==1).flatten()
		#	rec_obj_indexes= np.argwhere(pred>detThr).flatten()
		#	n= len(true_obj_indexes)
		#	nrec= len(rec_obj_indexes)
		#	ntrue= 0
		#	nrec_true= 0
		#	nrec_false= 0
		
		#	for index in range(self.nobjects):
		#		obj_data= []
	 	#		obj_data.append(target[index])
		#		obj_data.append(pred[index])
		#		for k in range(self.npars):
		#			obj_data.append(target_pars[k+index*self.npars])
		#			obj_data.append(pred_pars[k+index*self.npars])
				
		#		nnout_test.append(obj_data)

		#	for index in true_obj_indexes:
		#		x0_true= target_pars[0 + index*self.npars]
		#		y0_true= target_pars[1 + index*self.npars]
		#		S_true= 1
		#		if (2 + index*self.npars)<target_pars.size:
		#			S_true= target_pars[2 + index*self.npars]

		#		if pred[index]>detThr:
		#			ntrue+= 1
		#			x0_rec= pred_pars[0 + index*self.npars]
		#			y0_rec= pred_pars[1 + index*self.npars]
		#			S_rec= 1
		#			if (2 + index*self.npars)<pred_pars.size:			
		#				S_rec= pred_pars[2 + index*self.npars]

		#			s_list_test.append(np.log10(S_true))
		#			spull_list_test.append(S_rec/S_true-1)
		#			xpull_list_test.append(x0_rec-x0_true)
		#			ypull_list_test.append(y0_rec-y0_true)

		#	for index in rec_obj_indexes:
		#		if target[index]==1:
		#			nrec_true+= 1
		#		else:
		#			nrec_false+= 1

		#	nobjs_tot+= n
		#	nobjs_rec+= nrec
		#	nobjs_true+= ntrue
		#	nobjs_rec_true+= nrec_true 
		#	nobjs_rec_false+= nrec_false

		#logger.info("NN Test Results: nobjs_true=%d, nobjs_tot=%d, nobjs_rec_true=%d, nobjs_rec=%d" % (nobjs_true,nobjs_tot,nobjs_rec_true,nobjs_rec))

		#completeness_test= 0
		#reliability_test= 0
		#if nobjs_tot>0:
		#	completeness_test= float(nobjs_true)/float(nobjs_tot)
		#if nobjs_rec>0:
		#	reliability_test= float(nobjs_rec_true)/float(nobjs_rec)

		#logger.info("NN Test Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_test),nobjs_rec_true,nobjs_rec,str(reliability_test)))

		# - Write ascii file with results
		#logger.info("Write ascii file with NN performances on test data ...")
		#Utils.write_ascii(np.array(nnout_test),self.outfile_nnout_test,'# target label - predicted label - target pars - predicted pars')	


		#================================
		#==   PLOT LOSS
		#================================
		# - Plot the total loss, type loss, spars loss
		logger.info("Plot the network loss metric to file ...")
		lossNames = ["loss"]
		plt.style.use("ggplot")
		#(fig, ax) = plt.subplots(1, 1, figsize=(20,20),squeeze=False)
		(fig, ax) = plt.subplots(3, 1, figsize=(20,20))
		
		# Total loss
		ax[0].set_title("Total Loss")
		ax[0].set_xlabel("Epoch #")
		ax[0].set_ylabel("Loss")
		ax[0].plot(np.arange(0, self.nepochs), self.train_loss_vs_epoch[0], label="TRAIN SAMPLE")
		ax[0].plot(np.arange(0, self.nepochs), self.test_loss_vs_epoch[0], label="TEST SAMPLE")
		ax[0].legend()		

		# Type loss
		ax[1].set_title("Classification Loss")
		ax[1].set_xlabel("Epoch #")
		ax[1].set_ylabel("Loss")
		ax[1].plot(np.arange(0, self.nepochs), self.train_type_loss_vs_epoch[0], label="TRAIN SAMPLE")
		ax[1].plot(np.arange(0, self.nepochs), self.test_type_loss_vs_epoch[0], label="TEST SAMPLE")
		ax[1].legend()

		# Pars loss
		ax[2].set_title("Pars Loss")
		ax[2].set_xlabel("Epoch #")
		ax[2].set_ylabel("Loss")
		ax[2].plot(np.arange(0, self.nepochs), self.train_pars_loss_vs_epoch[0], label="TRAIN SAMPLE")
		ax[2].plot(np.arange(0, self.nepochs), self.test_pars_loss_vs_epoch[0], label="TEST SAMPLE")
		ax[2].legend()	

		#for (i, lossName) in enumerate(lossNames):
		#	title = "Loss for {}".format(lossName) if lossName != "loss" else "Total loss"
		#	ax[i,0].set_title(title)
		#	ax[i,0].set_xlabel("Epoch #")
		#	ax[i,0].set_ylabel("Loss")
		#	ax[i,0].plot(np.arange(0, self.nepochs), self.train_loss_vs_epoch[i], label="TRAIN SAMPLE - " + lossName)
		#	ax[i,0].plot(np.arange(0, self.nepochs), self.test_loss_vs_epoch[i], label="TEST SAMPLE - " + lossName)
		#	ax[i,0].legend()

		plt.tight_layout()
		plt.savefig(self.outfile_loss)
		plt.close()

		#================================
		#==   PLOT ACCURACY
		#================================
		# - Plot the accuracy
		logger.info("Plot the network accuracy metric to file ...")
		accuracyNames = ["acc"]
		plt.style.use("ggplot")
		(fig, ax) = plt.subplots(1, 1, figsize=(20,20),squeeze=False)

		for (i, accuracyName) in enumerate(accuracyNames):
			ax[i,0].set_title("Accuracy for {}".format(accuracyName))
			ax[i,0].set_xlabel("Epoch #")
			ax[i,0].set_ylabel("Accuracy")
			ax[i,0].plot(np.arange(0, self.nepochs), self.train_accuracy_vs_epoch[i], label="TRAIN SAMPLE - " + accuracyName)
			ax[i,0].plot(np.arange(0, self.nepochs), self.test_accuracy_vs_epoch[i], label="TEST SAMPLE - " + accuracyName)
			ax[i,0].legend()

		plt.tight_layout()
		plt.savefig(self.outfile_accuracy)
		plt.close()

		#================================
		#==   PLOT RECO ACCURACY
		#================================
		# - Plot x, y position reco accuracy for detected sources
		if self.learn_pars:

			logger.info("Plot the source (x, y) position accuracy ...")
			plt.style.use("ggplot")
			(fig, ax) = plt.subplots(2, 2, figsize=(20,20))

			ax[0,0].set_title("x Position Accuracy")
			ax[0,0].set_xlabel("logS (Jy/beam)")
			ax[0,0].set_ylabel("dx")
			ax[0,0].scatter(np.array(s_list),np.array(xpull_list),label="TRAIN SAMPLE")
			ax[0,0].legend()

			ax[0,1].set_title("y Position Accuracy")
			ax[0,1].set_xlabel("logS (Jy/beam)")
			ax[0,1].set_ylabel("dy")
			ax[0,1].scatter(np.array(s_list),np.array(ypull_list),label="TRAIN SAMPLE")
			ax[0,1].legend()

			ax[1,0].set_title("x Position Accuracy")
			ax[1,0].set_xlabel("logS (Jy/beam)")
			ax[1,0].set_ylabel("dx")
			ax[1,0].scatter(np.array(s_list_test),np.array(xpull_list_test),label="TEST SAMPLE")
			ax[1,0].legend()

			ax[1,1].set_title("y Position Accuracy")
			ax[1,1].set_xlabel("logS (Jy/beam)")
			ax[1,1].set_ylabel("dy")
			ax[1,1].scatter(np.array(s_list_test),np.array(ypull_list_test),label="TEST SAMPLE")
			ax[1,1].legend()

			plt.tight_layout()
			plt.savefig(self.outfile_posaccuracy)
			plt.close()

		#================================
		#==   PLOT FLUX ACCURACY
		#================================
		# - Plot flux reco accuracy for detected sources
		if self.learn_pars:
			logger.info("Plot the source flux accuracy ...")
			plt.style.use("ggplot")
			(fig, ax) = plt.subplots(2, 1, figsize=(20,20))

			ax[0].set_title("Flux Accuracy")
			ax[0].set_xlabel("logS (Jy/beam)")
			ax[0].set_ylabel("dS")
			ax[0].scatter(np.array(s_list),np.array(spull_list),label="TRAIN SAMPLE")
			ax[0].legend()

			ax[1].set_title("Flux Accuracy")
			ax[1].set_xlabel("logS (Jy/beam)")
			ax[1].set_ylabel("dS")
			ax[1].scatter(np.array(s_list_test),np.array(spull_list_test),label="TEST SAMPLE")
			ax[1].legend()

			plt.tight_layout()
			plt.savefig(self.outfile_fluxaccuracy)
			plt.close()
	
		#===================================
		#==   PLOT CONV LAYER ACTIVATIONS
		#===================================
		#logger.info("Plot the convolution layer activations ...")
		#self.__draw_conv_layer_activations(self.inputs_train)

		return 0


	#########################################
	##     EVAL NN RESULTS ON TRAIN
	#########################################
	def __eval_train_results(self):
		""" Evaluate NN results on train data """

		logger.info("Computing NN performance metrics on train data ...")
		nnout_train= self.model.predict(self.inputs_train)

		detThr= 0.5
		nobjs_tot= 0
		nobjs_true= 0
		nobjs_rec= 0
		nobjs_rec_true= 0
		nobjs_rec_false= 0
		s_list= []
		xpull_list= []
		ypull_list= []
		spull_list= []
		nnout_train= []

		# - Computing true & false detections		
		if self.learn_labels:
			predictions_labels_train= nnout_train[:,:self.nobjects]
			nsamples_train= self.outputs_labels_train.shape[0]
					
			if self.learn_pars:
				predictions_train= nnout_train[:,self.nobjects:]

				for i in range(nsamples_train):
					target= self.outputs_labels_train[i,:]
					pred= predictions_labels_train[i,:]	
					target_pars= self.outputs_train[i,:]
					pred_pars= predictions_train[i,:]

					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0

					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(target[index])
						obj_data.append(pred[index])
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_train.append(obj_data)

					for index in true_obj_indexes:
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars) < target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						if pred[index]>detThr:
							ntrue+= 1
							x0_rec= pred_pars[0 + index*self.npars]
							y0_rec= pred_pars[1 + index*self.npars]
							S_rec= 1
							if (2 + index*self.npars)<pred_pars.size:
								S_rec= pred_pars[2 + index*self.npars]

							s_list.append(np.log10(S_true))
							spull_list.append(S_rec/S_true-1)
							xpull_list.append(x0_rec-x0_true)
							ypull_list.append(y0_rec-y0_true)

					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1
	
					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_train= 0
				reliability_train= 0
				if nobjs_tot>0:	
					completeness_train= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_train= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Train Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_train),nobjs_rec_true,nobjs_rec,str(reliability_train)))

			else:	

				for i in range(nsamples_train):
					target= self.outputs_labels_train[i,:]
					pred= predictions_labels_train[i,:]	
					
					true_obj_indexes= np.argwhere(target==1).flatten()
					rec_obj_indexes= np.argwhere(pred>detThr).flatten()
					n= len(true_obj_indexes)
					nrec= len(rec_obj_indexes)
					ntrue= 0
					nrec_true= 0
					nrec_false= 0

					for index in range(self.nobjects):
						obj_data= []
						obj_data.append(target[index])
						obj_data.append(pred[index])
			
						nnout_train.append(obj_data)
				
					for index in true_obj_indexes:	
						if pred[index]>detThr:
							ntrue+= 1
							
					for index in rec_obj_indexes:
						if target[index]==1:
							nrec_true+= 1
						else:
							nrec_false+= 1
	
					nobjs_tot+= n
					nobjs_rec+= nrec
					nobjs_true+= ntrue
					nobjs_rec_true+= nrec_true 
					nobjs_rec_false+= nrec_false

				completeness_train= 0
				reliability_train= 0
				if nobjs_tot>0:	
					completeness_train= float(nobjs_true)/float(nobjs_tot)
				if nobjs_rec>0:
					reliability_train= float(nobjs_rec_true)/float(nobjs_rec)

				logger.info("NN Train Results: Completeness(det/tot=%d/%d)=%s, Reliability(true/rec=%d/%d)=%s" % (nobjs_true,nobjs_tot,str(completeness_train),nobjs_rec_true,nobjs_rec,str(reliability_train)))

		else:

			if self.learn_pars:

				nsamples_train= self.outputs_train.shape[0]
				predictions_train= nnout_train

				for i in range(nsamples_train):
					target_pars= self.outputs_train[i,:]
					pred_pars= predictions_train[i,:]

					for index in range(self.nobjects):
						x0_true= target_pars[0 + index*self.npars]
						y0_true= target_pars[1 + index*self.npars]
						S_true= 1
						if (2 + index*self.npars) < target_pars.size:
							S_true= target_pars[2 + index*self.npars]

						x0_rec= pred_pars[0 + index*self.npars]
						y0_rec= pred_pars[1 + index*self.npars]
						S_rec= 1
						if (2 + index*self.npars)<pred_pars.size:
							S_rec= pred_pars[2 + index*self.npars]

						s_list.append(np.log10(S_true))
						spull_list.append(S_rec/S_true-1)
						xpull_list.append(x0_rec-x0_true)
						ypull_list.append(y0_rec-y0_true)

						obj_data= []
						obj_data.append(target[index])
						obj_data.append(pred[index])
						for k in range(self.npars):
							obj_data.append(target_pars[k+index*self.npars])
							obj_data.append(pred_pars[k+index*self.npars])
				
						nnout_train.append(obj_data)

			else:
				logger.error("You need to select learning of at least one between pars or labels!")
				return -1
					
		# - Write ascii file with results
		logger.info("Writing ascii file with NN performances on train data ...")
		Utils.write_ascii(np.array(nnout_train),self.outfile_nnout_train,'# target label - predicted label - target pars - predicted pars')	

		return 0


	#########################################
	##     DRAW NN CONV LAYER ACTIVATIONS
	#########################################
	def __draw_conv_layer_activations(self,inputs):
		""" Draw convolution layer activations for a given input """

		# - Create activation model	
		layer_name= None
		layer_outputs = [layer.output for layer in self.model.layers if layer.name == layer_name or layer_name is None][1:]

		activation_model= models.Model(inputs=self.model.input, outputs=layer_outputs)

		# - Get layer names
		layer_names = []
		for layer in self.model.layers:
			layer_names.append(layer.name)

		logger.info("== NN Layer names ==")
		print(layer_names)

		# - Get activations
		activations= activation_model.predict(inputs)
		images_per_row = 16

		for layer_name, layer_activation in zip(layer_names, activations):

			# Skip input layer
			if layer_name=='input':
				continue

			# Draw only conv layers
			if layer_activation.ndim!=4:
				continue

			n_features = layer_activation.shape[-1]
			size = layer_activation.shape[1]
			n_cols = n_features # images_per_row

			logger.info("layer_name=%s, n_features=%d, size=%d, n_cols=%d" % (layer_name,n_features,size,n_cols))
		

			# - Fill channel image
			channel_images= []
			for index in range(n_features):
				channel_image = layer_activation[0,:,:, index]
				channel_images.append(channel_image)

			# - Organize images in a grid
			nimg_row= int(np.sqrt(n_features))
			nimg_col= int(np.ceil(n_features/float(nimg_row)))
			logger.info("Organize %d feature maps in pretty image of size (%d,%d)" % (n_features,nimg_row,nimg_col))

			display_grid = np.zeros((size*nimg_col, size*nimg_row))
		
			for col in range(nimg_col):
				for row in range(nimg_row):
					index= col * nimg_row + row
					if index>=n_features:
						continue
					logger.debug("Filling image at index=%d (col=%d, row=%d)" % (index,col,row))
					channel_image= channel_images[index]
					display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
			
		
			# - Draw plots
			fig_name= 'act_layer' + layer_name + '.png' 

			scale = 1. / size
			plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
			plt.title(layer_name)
			plt.grid(False)
			#plt.imshow(display_grid, aspect='auto', cmap='viridis')
			#plt.show()
			plt.tight_layout()
			plt.savefig(fig_name)
			plt.close()


	#####################################
	##     RUN NN TRAIN
	#####################################
	def train(self):
		""" Run network training """

		#===========================
		#==   SET TRAINING DATA
		#===========================	
		logger.info("Setting training data from provider ...")
		status= self.__set_data()
		if status<0:
			logger.error("Train data set failed!")
			return -1

		#===========================
		#==   BUILD NN
		#===========================
		#- Create the network
		logger.info("Building network architecture ...")
		status= self.__build_network(self.nnarc_file)
		if status<0:
			logger.error("NN build failed!")
			return -1

		#===========================
		#==   TRAIN NN
		#===========================
		logger.info("Training network ...")
		status= self.__train_network()
		if status<0:
			logger.error("NN train failed!")
			return -1

		#===========================
		#==   EVALUATE NN TRAIN
		#===========================
		logger.info("Evaluating trained network results ...")
		status= self.__evaluate_train_results()
		if status<0:
			logger.error("NN train evaluation failed!")
			return -1

		return 0

