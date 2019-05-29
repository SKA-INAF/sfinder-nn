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
#from . import logger
from .utils import Utils

logger = logging.getLogger(__name__)

##############################
##     CLASS DEFINITIONS
##############################
class DataGenerator(object):
	""" Class to generate train data

			Attributes:
				None
	"""

	def __init__(self):
		""" Return a DataGenerator object """
		
		# - Bkg generation
		self.img_file= None 
		self.img_data= None
		self.img_sizex= 0
		self.img_sizey= 0
		self.pixsize= 4 # in arcsec
		self.gen_bkg_from_img= True
		self.bkg_rms= 300.e-6
		self.bkg_mean= 0

		# - Source generation
		self.gridx= None
		self.gridy= None
		self.nsources_max= 5
		self.source_gen_marginx= 2
		self.source_gen_marginy= 2
		self.Smin= 1.e-6 # in Jy 
		self.Smax= 1 # in Jy
		self.Smodel= 'uniform'
		self.Sslope= 1.6
		self.truncate_models= False
		self.trunc_thr= 0.01 # 1% flux truncation at maximum

		# - Data generation
		self.nsamples_bkg= 10
		self.nsamples_source= 10
		self.train_img_sizex= 101
		self.train_img_sizey= 101
		
	#################################
	##     SETTER/GETTER METHODS
	#################################
	def enable_bkg_generation_from_img(self,choice):
		""" Turn on/off bkg generation from input image. """
		self.gen_bkg_from_img= choice

	def set_gen_bkg_rms(self,rms):
		""" Set generated bkg rms """
		self.bkg_rms= rms

	def set_gen_bkg_mean(self,mean):
		""" Set generated bkg mean"""
		self.bkg_mean= mean

	def set_img_filename(self,filename):
		""" Set the input residual image used to generate train data """
		self.img_file= filename

	def set_margins(self,marginx,marginy):
		""" Set margin in X & Y """
		self.marginx= marginx
		self.marginy= marginy

	def set_source_margins(self,marginx,marginy):
		""" Set margin in X & Y for source generation """
		self.source_gen_marginx= marginx
		self.source_gen_marginy= marginy

	def set_nsources_max(self,n):
		""" Set the maximum number of sources to be generated in train image """
		self.nsources_max= n

	def set_source_flux_rand_model(self,model):
		""" Set the source flux random model """
		self.Smodel= model

	def set_source_flux_rand_exp_slope(self,slope):
		""" Set the source flux expo model slope par """
		self.Sslope= slope	
	
	def set_source_flux_range(self,Smin,Smax):
		""" Set source flux range """
		self.Smin= Smin
		self.Smax= Smax	

	def set_beam_bmaj_range(self,bmaj_min,bmaj_max):
		""" Set beam bmaj range """
		self.beam_bmaj_min= bmaj_min
		self.beam_bmaj_max= bmaj_max	

	def set_beam_bmin_range(self,bmin_min,bmin_max):
		""" Set beam bmin range """
		self.beam_bmin_min= bmin_min
		self.beam_bmin_max= bmin_max	

	def set_beam_pa_range(self,pa_min,pa_max):
		""" Set beam pa range """
		self.beam_bpa_min= pa_min
		self.beam_bpa_max= pa_max	

	def set_bkg_sample_size(self,n):
		""" Set number of images for bkg to be generated """
		self.nsamples_bkg= n

	def set_source_sample_size(self,n):
		""" Set number of images for sources to be generated """
		self.nsamples_source= n

	def set_train_img_size(self,nx,ny):
		""" Set size of input image given to the network for training """
		self.train_img_sizex= nx
		self.train_img_sizey= ny
		
	#################################
	##     READ INPUT IMAGE
	#################################
	def read_img(self):
		""" Read input FITS image and set image data """

		# - Read FITS image
		try:
			self.img_data, header= Utils.read_fits(self.img_file)

		except Exception as ex:
			errmsg= 'Cannot read input image file: ' + self.img_file
			logger.error(errmsg)
			return -1

		imgsize= np.shape(self.img_data)
		self.img_sizex= imgsize[1]
		self.img_sizey= imgsize[0]

		# - Read fits metadata
		dx= np.abs(header['CDELT1']*3600.) # in arcsec
		dy= np.abs(header['CDELT2']*3600.) # in arcsec
		self.pixsize= min(dx,dy)

		return 0

	#################################
	##     MAKE BKG TRAIN DATA
	#################################
	def make_bkg_train_data(self,writeimg=False):
		""" Prepare bkg train data """

		# - Init data
		nx= self.img_sizex
		ny= self.img_sizey
		marginx= self.train_img_sizex/2
		marginy= self.train_img_sizey/2
		logger.info("Input image size (%s,%s), margins(%s,%s), crop image size(%s,%s)" % (nx,ny,marginx,marginy,self.train_img_sizex,self.train_img_sizey))
		
		# - Extract nsamples img
		index= 0

		while index < self.nsamples_bkg:
			if index%100==0 :
				logger.info("Generating bkg train image no. %s/%s ..." % (index+1,self.nsamples_bkg))
	
			if self.gen_bkg_from_img:
				# - Generate crop img center randomly
				x0= int(np.random.uniform(marginx,nx-marginx-1))
				y0= int(np.random.uniform(marginy,ny-marginy-1))
				logger.info("Extract crop image around pos(%s,%s)" % (x0,y0))
			
				# - Extract crop img data
				data_crop= Utils.crop_img(self.img_data,x0,y0,self.train_img_sizex,self.train_img_sizey)

			else:
				# - Generate random bkg data
				data_crop= self.generate_noise(self.train_img_sizex,self.train_img_sizey,self.bkg_rms,self.bkg_mean)			

			imgcropsize= np.shape(data_crop)
			
			# - Check data integrity (skip if all zeros or nan/inf)
			n_nonzero= np.count_nonzero(data_crop)
			n_finite= (np.isfinite(data_crop)).sum()
			if n_nonzero<=0 or n_finite<=0:
				logger.warn("Skip sample image (all pixels NaN/inf/zero)...")
				continue

			# - Save crop img to file?
			outfilename= 'train_bkg-RUN' + str(index+1) + '.fits'
			if writeimg:
				self.write_fits(data_crop,outfilename)

		return 0

	#################################
	##     GENERATE NOISE IMAGE
	#################################
	def generate_noise(self,nx,ny,sigma,mean=0):
		""" Generate image data from random noise """
		data= np.random.normal(mean,sigma,(ny,nx))
		return data

	#################################
	##     GENERATE BLOB
	#################################
	def generate_blob(self,ampl,x0,y0,sigmax,sigmay,theta,trunc_thr=0.01):
		""" Generate a blob 
				Arguments: 
					ampl: peak flux in Jy
					x0, y0: gaussian means in pixels
					sigmax, sigmay: gaussian sigmas in pixels
					theta: rotation in degrees
					trunc_thr: truncation significance threshold
		"""
		data= Gaussian2D(ampl,x0,y0,sigmax,sigmay,theta=math.radians(theta))(self.gridx, self.gridy)
		
		## Truncate data such that sum(data)_trunc/sum(data)<f
		f= trunc_thr 
		if self.truncate_models:
			totFlux= (float)(np.sum(data,axis=None))
			
			data_vect_sorted= np.ravel(data)
			data_csum= np.cumsum(data_vect_sorted)/totFlux
			fluxThr= data_vect_sorted[np.argmin(data_csum<f)]
			data[data<fluxThr] = 0		

		return data

	#################################
	##     MAKE SOURCE TRAIN DATA
	#################################
	def make_source_train_data(self,writeimg=False):
		""" Prepare source train data """

		# - Init data
		nx= self.img_sizex
		ny= self.img_sizey
		marginx= self.train_img_sizex/2
		marginy= self.train_img_sizey/2
		marginx_source= self.source_gen_marginx
		marginy_source= self.source_gen_marginy

		# - Initialize grid for source generation	
		logger.info("Generating grid for source generation ...")
		self.gridy, self.gridx = np.mgrid[0:self.train_img_sizey, 0:self.train_img_sizex]


		# - Set source randomization pars
		S_min= self.Smin 
		S_max= self.Smax 
		lgS_min= np.log10(S_min)
		lgS_max= np.log10(S_max)
		randomize_flux= False
		if self.Smin<self.Smax:
			randomize_flux= True

		randomize_gaus= False
		Bmaj_min= self.beam_bmaj_min
		Bmaj_max= self.beam_bmaj_max
		Bmin_min= self.beam_bmin_min
		Bmin_max= self.beam_bmin_max
		Pa_min= self.beam_bpa_min
		Pa_max= self.beam_bpa_max	
		if self.beam_bmaj_min<self.beam_bmaj_max:
			randomize_gaus= True
		if self.beam_bmin_min<self.beam_bmin_max:
			randomize_gaus= True
		if self.beam_bpa_min<self.beam_bpa_max:
			randomize_gaus= True
		
		# - Extract nsamples img
		index= 0

		while index < self.nsamples_source:
			if index%100==0 :
				logger.info("Generating source train image no. %s/%s ..." % (index+1,self.nsamples_source))
	
			if self.gen_bkg_from_img:
				# - Generate crop img center randomly
				x0= int(np.random.uniform(marginx,nx-marginx-1))
				y0= int(np.random.uniform(marginy,ny-marginy-1))
				ix= int(np.round(x0))
				iy= int(np.round(y0))
				if self.img_data[iy,ix]==0 or np.isnan(self.img_data[iy,ix]):
					logger.warn("Skip sample image crop centered on (%s,%s) (pixel is zero or nan) ..." % (x0,y0))
					continue
			
				# - Extract crop img data
				data_crop= Utils.crop_img(self.img_data,x0,y0,self.train_img_sizex,self.train_img_sizey)

			else:
				# - Generate random bkg data
				data_crop= self.generate_noise(self.train_img_sizex,self.train_img_sizey,self.bkg_rms,self.bkg_mean)	


			imgcropsize= np.shape(data_crop)
			
			# - Check data integrity (skip if all zeros or nan/inf)
			n_nonzero= np.count_nonzero(data_crop)
			n_finite= (np.isfinite(data_crop)).sum()
			if n_nonzero<=0 or n_finite<=0:
				logger.warn("Skip sample image crop centered on (%s,%s) (all pixels NaN/inf/zero) ..." % (x0,y0))
				continue

			# - Generate and add sources to cropped image
			nsources_max= int(round(np.random.uniform(1,self.nsources_max)))
			sources_data = Box2D(amplitude=0,x_0=0,y_0=0,x_width=2*self.train_img_sizex, y_width=2*self.train_img_sizey)(self.gridx, self.gridy)
			mask_data = Box2D(amplitude=0,x_0=0,y_0=0,x_width=2*self.train_img_sizex, y_width=2*self.train_img_sizey)(self.gridx, self.gridy)
			source_pars= []
			nsources= 0

			logger.info("Generating #%d sources in image ..." % (nsources_max))

			while nsources < nsources_max:
				# Generate source position
				x0_source= np.random.uniform(marginx_source,self.train_img_sizex-marginx_source-1)
				y0_source= np.random.uniform(marginy_source,self.train_img_sizey-marginy_source-1)
				ix= int(np.round(x0_source))
				iy= int(np.round(y0_source))
				
				# Skip if pixel already filled by a source or if crop data is nan
				if mask_data[iy,ix]!=0:
					logger.warn("Generated source position (%s,%s) is already taken (mask=%s) in image crop centered on (%s,%s), skip generation..." % (x0_source,y0_source,mask_data[iy,ix],x0,y0))
					continue
				if data_crop[iy,ix]==0 or np.isnan(data_crop[iy,ix]):
					logger.warn("Generated source position (%s,%s) on zero or nan pixel (data=%s) in image crop centered on (%s,%s), skip generation..." % (x0_source,y0_source,data_crop[iy,ix],x0,y0))
					continue
				

				## Generate flux uniform or expo in log
				S= S_min
				if randomize_flux:
					if self.Smodel=='uniform':
						lgS= np.random.uniform(lgS_min,lgS_max)
					elif self.Smodel=='exp':
						x= np.random.exponential(scale=1./self.Sslope)
						lgS= x + lgS_min
						if lgS>lgS_max:
							continue
					else:
						lgS= np.random.uniform(lgS_min,lgS_max)
					S= np.power(10,lgS)
				
				## Generate gaus pars
				if randomize_gaus:
					bmin= random.uniform(Bmin_min,Bmin_max)
					bmaj= random.uniform(bmin,Bmaj_max)
					pa= random.uniform(Pa_min,Pa_max)
				else:
					bmin= self.beam_bmin_min
					bmaj= self.beam_bmaj_min
					pa= self.beam_bpa_min

				sigmax= bmaj/(self.pixsize * SIGMA_TO_FWHM)
				sigmay= bmaj/(self.pixsize * SIGMA_TO_FWHM)
				theta = 90 + pa
				theta_rad= np.radians(theta)

				## Generate gaus 2D data
				logger.info("Generating source no. %d: (x0,y0,S,sigmax,sigmay,theta)=(%s,%s,%s,%s,%s,%s)" % (nsources,x0_source,y0_source,S,sigmax,sigmay,theta))
				blob_data= self.generate_blob(ampl=S,x0=x0_source,y0=y0_source,sigmax=sigmax,sigmay=sigmay,theta=theta,trunc_thr=self.trunc_thr)
				if blob_data is None:
					logger.warn("Failed to generate Gaus2D (hint: too large trunc threshold), skip and regenerate...")
					continue

				## Update source mask & counts				
				sources_data+= blob_data
				mask_data[iy,ix]+= S
				nsources+= 1
				sname= 'S' + str(nsources)
				source_pars.append([sname,x0_source,y0_source,S,sigmax,sigmay,theta_rad])
				
			
			# - Add generated sources to train image
			data_crop+= sources_data

			# - Save crop img and source pars to file?
			outfilename= 'train_source-RUN' + str(index+1) + '.fits'
			outfilename_pars= 'train_source_pars-RUN' + str(index+1) + '.dat'
			if writeimg:
				self.write_fits(data_crop,outfilename)
				self.write_ascii(np.array(source_pars),outfilename_pars,'# sname x0(pix) y0(pix) S(Jy/beam) sigmaX(pix) sigmaY(pix) theta(rad)')
	

			# Update sample counter
			index+= 1

		return 0


	#############################
	##     GENERATE TRAIN DATA
	#############################
	def generate_train_data(self):
		""" Generate training data """
	
		# - Read input image
		if self.gen_bkg_from_img:
			logger.info("Reading input image %s ..." % self.img_file)
			status= self.read_img()
			if status<0:
				return -1	

		# - Generate train data for bkg
		logger.info("Generating train data for bkg ...")
		status= self.make_bkg_train_data(self.writeimg)
		if status<0:
			return -1	

		# - Generate train data for sources
		logger.info("Generating train data for sources ...")
		status= self.make_source_train_data(self.writeimg)
		if status<0:
			return -1	

		return 0

