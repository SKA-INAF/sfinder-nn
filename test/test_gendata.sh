#!/bin/bash

generate_data.py --nx=51 --ny=51 --nsamples_bkg=10 --nsamples_source=10 --generate_bkg --bkg_rms=10.e-4 --bkg_mean=0  --nsources_max=1 --marginx_source=10 --marginy_source=10 --Smin=1 --Smax=1 --Smodel=exp --Sslope=1.6 --bmaj_min=24 --bmaj_max=24 --bmin_min=20 --bmin_max=20 --pa_min=90 --pa_max=90
