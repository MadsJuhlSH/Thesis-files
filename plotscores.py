#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to plot the different variables
"""
print("Program running...")
# Import packages and functions
import sys
import os
from os import path
import argparse
import logging as log
from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import math as m

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from scipy.special import logit
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error

from utils import timediftostr, Build_Folder_Structure, header, footer, print_col_names, h5ToDf, accuracy, HyperOpt_RandSearch, auc_eval, HeatMap_rand

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

print("Packages and functions loaded")
t_total_start = time()

"""
weightTypes = [ "regWeight_nEst10", "regWeight_nEst40", "regWeight_nEst100", "revWeight_nEst10", "revWeight_nEst40", "revWeight_nEst100" ]
weightNames = weightTypes.copy()
weightsString = "Choose weights:"
for iWeight, weight in enumerate(weightNames):
    weightsString = weightsString + f" {iWeight} = {weight},"
    """
#%%############################################################################
#   Parser
###############################################################################
parser = argparse.ArgumentParser(description='Electron isolation with the LightGBM framework.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--outdir', action='store', default="output/Variables/", type=str,
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', type=int, required=False, default=10,
                    help='Amount of njobs (default = 10)')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["ele", "muo"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')
#parser.add_argument('--weights', action='store', type=int, required=True,
#                    help=weightsString)


args = parser.parse_args()

#%%############################################################################
#   Filenames and directories
###############################################################################
# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"
output_dir = args.outdir + "Variables_Plot_" + args.tag + "_" + timelabel + "/"
# Check if output folder already exists
if os.path.exists(output_dir):
    log.error(f"Output already exists - please remove yourself. Output dir: {output_dir}")
    quit()

# Figure subdirectory
figure_dir = output_dir + "Images/"

# Build the folder structure
Build_Folder_Structure([figure_dir])

#%%############################################################################
#   Print initial log info
###############################################################################
# Log INFO for run
print(f'')
print(f'---------------------------- INFO --------------------------------')
print(f'Datafile:                 {args.paths}')
print(f'Output directory:         {output_dir}')
print(f'Number of threads:        {args.njobs}')
print(f'------------------------------------------------------------------')
#%%############################################################################
#   Importing data.
###############################################################################
filenames = []
data_name = []
data_list = []

morebkg = 1
log.info(f"Importing files and retreiwing names")
for path in args.paths:
    # Name of data file
    filename_base = os.path.basename(path)
    filename = os.path.splitext(filename_base)[0]
    filenames.append(filename)

    # Name of process
    name = filename.split("_")
    data_name.append(name[0])

    # Data
    print(f"Using path: {path}")
    data_get = h5ToDf(path)
    data_get["process"] = name[0]
    data_list.append(data_get)

header("Importing and separating data")
t = time()

# Get hdf5 datafile as dataframe
data = pd.concat(data_list, ignore_index = True)
print(f"Starting plotting")
SIG = data["type"] == 1
BKG = data["type"] == 0
DataSig = data[SIG]
DataBkg = data[BKG]
if args.PartType == "ele":
    scorelist = ["pho_pIso_score","pho_pPid_score","ele1_eIso_score","ele1_ePid_score","ele2_eIso_score","ele2_ePid_score","Zee_score"]
elif args.PartType =="muo":
    scorelist = ["pho_pIso_score","pho_pPid_score","muo1_mIso_score","muo1_mPid_score","muo2_mIso_score","muo2_mPid_score","Zmm_score"]
for i in scorelist:
    print(f"Plotting for variable: {i}")
    #ndata = m.ceil(len(data[i])*0.99)
    #Ndata = m.ceil(ndata*0.99)
    #Data = data.nsmallest(ndata,i)
    #DATA = Data.nlargest(Ndata,i)
    fig, ax = plt.subplots(figsize=(12,12))
    #nbins = np.linspace(data[i].min(),data[i].max(),75)
    nbins = np.linspace(-10,20,201)
    ax.hist(data[i][SIG],bins=nbins,histtype = "step",label="Signal")
    ax.hist(data[i][BKG],bins=nbins,histtype = "step",label = "Background")
    ax.legend(loc="best")
    ax.set_title("Plot of the distribution of {}".format(i))
    fig.savefig(figure_dir + i)
    plt.close(fig)
    

print(f'')
print(f'')
print(f"END OF PROGRAM")