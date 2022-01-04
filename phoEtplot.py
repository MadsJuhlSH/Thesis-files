#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to plot the et from different types of photons
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
parser.add_argument('path', type=str,
                    help='Data file to train on.')
parser.add_argument('--outdir', action='store', default="output/Variables/", type=str,
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', type=int, required=False, default=10,
                    help='Amount of njobs (default = 10)')
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
output_dir = args.outdir + "PhotonEt_" + args.tag + "_" + timelabel + "/"
# Check if output folder already exists
if os.path.exists(output_dir):
    log.error(f"Output already exists - please remove yourself. Output dir: {output_dir}")
    quit()

# Figure subdirectory
figure_dir = output_dir 

# Build the folder structure
Build_Folder_Structure([figure_dir])

#%%############################################################################
#   Print initial log info
###############################################################################
# Log INFO for run
print(f'')
print(f'---------------------------- INFO --------------------------------')
print(f'Datafile:                 {args.path}')
print(f'Output directory:         {output_dir}')
print(f'Number of threads:        {args.njobs}')
print(f'------------------------------------------------------------------')
#%%############################################################################
#   Importing data.
###############################################################################
header("Importing and plotting data")
t = time()
particleType = {0 : 'Unknown',
                1 : 'UnknownElectron',
                2 : 'IsoElectron',
                3 : 'NonIsoElectron',
                4 : 'BkgElectron',
                5 : 'UnknownMuon',
                6 : 'IsoMuon',
                7 : 'NonIsoMuon',
                8 : 'BkgMuon',
                13: 'UnknownPhoton',
                14: 'IsoPhoton',
                15: 'NonIsoPhoton',
                16: 'BkgPhoton',
                17: 'Hadron',
                35: 'LJet',
                36: 'GJet',
                38: 'UnknownJet'}
# Get hdf5 datafile as dataframe
data = h5ToDf(args.path)
min,max,nbins = 1,150,150
print("Plotting overall et distribution.")
fig, ax = plt.subplots()
ax.set_title(args.tag+r" $E_T$")
ax.set_xlabel(r"$E_T$")
ax.set_ylabel("Frequency")

# Add Values to histogram 
ax.hist(data["pho_et"][data["type"]==1],bins=np.linspace(min,max,nbins),histtype="step", label = "Signal")
ax.hist(data["pho_et"][data["type"]==0],bins=np.linspace(min,max,nbins),histtype="step", label = "Background")

ax.legend()

# Save Histogram 
plt.tight_layout()
fig.savefig(output_dir+args.tag+" Et.png")
print("Plotting et distribution from each file.")
opaque = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
typename =["VBF Hllg", "gg Hllg", r"gg $\gamma\gamma$", r"VBF $\gamma\gamma$", r"gg $\gamma*\gamma$", r"VBF $\gamma*\gamma$",r"Zee$\gamma$", r"Z$\mu\mu\gamma$"]
fig, ax = plt.subplots()
ax.set_title(args.tag + r" $E_T$ for different interactions")
ax.set_xlabel(r"$E_T$")
ax.set_ylabel("Frequency")
#TYPES = np.unique(data["pathtype"])
for i in np.arange(0,len(typename),1):
    ax.hist(data["pho_et"][data["type"]==1][data["pathtype"]==i],bins=np.linspace(min,max,nbins),histtype="step", label = typename[i] + " Signal",color = "blue", alpha=opaque[i])
    ax.hist(data["pho_et"][data["type"]==0][data["pathtype"]==i],bins=np.linspace(min,max,nbins),histtype="step", label = typename[i] + " Background", color= "red", alpha = opaque[i])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig.savefig(output_dir+args.tag+" type_et.png")

print("Plottign et for truth type")
fig,ax = plt.subplots()
truthtypes = np.unique(data["pho_truthType"])
for i in truthtypes:
    if len(data["pho_et"][data["type"]==1][data["pho_truthType"]==i])>0:
        ax.hist(data["pho_et"][data["type"]==1][data["pho_truthType"]==i],bins=np.linspace(min,max,nbins),histtype="step", label = particleType[i] + " Signal")
    if len(data["pho_et"][data["type"]==0][data["pho_truthType"]==i])>0:
        ax.hist(data["pho_et"][data["type"]==0][data["pho_truthType"]==i],bins=np.linspace(min,max,nbins),histtype="step", label = particleType[i] + " Background")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig.savefig(output_dir+args.tag+" particletype_et.png")    
print("Program done.")