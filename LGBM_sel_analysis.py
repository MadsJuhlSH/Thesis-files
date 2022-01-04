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
import seaborn as sns

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
parser.add_argument('--outdir', action='store', default="output/LGBM_sel_analysis/", type=str,
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
output_dir = args.outdir + "LGBMAnalysis" + args.tag + "_" + timelabel + "/"
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
print(f'Datafile:                 {args.path}')
print(f'Output directory:         {output_dir}')
print(f'Number of threads:        {args.njobs}')
print(f'------------------------------------------------------------------')
#%%############################################################################
#   Importing data.
###############################################################################
header("Importing and separating data")
t = time()

# Get hdf5 datafile as dataframe
data = h5ToDf(args.path)
SIG = data["type"] == 1
BKG = data["type"] == 0


print(f"There are {np.sum(SIG)} signal and {np.sum(BKG)} background")
nbins = 100
print("plotting selected variables")
fig, ax = plt.subplots(2,2,figsize=(12,12))
ax[0,0].hist(data["invM"][SIG],bins=np.linspace(1,151,150), histtype="step", label ="Signal")
ax[0,0].hist(data["invM"][BKG],bins=np.linspace(1,151,150),histtype = "step",label = "Background")
ax[0,0].set_title("invM")
ax[0,0].legend(loc="best")
ax[0,1].hist(data["Zee_score"][SIG],bins=nbins, histtype="step", label ="Signal")
ax[0,1].hist(data["Zee_score"][BKG],bins=nbins,histtype = "step",label = "Background")
ax[0,1].set_title("Zee_score")
ax[0,1].legend(loc="best")
ax[0,1].set_yscale("log")
ax[1,0].hist(data["pho_pIso_score"][SIG],bins=nbins, histtype="step", label ="Signal")
ax[1,0].hist(data["pho_pIso_score"][BKG],bins=nbins,histtype = "step",label = "Background")
ax[1,0].set_title("pho_pIso_score")
ax[1,0].legend(loc="best")
ax[1,0].set_yscale("log")
ax[1,1].hist(data["pho_pPid_score"][SIG],bins=nbins, histtype="step", label ="Signal")
ax[1,1].hist(data["pho_pPid_score"][BKG],bins=nbins,histtype = "step",label = "Background")
ax[1,1].set_title("pho_pPid_score")
ax[1,1].legend(loc="best")
ax[1,1].set_yscale("log")

fig.savefig(figure_dir+"_Mass_LGBMscores")
plt.close(fig)
"""
ZeeSig = data["Zee_score"] >=0 
ZeeBkg = data["Zee_score"] <0
print("Plotting pIso and pPid based on Zee-score")
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].hist(data["pho_pIso_score"][ZeeSig],bins=nbins, histtype="step", label ="Signal")
ax[0].hist(data["pho_pIso_score"][ZeeBkg],bins=nbins,histtype = "step",label = "Background")
ax[0].set_title("pho_pIso_score")
ax[0].legend(loc="best")
ax[0].set_yscale("log")
ax[1].hist(data["pho_pPid_score"][ZeeSig],bins=nbins, histtype="step", label ="Signal")
ax[1].hist(data["pho_pPid_score"][ZeeBkg],bins=nbins,histtype = "step",label = "Background")
ax[1].set_title("pho_pPid_score")
ax[1].legend(loc="best")
ax[1].set_yscale("log")

fig.savefig(figure_dir+"Pho_score_from_zee_score")
plt.close(fig)
print("Plotting Zee-score based on pIso and pPid score")
pPidSig = data["pho_pIso_score"] >=0 
pPidBkg = data["pho_pIso_score"] <0
pIsoSig = data["pho_pPid_score"] >=0 
pIsoBkg = data["pho_pPid_score"] <0
fig, ax = plt.subplots(1,2,figsize=(12,6))

ax[0].hist(data["Zee_score"][pPidSig],bins=nbins, histtype="step", label ="Signal")
ax[0].hist(data["Zee_score"][pPidBkg],bins=nbins,histtype = "step",label = "Background")
ax[0].set_title("Pho Pid")
ax[0].legend(loc="best")
ax[0].set_yscale("log")
ax[1].hist(data["Zee_score"][pIsoSig],bins=nbins, histtype="step", label ="Signal")
ax[1].hist(data["Zee_score"][pIsoBkg],bins=nbins,histtype = "step",label = "Background")
ax[1].set_title("Pho Iso")
ax[1].legend(loc="best")
ax[1].set_yscale("log")
fig.savefig(figure_dir+"Zee_score_from_pIsopPid")
plt.close(fig)
"""
print("Scatter plot of invM and pIso")
massCut = np.multiply(data["invM"]>50,data["invM"]<150)
fig = sns.jointplot(x=data["invM"][massCut],y=data["pho_pIso_score"][massCut],hue=data["type"],kind="kde")
fig.set_axis_labels("Invariant mass (GeV)", "Logit transformed prediction of photon isolation")
fig.savefig(figure_dir+f"invMvsPIso")

print("Scatter plot of pPid and pIso")


fig = sns.jointplot(x=data["pho_pPid_score"],y=data["pho_pIso_score"],hue=data["type"],kind="kde")
fig.set_axis_labels("Logit transformed prediction of photon identification", "Logit transformed prediction of photon isolation")
fig.savefig(figure_dir+f"PPidvsPIso")

print("Transverse energy of photons and their ratio (bkg/sig)")
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].hist(data["pho_et"][SIG],bins=np.linspace(10,60,51), histtype="step", label ="Signal")
ax[0].hist(data["pho_et"][BKG],bins=np.linspace(10,60,51),histtype = "step",label = "Background")
ax[0].set_title("Transverse energy of photon")
ax[0].legend(loc="best")
ax[0].set_yscale("log")
valsig,binsedge = np.histogram(data["pho_et"][SIG],bins=np.linspace(10,60,51))
valbkg,binsedge = np.histogram(data["pho_et"][BKG],bins=np.linspace(10,60,51))
ax[1].bar(binsedge[:-1],valbkg/valsig)
ax[1].set_title("Ratio of signal and background transverse energy of the photons")
#ax[1].legend(loc="best")
fig.savefig(figure_dir+"ratio_of_pho_et")
print(f'')
print(f'')
print(f"END OF PROGRAM")