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
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["muoIso", "muoPid", "eleIso", "elePid", "phoIso", "phoPid","gg", 'ee', 'mm',"eeg","mmg","Troelstest", 'Data'],
                    help = 'The choice of particle, determines what is plotted and the corresponding text')

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
if args.PartType == "Data":
    print(f"The variables are: {data.columns}")
    for i in data.columns:
        print(f"Plotting for variable: {i}")
        ndata = m.ceil(len(data[i])*0.99)
        Ndata = m.ceil(ndata*0.99)
        Data = data.nsmallest(ndata,i)
        DATA = Data.nlargest(Ndata,i)
        fig, ax = plt.subplots(figsize=(12,12))
        nbins = np.linspace(DATA[i].min(),DATA[i].max(),75)
        ax.hist(data[i],bins=nbins,histtype = "step",label = "Signal")
        ax.legend(loc="best")
        ax.set_title("Plot of the distribution of {}".format(i))
        fig.savefig(figure_dir + i)
        plt.close(fig)
elif args.PartType !="Troelstest":
    SIG = data["type"] == 1
    BKG = data["type"] == 0
    print(f"There are {sum(SIG)} signal and {sum(BKG)} background pairs.")
    DataSig = data[SIG]
    DataBkg = data[BKG]
    counter =0
    for i in data.columns:
        print(f"Plotting for variable: {i}")
        #ndata = m.ceil(len(data[i])*0.99)
        #Ndata = m.ceil(ndata*0.99)
        #Data = data.nsmallest(ndata,i)
        #DATA = Data.nlargest(Ndata,i)
        fig, ax = plt.subplots(figsize=(12,12))
        nbins = np.linspace(data[i].min(),data[i].max(),75)
        ax.hist(data[i][SIG],bins=nbins,histtype = "step",label="Signal")
        ax.hist(data[i][BKG],bins=nbins,histtype = "step",label = "Background")
        ax.legend(loc="best")
        if i=="invM":
            ax.set_yscale("log")
        ax.set_title("Plot of the distribution of {}".format(i))
        fig.savefig(figure_dir + i)
        plt.close(fig)

else:
    print(f"The variables are: {data.columns}")
    for i in data.columns:
        SIG = data["type"] == 1
        BKG = data["type"] == 0
        print(f"Plotting for variable: {i}")
        #ndata = m.ceil(len(data[i])*0.99)
        #Ndata = m.ceil(ndata*0.99)
        #Data = data.nsmallest(ndata,i)
        #DATA = Data.nlargest(Ndata,i)
        fig, ax = plt.subplots(figsize=(12,12))
        nbins = np.linspace(data[i].min(),data[i].max(),75)
        ax.hist(data[i][SIG],bins=nbins,histtype = "step",label = "Signal")
        ax.hist(data[i][BKG],bins=nbins,histtype = "step",label = "Background")
        ax.legend(loc="best")
        ax.set_title("Plot of the distribution of {}".format(i))
        fig.savefig(figure_dir + i)
        plt.close(fig)


"""
if morebkg:
    print(f"There are {DataSig.shape[0]} signal and {DataBkg.shape[0]} background before additional background")
    while DataBkg.shape[0]/DataSig.shape[0]<0.9:
        counter +=1
        #print(counter)
        databkg = data[BKG]
        DataBkg = pd.concat([DataBkg,databkg], ignore_index=True)
        #print(f"There are now {DataBkg.shape[0]} background events after appending")
        if counter%5==0:
            print(f"Counter for additional background: {counter}")
        if counter == 100:
            break

print(f"There are {DataSig.shape[0]} signal and {DataBkg.shape[0]} background")
#log.info("Variable: {}, min: {}, max: {}".format(i,data[i].min(), data[i].max()))

fig, ax = plt.subplots()
if (args.PartType == "ee")or(args.PartType == "mm") or (args.PartType == "gg")or(args.PartType == "eeg") or (args.PartType == "mmg"):
    if args.PartType == "ee":
        ax.set_title(r"Z$\rightarrow$ee Invariant mass")
    elif args.PartType =="mm":
        ax.set_title(r"Z$\rightarrow \mu\mu$ Invariant Mass")
    elif args.PartType =="gg":
        ax.set_title(r"H$\rightarrow\gamma\gamma$ Invariant Mass")
    elif args.PartType == "eeg":
        ax.set_title(r"$Z\rightarrow ee\gamma$ Invariant Mass")
    elif args.PartType =="mmg":
        ax.set_title(r"$Z\rightarrow \mu\mu\gamma$ Invariant Mass")
    ax.set_xlabel(r"invM (GeV)")
    ax.set_ylabel("Frequency")

    # Add Values to histogram 
    if args.PartType == "eeg":
        ax.hist(DataSig["invM"],bins=np.linspace(0,300,301),histtype="step", label = "Signal")
        ax.hist(DataBkg["invM"],bins=np.linspace(0,300,301),histtype="step", label = "Background")
    else:
        ax.hist(DataSig["invM"],bins=np.linspace(0,150,151),histtype="step", label = "Signal")
        ax.hist(DataBkg["invM"],bins=np.linspace(0,150,151),histtype="step", label = "Background")
    ax.legend()
    

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(figure_dir+"invM.png")
    ax.set_yscale("log")
    fig.savefig(figure_dir+"invM_log.png")
elif (args.PartType == "phoIso") or (args.PartType == "phoPid"):
    if args.PartType =="phoIso":
        ax.set_title(r"Iso Photon $E_T$")
    elif args.PartType =="phoPid":
        ax.set_title(r"Pid Photon $E_T$")
    ax.set_xlabel(r"$E_T$ (GeV)")
    ax.set_ylabel("Frequency")

    # Add Values to histogram 
    ax.hist(DataSig["pho_et"],bins=np.linspace(0,100,101),histtype="step", label = "Signal")
    ax.hist(DataBkg["pho_et"],bins=np.linspace(0,100,101),histtype="step", label = "Background")
    ax.legend()
    

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(figure_dir+"et.png")
    ax.set_yscale("log")
    fig.savefig(figure_dir+"et_log.png")
elif (args.PartType == "eleIso") or (args.PartType == "elePid"):
    if args.PartType =="eleIso":
        ax.set_title(r"Iso Electron $E_T$")
    elif args.PartType =="elePid":
        ax.set_title(r"Pid Electron $E_T$")
    ax.set_xlabel(r"$E_T$ (GeV)")
    ax.set_ylabel("Frequency")

    # Add Values to histogram 
    ax.hist(DataSig["ele_et"],bins=np.linspace(0,100,101),histtype="step", label = "Signal")
    ax.hist(DataBkg["ele_et"],bins=np.linspace(0,100,101),histtype="step", label = "Background")
    ax.legend()
    

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(figure_dir+"et.png")
    ax.set_yscale("log")
    fig.savefig(figure_dir+"et_log.png")
elif (args.PartType == "muoIso") or (args.PartType == "muoPid"):
    if args.PartType =="muoIso":
        ax.set_title(r"Iso Muon $P_T$")
    elif args.PartType =="muoPid":
        ax.set_title(r"Pid Muon $P_T$")
    ax.set_xlabel(r"$P_T$ (GeV)")
    ax.set_ylabel("Frequency")

    # Add Values to histogram 
    ax.hist(DataSig["muo_pt"],bins=np.linspace(0,100,101),histtype="step", label = "Signal")
    ax.hist(DataBkg["muo_pt"],bins=np.linspace(0,100,101),histtype="step", label = "Background")
    ax.legend()
    

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(figure_dir+"pt.png")
    ax.set_yscale("log")
    fig.savefig(figure_dir+"pt_log.png")
"""

print(f'')
print(f'')
print(f"END OF PROGRAM")
