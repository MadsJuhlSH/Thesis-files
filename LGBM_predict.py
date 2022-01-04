#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using LightGBM to train a boosted decision tree to identify signal
and background based on variables.

This program makes predictions based on a trained model and creates plots.
"""
print("Program running...")

# Import packages and functions
#from Pho_search.Compare_roc import PartType
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
import math as m

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from scipy.special import logit
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error

from utils import timediftostr, Build_Folder_Structure, header, footer, print_col_names, h5ToDf, accuracy, HyperOpt_RandSearch, auc_eval


print("Packages and functions loaded")
t_total_start = time()

weightNames = [ "regWeight_nEst10", "regWeight_nEst40", "regWeight_nEst100", "revWeight_nEst10", "revWeight_nEst40", "revWeight_nEst100" ]

weightsString = "Choose weights:"
for iWeight, weight in enumerate(weightNames):
    weightsString = weightsString + f" {iWeight} = {weight},"


#%%############################################################################
#   Parser
###############################################################################
parser = argparse.ArgumentParser(description='Electron isolation with the LightGBM framework.')
parser.add_argument('paths', type=str, nargs='+',
                    help='Data file(s) to predict from.')
parser.add_argument('--outdir', action='store', type=str, default="output/Training/predictions/FINAL/",
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--model', action='store', type=str, required=True,
                    help='Input model: What directory to load data from.')
parser.add_argument('--weights', action='store', type=int, required=True,
                    help=f'Choose weights: 0 = {weightNames[0]}, 1 = {weightNames[1]}, 2 = {weightNames[2]}')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["eeg", "mmg", "muoIso", "muoPid", "eleIso", "elePid", "phoIso", "phoPid", 'ee', 'mm', 'gg'],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')

args = parser.parse_args()

log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "eleIso") +(args.PartType != "elePid") + (args.PartType != "muoIso") +(args.PartType != "muoPid")+ (args.PartType != "eeg") + (args.PartType != "mmg") + (args.PartType != "phoIso")+(args.PartType != "phoPid")+(args.PartType != "ee"))+(args.PartType != "mm")+(args.PartType !="gg")!=10:
    log.error("Unknown lepton, use either eleIso, elePid, muoIso, muoPid, eeg, mmg, ee, mm, gg, phoPid or phoIso")
    quit()

#%%############################################################################
#   Functions
###############################################################################
def rocVars(predColumn, setMask):
    true = data.loc[setMask][truth_var]
    predict = data.loc[setMask][predColumn]
    fpr, tpr, thresholds = roc_curve(true, predict, sample_weight=data.loc[setMask][weightName])
    aucCalc = auc(fpr, tpr)
    return fpr, tpr, aucCalc
def getSameFpr(fprArray, tprArray, fprGoal, thresholds):
    # Round fpr to compare
    fprMask = (np.around(fprArray,decimals=4) == np.around(fprGoal,decimals=4))

    # If round to 4 decimals does not give any results round to 3 or 2 decimals
    if np.sum(fprMask) == 0:
        fprMask = (np.around(fprArray,decimals=3) == np.around(fprGoal,decimals=3))
    if np.sum(fprMask) == 0:
        fprMask = (np.around(fprArray,decimals=2) == np.around(fprGoal,decimals=2))
    if np.sum(fprMask) == 0:
        fprMask = (np.around(fprArray,decimals=1) == np.around(fprGoal,decimals=1))
        print(f"FprMask only has one decimal!")
    
    # Possible fpr and tpr values
    fprChosen = fprArray[fprMask]
    tprChosen = tprArray[fprMask]
    thresholdsChosen = thresholds[fprMask]

    # Number of possible fpr values to choose from
    nfprMask = np.sum(fprMask)

    # Calculate difference between the possible fpr and the goal fpr
    fprDiff = fprChosen - fprGoal

    # Choose index: More than one possibility
    if nfprMask>1:
        # If there all possible fpr are the same, choose half way point
        if np.sum(fprDiff)==0:
            idx = int(nfprMask/2) # Half way point
        # If the possible fpr are not the same, get minimum difference
        else:
            idx = np.argmin(np.abs(fprDiff))
    # Choose index: Only one possibility
    else:
        idx = 0
    #print(f"fprMask shape: {[len(a) for a in fprMask]}")
    return fprMask, idx, fprChosen[idx], tprChosen[idx], thresholdsChosen[idx]


#%%############################################################################
#   Filenames and directories
###############################################################################
# Name of data file
filename_base = os.path.basename(args.paths[0])
filename_split = os.path.splitext(filename_base)[0]
filename = filename_split.split("_")[1]
fileTypes = ""
data_list = []

log.info(f"Importing files and retreiwing names")
for path in args.paths:
    # Get filetype
    filename_base = os.path.basename(path)
    filename_split = os.path.splitext(filename_base)[0]
    filetype = filename_split.split("_")[-1]
    fileTypes = fileTypes + filetype

    # Data
    data_get = h5ToDf(path)
    data_list.append(data_get)

# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"
# timelabel = f"{datetime.date(now)}_{int(timestamp)}"

# Get weight name and type
weightName = weightNames[args.weights]


# Create names for output
if args.tag:
    output_dir = args.outdir + args.PartType +"_LGBM_" + filename + "_" + weightName + "_" + args.tag + "_" + timelabel + "_" + fileTypes + "/"
    figname = args.PartType +"_LGBM_" + filename + "_" + weightName + "_" + args.tag + "_" + timelabel + "_" + fileTypes + "_"
elif not args.tag:
    output_dir = args.outdir +args.PartType + "_LGBM_" + filename + "_" + weightName + "_" + timelabel + "_" + fileTypes + "/"
    figname = args.PartType +"_LGBM_" + filename + "_" + weightName + "_" + timelabel + "_" + fileTypes + "_"

# Check if output folder already exists
if os.path.exists(output_dir):
    log.error(f"Output already exists - please remove yourself. Output dir: {output_dir}")
    quit()

# Build the folder structure
Build_Folder_Structure([output_dir])


#%%############################################################################
#   Set parameters
###############################################################################
# Set random seed
np.random.seed(seed=42)

# Define variables needed
truth_var = "type"
if args.PartType == "gg":
    training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                #'pho_deltad0',
                #'pho_deltad0Sig',
                'pho_deltaZ0',
                'pho_deltaZ0sig',
                'pho1_pIso_score',
                'pho1_pPid_score',
                'pho2_pIso_score',
                'pho2_pPid_score'
                ]
elif args.PartType == "eeg":
    training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                #'ele1_d0',
                #'ele2_d0',
                #'ele1_d0Sig',
                #'ele2_d0Sig',
                #'ele1_ECIDSResult',
                #'ele2_ECIDSResult',
                #'ele1_charge',
                #'ele2_charge',
                #'ele1_ePid_score',
                #'ele2_ePid_score',
                #'ele1_eIso_score',
                #'ele2_eIso_score',
                'ee_score',
                'pho_pIso_score',
                'pho_pPid_score']
    """training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                'ele1_d0',
                'ele2_d0',
                'ele1_d0Sig',
                'ele2_d0Sig',
                'ele1_ECIDSResult',
                'ele2_ECIDSResult',
                'ele1_charge',
                'ele2_charge',
                'ele1_ePid_score',
                'ele2_ePid_score',
                'ele1_eIso_score',
                'ele2_eIso_score',
                'pho_pIso_score',
                'pho_pPid_score']"""
elif args.PartType == "mmg":
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    #'muo1_charge',
                    #'muo2_charge',
                    #'pho_z0',
                    #'muo1_delta_z0',
                    #'muo2_delta_z0',
                    #'muo1_vertex_z',
                    #'pho_isConv'
                    #'muo1_mPid_score',
                    #'muo2_mPid_score',
                    #'muo1_mIso_score',
                    #'muo2_mIso_score',
                    'mm_score',
                    'pho_pIso_score',
                    'pho_pPid_score']
    """training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'muo1_charge',
                    'muo2_charge',
                    'pho_z0',
                    'muo1_delta_z0',
                    'muo2_delta_z0',
                    'muo1_vertex_z',
                    #'pho_isConv'
                    'muo1_mPid_score',
                    'muo2_mPid_score',
                    'muo1_mIso_score',
                    'muo2_mIso_score',
                    'pho_pIso_score',
                    'pho_pPid_score']"""
elif args.PartType == "ee":
    training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                'ele_deltad0',
                'ele_deltad0sig',
                #'ele1_d0',
                #'ele2_d0',
                #'ele1_d0Sig',
                #'ele2_d0Sig',
                "ele_deltaZ0",
                "ele_deltaZ0sig",
                #'ele1_ECIDSResult',
                #'ele2_ECIDSResult',
                #'ele1_charge',
                #'ele2_charge',
                'ele1_ePid_score',
                'ele2_ePid_score',
                'ele1_eIso_score',
                'ele2_eIso_score']
elif args.PartType == "mm":
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'muo1_delta_z0',
                    'muo2_delta_z0',
                    'muo1_delta_z0_sin_theta',
                    'muo2_delta_z0_sin_theta',
                    'muo1_vertex_z',
                    'muo2_vertex_z',
                    #'muo1_charge',
                    #'muo2_charge',
                    'muo1_mPid_score',
                    'muo2_mPid_score',
                    'muo1_mIso_score',
                    'muo2_mIso_score']
elif args.PartType == "eleIso": 
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'ele_ptvarcone20_rel',
                    'ele_ptvarcone40_rel',
                    'ele_topoetcone20_rel',
                    'ele_topoetcone40_rel',
                    'ele_topoetcone20ptCorrection',
                    ]
    
    """training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'ele_ptvarcone20_rel',
                    'ele_ptvarcone40_rel',
                    'ele_topoetcone20_rel',
                    'ele_topoetcone40_rel',
                    'ele_expectInnermostPixelLayerHit',
                    'ele_expectNextToInnermostPixelLayerHit',
                    'ele_core57cellsEnergyCorrection',
                    'ele_nTracks',
                    'ele_numberOfInnermostPixelHits',
                    'ele_numberOfPixelHits',
                    'ele_numberOfSCTHits',
                    'ele_numberOfTRTHits',
                    'ele_topoetcone20ptCorrection',
                    ]"""
elif args.PartType == "elePid":
    training_var = ['ele_d0',
                    'ele_d0Sig',
                    'ele_Rhad1',
                    'ele_Rhad',
                    'ele_f3',
                    'ele_weta2',
                    'ele_Rphi',
                    'ele_Reta',
                    'ele_Eratio',
                    'ele_f1',
                    'ele_dPOverP',
                    'ele_deltaEta1',
                    'ele_deltaPhiRescaled2',
                    'ele_expectInnermostPixelLayerHit',
                    'ele_expectNextToInnermostPixelLayerHit',
                    'ele_core57cellsEnergyCorrection',
                    'ele_nTracks',
                    'ele_numberOfInnermostPixelHits',
                    'ele_numberOfPixelHits',
                    'ele_numberOfSCTHits',
                    'ele_numberOfTRTHits',
                    'ele_TRTPID'
                    ]
    """training_var = ['ele_d0',
                    'ele_d0Sig',
                    'ele_Rhad1',
                    'ele_Rhad',
                    'ele_f3',
                    'ele_weta2',
                    'ele_Rphi',
                    'ele_Reta',
                    'ele_Eratio',
                    'ele_f1',
                    'ele_dPOverP',
                    'ele_deltaEta1',
                    'ele_deltaPhiRescaled2',
                    'ele_TRTPID'
                    ]"""
elif args.PartType == "muoIso":
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    #'muo_etcon20',
                    'muo_ptvarcone20_rel',
                    'muo_ptvarcone40_rel',
                    'muo_etconecoreConeEnergyCorrection'
                    #'muo_topoetconecoreConeEnergyCorrection',
                    ]
elif args.PartType == "muoPid":
    training_var = ['muo_priTrack_d0',
                    'muo_priTrack_d0Sig',
                    'muo_numberOfPrecisionHoleLayers', 
                    'muo_numberOfPrecisionLayers',
                    'muo_quality',
                    #'muo_ET_TileCore',
                    'muo_MuonSpectrometerPt',
                    #'muo_deltatheta_1',
                    'muo_scatteringCurvatureSignificance', 
                    'muo_scatteringNeighbourSignificance',
                    'muo_momentumBalanceSignificance',
                    'muo_EnergyLoss',
                    'muo_energyLossType',
                    #'muo_priTrack_numberOfPixelHits',
                    #'muo_priTrack_numberOfSCTHits',
                    #'muo_priTrack_numberOfTRTHits'
                    ]
elif args.PartType == "phoIso":
    training_var = ['correctedScaledAverageMu',
                    'NvtxReco',
                    'pho_et',
                    'pho_ptvarcone20',
                    'pho_topoetcone20',
                    'pho_topoetcone40']
elif args.PartType == "phoPid":
    training_var = ['pho_isPhotonEMLoose',
                    'pho_isPhotonEMTight',
                    'pho_Rhad1',
                    'pho_Rhad',
                    'pho_weta2',
                    'pho_Rphi',
                    'pho_Reta',
                    'pho_Eratio',
                    'pho_f1',
                    'pho_wtots1',
                    'pho_DeltaE',
                    'pho_weta1',
                    'pho_fracs1',
                    #'pho_ConversionType',
                    'pho_ConversionRadius',
                    'pho_VertexConvEtOverPt',
                    'pho_VertexConvPtRatio',
                    'pho_z0',
                    'pho_z0Sig',
                    #'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']
"""['pho_Rhad',
'pho_Rhad1',
'pho_Reta',
'pho_weta2',
'pho_Rphi',
'pho_wtots1',
'pho_Eratio',
'NvtxReco',
'correctedScaledAverageMu',
'pho_ConversionRadius',
'pho_ConversionType',
'pho_f1',
'pho_r33over37allcalo']
"""
# Set dataset type: 0 = train, 1 = valid, 2 = test
datatype = {0 : "train",
            1 : "valid",
            2 : "test",
            3 : "store"}
datatypePrint = {0 : "Training",
                 1 : "Validation",
                 2 : "Test",
                 3 : "Saved"}

# Define particleType
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
particleColor = {0 : 'C3',
                 1 : 'C1',
                 2 : 'C0',
                 3 : 'C2',
                 4 : 'C4',
                 5 : 'C0',
                 6 : 'C1',
                 7 : 'C2',
                 8 : 'C3',
                 13: 'C0',
                 14: 'C1',
                 15: 'C2',
                 16: 'C3'}

#%%############################################################################
#   Print initial log info
###############################################################################
# Log INFO for run
print(f'')
print(f'---------------------------- INFO --------------------------------')
for i, path in enumerate(args.paths):
    if i==0:
        print(f'Datafile(s):       {path}')
    else:
        print(f'                   {path}')
print(f'Output directory:         {output_dir}')
print(f'Model:                    {args.model}')
print(f'------------------------------------------------------------------')


#%%############################################################################
#   Importing data.
###############################################################################
header("Importing data")
t = time()

# # Get hdf5 datafile as dataframe
# data = h5ToDf(args.path)

# Concatenate data
data = pd.concat(data_list, ignore_index=True)

if args.PartType == "eleIso":
    data['ele_atlasIso'] = ( data['ele_ptvarcone20'] / data['ele_e'] < 0.15 ) & ( data['ele_topoetcone20'] / data['ele_e'] < 0.20 )

# Print column names to log
print_col_names(data.columns,training_var,truth_var)

# Separate label and training data
X = data[training_var][:]
y = data[truth_var][:]

# Get lists of unique dataset types and truth types
datasetTypes = np.sort(data["dataset"].unique())
if (args.PartType == "eleIso") or (args.PartType == "elePid"):
    truthTypes = np.sort(data["ele_truthType"].unique())
    truthTypes = truthTypes[truthTypes <5]
    print(truthTypes)

    # Print number of Particles
    print(f"Number of Particles by datatype:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}: {np.sum(data['dataset']==setType)}")
    print()
    print(f"Number of Particles by truthType:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}:")
        for trueType in truthTypes:
            print(f"        {particleType[trueType]}: {len(data.loc[(data['ele_truthType']==trueType) & (data['dataset']==setType)])}")
        print()
elif (args.PartType == "muoIso") or (args.PartType == "muoPid"):
    truthTypes = np.sort(data["muo_truthType"].unique())
    truthTypes = truthTypes[(truthTypes >4) ]#& (truthTypes < 9) ]
    print(truthTypes)

    # Print number of Particles
    print(f"Number of Particles by datatype:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}: {np.sum(data['dataset']==setType)}")
    print()
    print(f"Number of Particles by truthType:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}:")
        for trueType in truthTypes:
            print(f"        {particleType[trueType]}: {len(data.loc[(data['muo_truthType']==trueType) & (data['dataset']==setType)])}")
        print()
elif (args.PartType == "phoIso") or (args.PartType == "phoPid"):
    truthTypes = np.sort(data["pho_truthType"].unique())
    truthTypes = truthTypes[(truthTypes >12)]# & (truthTypes < 17) ]
    print(truthTypes)

    # Print number of Particles
    print(f"Number of Particles by datatype:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}: {np.sum(data['dataset']==setType)}")
    print()
    print(f"Number of Particles by truthType:")
    for setType in datasetTypes:
        print(f"    {datatype[setType]}:")
        for trueType in truthTypes:
            print(f"        {particleType[trueType]}: {len(data.loc[(data['pho_truthType']==trueType) & (data['dataset']==setType)])}")
        print()
elif (args.PartType == "eeg") or (args.PartType == "ee") or (args.PartType == "mmg") or (args.PartType == "mm") or (args.PartType =="gg"):
    # Get list of unique dataset types
    truthTypes = np.sort(data["type"].unique())

    nTot = len(data['type'])
    nSig = len(data.loc[(data['type']==1)])
    nBkg = len(data.loc[(data['type']==0)])
    print(f"Number of events in test sample: {nTot}")
    print(f"        Signal:     {nSig} ({nSig/nTot*100:.2f}%)")
    print(f"        Background: {nBkg} ({nBkg/nTot*100:.2f}%)")



# Import model
gbm = lgb.Booster(model_file = args.model)

footer("Importing data",t)


#%%############################################################################
#   Predict on training set, validation set and all data and save scores,
#   or load pre-predicted scores
###############################################################################
header("Running Prediction")
t = time()

# Predict on all data and save variables in dataframe
print("Predicting on data...")
y_pred_data = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration)
data["predLGBM"] = y_pred_data

# Signal selection on non logit transformed
sigSel = 0.5

# Select signal
print(f"Make LGBM selection with non logit transformed cut: {sigSel}\n")
data["selLGBM"] = 0
data.loc[(y_pred_data>sigSel), ["selLGBM"]] = 1

# Print AUC score
for setType in datasetTypes:
    print(f"Prediction on {datatype[setType]} set:")
    if setType == 3:
        print(f"        Set does not have labels. Continue...")
        continue
    setMask = data["dataset"]==setType
    nEvents = data[setMask].shape[0]

    aucScore = roc_auc_score(data[truth_var][setMask], data['predLGBM'][setMask])
    aucScore_weight = roc_auc_score(data[truth_var][setMask], data['predLGBM'][setMask], sample_weight=data[weightName][setMask])
    accScore = np.sum(data['type'][setMask] == data['selLGBM'][setMask]) / nEvents

    print(f"        AUC score of prediction on {datatype[setType]}:            {aucScore:.6f}")
    print(f"        AUC score of prediction on {datatype[setType]} (Weighted): {aucScore_weight:.6f}")
    print(f"        Accuracy score of prediction on {datatype[setType]}:       {accScore * 100:.4f}%   (With signal selection: {sigSel})")
    print(f"")


footer("Predicting",t)


# %%############################################################################
#   Plot
# ##############################################################################
header('Plotting results')

######## LightGBM prediction ########
plt_range = [(0,1),(-10,10)]
bins = 100

for setType in datasetTypes:
    setMask = data["dataset"]==setType

    print(f"Plotting prediction of background and signal for {datatype[setType]}")
    fig, ax = plt.subplots(2,2, figsize=(8,5), sharex='col')
    for i in range(2):
        for j in range(2):
            plt_sig = data.loc[(data[truth_var]==1) & setMask]['predLGBM']
            plt_bkg = data.loc[(data[truth_var]==0) & setMask]['predLGBM']
            line = sigSel
            if j == 1:
                plt_sig = logit(plt_sig)
                plt_bkg = logit(plt_bkg)
                line = logit(sigSel)

            ax[i,j].hist(plt_sig,bins=bins, range=plt_range[j], label="Signal", histtype='step', color="C0")
            ax[i,j].hist(plt_bkg,bins=bins, range=plt_range[j], label="Background", histtype='step', color="C3")
            # ax[i,j].axvline(line, label="Selection cut", color="k")

        ax[1,i].set_yscale("log")
        ax[1,i].set_ylim(10**0/2, 10**6)
    ax[1,0].set_xlabel(f"LGBM prediction")
    ax[1,1].set_xlabel(f"Logit(LGBM prediction)")
    ax[0,0].set_ylabel(f"Frequency - non log scale")
    ax[1,0].set_ylabel(f"Frequency - log scale")

    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",ncol=4)
    fig.suptitle(f"LGBM prediction - {datatypePrint[setType]}")

    fig.tight_layout(rect=[0,0.05,1,0.95], h_pad=0.3, w_pad=0.3)
    fig.savefig(output_dir + f"LGBM_prediction_{datatype[setType]}.png")

    print(f"Plotting prediction of truth types for {datatype[setType]}")
    fig, ax = plt.subplots(2,2, figsize=(8,5), sharex='col')
    for trueType in truthTypes:
        for i in range(2):
            for j in range(2):
                if setType == 3:
                    if (args.PartType == "eleIso") or (args.PartType == "elePid"):
                        prediction = data.loc[(data['ele_truthType']==trueType)]['predLGBM']
                    elif (args.PartType == "muoIso") or (args.PartType == "muoPid"):
                        prediction = data.loc[(data['muo_truthType']==trueType)]['predLGBM']
                    elif (args.PartType == "phoIso") or (args.PartType == "phoPid"):
                        prediction = data.loc[(data['pho_truthType']==trueType)]['predLGBM']
                    elif (args.PartType == "eeg")or (args.PartType == "ee") or (args.PartType == "mmg") or (args.PartType == "mm")or (args.PartType == "gg"):
                        prediction = data.loc[(data['type']==1)]['predLGBM']
                    setTypePrint = ""
                    for iVal, val in enumerate(datasetTypes):
                        if iVal==0:
                            setTypePrint = datatypePrint[val]
                        else:
                            setTypePrint = setTypePrint + " and " + datatypePrint[val]
                    setTypePrint = setTypePrint+" set"
                else:
                    if (args.PartType == "eleIso") or (args.PartType == "elePid"):
                        prediction = data.loc[(data['ele_truthType']==trueType)& setMask]['predLGBM']
                    elif (args.PartType == "muoIso") or (args.PartType == "muoPid"):
                        prediction = data.loc[(data['muo_truthType']==trueType)& setMask]['predLGBM']
                    elif (args.PartType == "phoIso") or (args.PartType == "phoPid"):
                        prediction = data.loc[(data['pho_truthType']==trueType)& setMask]['predLGBM']
                    elif (args.PartType == "eeg") or (args.PartType == "ee") or (args.PartType == "mmg") or (args.PartType == "mm")or (args.PartType == "gg"):
                        prediction = data.loc[(data['type']==1)]['predLGBM']
                    setTypePrint = datatypePrint[setType]+" set"

                if j == 1:
                    prediction = logit(prediction)

                ax[i,j].hist(prediction, bins=bins, range=plt_range[j], label=particleType[trueType], histtype='step') #, color=particleColor[trueType])

            ax[1,i].set_yscale("log")
            ax[1,i].set_ylim(10**0/2, 10**6)
    ax[1,0].set_xlabel(f"LGBM prediction")
    ax[1,1].set_xlabel(f"Logit(LGBM prediction)")
    ax[0,0].set_ylabel(f"Frequency - non log scale")
    ax[1,0].set_ylabel(f"Frequency - log scale")

    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",ncol=4)
    fig.suptitle(f"LGBM prediction - {setTypePrint}")

    fig.tight_layout(rect=[0,0.05,1,0.95], h_pad=0.3, w_pad=0.3)
    fig.savefig(output_dir + f"LGBM_prediction_{setTypePrint}_truthTypes.png")


print(f"Plotting prediction of background and signal combined")
fig, ax = plt.subplots(2,2, figsize=(8,5), sharex='col')
linestyles = ["solid", "dashed"]
for i in range(2):
    for j in range(2):
        fig_sub, ax_sub = plt.subplots(figsize=(6,3))
        for iSet, setType in enumerate(datasetTypes):
            setMask = data["dataset"]==setType
            plt_sig = data.loc[(data[truth_var]==1) & setMask]['predLGBM']
            plt_bkg = data.loc[(data[truth_var]==0) & setMask]['predLGBM']
            line = sigSel
            if j == 1:
                plt_sig = logit(plt_sig)
                plt_bkg = logit(plt_bkg)
                line = logit(sigSel)

            ax[i,j].hist(plt_sig,bins=bins, range=plt_range[j], linestyle=linestyles[iSet], label=f"{datatypePrint[setType]} set: Signal", histtype='step', color="C0")
            ax[i,j].hist(plt_bkg,bins=bins, range=plt_range[j], linestyle=linestyles[iSet], label=f"{datatypePrint[setType]} set: Background", histtype='step', color="C3")
            # ax[i,j].axvline(line, label="Selection cut", color="k")

            # Subfigure
            ax_sub.hist(plt_sig,bins=bins, range=plt_range[j], linestyle=linestyles[iSet], label=f"{datatypePrint[setType]} set: Signal", histtype='step', color="C0")
            ax_sub.hist(plt_bkg,bins=bins, range=plt_range[j], linestyle=linestyles[iSet], label=f"{datatypePrint[setType]} set: Background", histtype='step', color="C3")

        # Subfigure
        if j==1:
            ax_sub.set_ylabel(f"Frequency - log scale")
            ax_sub.set_yscale("log")
            ax_sub.set_ylim(10**0/2, 10**6)
        else:
            ax_sub.set_ylabel(f"Frequency")
        if i == 1:
            ax_sub.set_xlabel(f"LGBM prediction")
        else:
            ax_sub.set_xlabel(f"Logit(LGBM prediction)")
        ax_sub.set_title("LGBM prediction")
        handles_sub, labels_sub = ax_sub.get_legend_handles_labels()
        fig_sub.legend(handles_sub, labels_sub, loc="lower center",ncol=2)
        fig_sub.tight_layout(rect=[0,0.15,1,1])
        fig_sub.savefig(output_dir + f"LGBM_prediction_combined{i}{j}.png")

        ax[1,i].set_yscale("log")
        ax[1,i].set_ylim(10**0/2, 10**6)
ax[1,0].set_xlabel(f"LGBM prediction")
ax[1,1].set_xlabel(f"Logit(LGBM prediction)")
ax[0,0].set_ylabel(f"Frequency - non log scale")
ax[1,0].set_ylabel(f"Frequency - log scale")

handles, labels = ax[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",ncol=2)
fig.suptitle(f"LGBM prediction")

fig.tight_layout(rect=[0,0.1,1,0.95], h_pad=0.3, w_pad=0.3)
fig.savefig(output_dir + f"LGBM_prediction_combined.png")

print(f"")

######## ROC curve ########
# Based on: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
print("Plotting ROC curve")

# Currently plotted with Pid, use Atlas Iso (point)
for setType in datasetTypes:
    if setType == 3:
        print(f"    AUC score {datatype[setType]}:")
        print(f"        Set does not have labels. Continue...")
        continue
    setMask = data["dataset"]==setType

    plotColumns = ['predLGBM'] #,'ele_atlasIso']
    plotNames = ['ML'] #, 'ATLAS Iso ']

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot([0, 1], [0, 1], 'k--')

    print(f"    AUC score {datatype[setType]}:")
    for iCol, column in enumerate(plotColumns):
        fprPlot, tprPlot, aucPlot = rocVars(column, setMask)
        if iCol==0:
            print(f"        {plotNames[iCol]}    {aucPlot:.4f}")
            for i in range(2):
                ax[i].plot(tprPlot, fprPlot, label=plotNames[iCol])
        else:
            print(f"        {plotNames[iCol]}    {aucPlot:.4f} (fpr = {fprPlot[1]}, tpr = {tprPlot[1]})")
            for i in range(2):
                ax[i].plot(tprPlot[1], fprPlot[1], ".", label=plotNames[iCol])

    ax[0].set_ylabel('Background efficiency \n(False positive rate)')
    ax[0].set_xlabel('Signal efficiency \n(True positive rate)')
    ax[1].set_xlabel('Signal efficiency \n(True positive rate)')

    ax[1].set_xlim(0.6, 1)
    ax[1].set_ylim(0, 1)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"ROC curve - {datatypePrint[setType]} set")

    fig.tight_layout(rect=[0,0.05,1,0.95], h_pad=0.3, w_pad=0.3)
    fig.savefig(output_dir + f"ROCweighted_{datatype[setType]}_v1.png")

for setType in datasetTypes:
    if setType == 3:
        print(f"    AUC score {datatype[setType]}:")
        print(f"        Set does not have labels. Continue...")
        continue

    setMask = data["dataset"]==setType

    plotColumns = ['predLGBM'] #,'ele_atlasIso']
    plotNames = ['ML'] #, 'ATLAS Iso ']

    fig, axes = plt.subplots(figsize=(6,4))
    # axins = axes.inset_axes([0.1, 0.4, 0.57, 0.57])
    axes.plot([0, 1], [0, 1], 'k--')

    # print(f"    AUC score {datatype[setType]}:")
    for iCol, column in enumerate(plotColumns):
        fprPlot, tprPlot, aucPlot = rocVars(column, setMask)
        if iCol==0:
            # print(f"        {plotNames[iCol]}    {aucPlot:.4f}")
            for ax in [axes]: #, axins]
                ax.plot(tprPlot, fprPlot, label=plotNames[iCol])
        else:
            # print(f"        {plotNames[iCol]}    {aucPlot:.4f} (fpr = {fprPlot[1]}, tpr = {tprPlot[1]})")
            for ax in [axes]: #, axins]:
                ax.plot(tprPlot[1], fprPlot[1], ".", label=plotNames[iCol])
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    # The following lines of codes, gives working point for ATLAS, suing their LH values.
    if args.PartType =="elePid":
        lhlfpr, lhltpr, aucLoose = rocVars('ele_LHLoose', setMask)
        lhmfpr, lhmtpr, aucMedium = rocVars('ele_LHMedium', setMask)
        lhtfpr, lhttpr, aucTight = rocVars('ele_LHTight', setMask) 
    elif args.PartType =="eleIso":
        data["ptvarcone20relcut"] = data['ele_ptvarcone20_rel']<0.15
        lhtfpr, lhttpr, aucTight = rocVars('ptvarcone20relcut', setMask) 
    elif args.PartType == "muoPid":
        lhlfpr, lhltpr, aucLoose = rocVars('muo_LHLoose', setMask)
        lhmfpr, lhmtpr, aucMedium = rocVars('muo_LHMedium', setMask)
        lhtfpr, lhttpr, aucTight = rocVars('muo_LHTight', setMask) 
    elif args.PartType =="muoIso": 
        data["ptvarcone30relcut"] = data['muo_ptvarcone30']/data['muo_pt']<0.06
        lhtfpr, lhttpr, aucTight = rocVars('ptvarcone30relcut', setMask) 
    elif (args.PartType == "phoIso") or (args.PartType == "phoPid"):
        lhlfpr, lhltpr, aucLoose = rocVars('pho_isPhotonEMLoose', setMask)
        lhtfpr, lhttpr, aucTight = rocVars('pho_isPhotonEMTight', setMask) 
    elif (args.PartType == "ee"):
        #Didnt have the following variable saved, so had to recreate it.
        #Remembered delta_z0 as the uncertainty, so used that, the other part is sin(theta) given with eta 
        data['ele1_delta_z0_sin_theta'] = data["ele1_z0"]*np.sin(2*np.arctan(np.exp(-data["ele1_eta"])))
        data['ele2_delta_z0_sin_theta'] = data["ele2_z0"]*np.sin(2*np.arctan(np.exp(-data["ele2_eta"])))
        data["ATLASmask"] = (   ( data[ 'ele1_charge']*data['ele2_charge']< 0) & 
                                                    (data["ele1_trigger"]==1)&
                                                    (data["ele1_LHLoose"]==1)&
                                                    (data["ele2_LHLoose"]==1)&
                                                    (data["ele1_et"]>10)&
                                                    (data["ele2_et"]>10)&
                                                    ( (np.abs( data["ele1_eta"]) <1.37) | ((np.abs(data['ele1_eta'])>1.52) & np.abs(data['ele1_eta'] <2.47)) ) &
                                                    ( (np.abs( data["ele2_eta"]) <1.37) | ((np.abs(data['ele2_eta'])>1.52) & np.abs(data['ele2_eta'] <2.47)) ) &
                                                    ( (np.abs(data['ele1_d0'])/data['ele1_d0Sig'])<5) & 
                                                    ( (np.abs(data['ele2_d0'])/data['ele2_d0Sig'])<5) & 
                                                    ( (np.abs(data['ele1_delta_z0_sin_theta'] <0.5))) & 
                                                    ( (np.abs(data['ele2_delta_z0_sin_theta'] <0.5))) 
                                                    # Should there be a cut in pt_varcone20? Atlas does this by removing the #% highest
                                                    )
        lhtfpr, lhttpr, aucTight = rocVars('ATLASmask', setMask)
    elif (args.PartType == "eeg"):
        data['ele1_delta_z0_sin_theta'] = data["ele1_z0"]*np.sin(2*np.arctan(np.exp(-data["ele1_eta"])))
        data['ele2_delta_z0_sin_theta'] = data["ele2_z0"]*np.sin(2*np.arctan(np.exp(-data["ele2_eta"])))
        data["ATLASmask"] = (   ( data[ 'ele1_charge']*data['ele2_charge']< 0) & 
                                                    (data["ele1_trigger"]==1)&
                                                    (data["ele1_LHLoose"]==1)&
                                                    (data["ele2_LHLoose"]==1)&
                                                    (data["ele1_et"]>10)&
                                                    (data["ele2_et"]>10)&
                                                    (data["invMll"]<82)&
                                                    ( (np.abs( data["ele1_eta"]) <1.37) | ((np.abs(data['ele1_eta'])>1.52) & np.abs(data['ele1_eta'] <2.47)) ) &
                                                    ( (np.abs( data["ele2_eta"]) <1.37) | ((np.abs(data['ele2_eta'])>1.52) & np.abs(data['ele2_eta'] <2.47)) ) &
                                                    ( (np.abs(data['ele1_d0'])/data['ele1_d0Sig'])<5) & 
                                                    ( (np.abs(data['ele2_d0'])/data['ele2_d0Sig'])<5) & 
                                                    ( (np.abs(data['ele1_delta_z0_sin_theta'] <0.5))) & 
                                                    ( (np.abs(data['ele2_delta_z0_sin_theta'] <0.5))) & 
                                                    # Should there be a cut in pt_varcone20? Atlas does this by removing the #% highest
                                                    ( (np.abs( data["pho_eta"]) <1.37) | ((np.abs(data['pho_eta'])>1.52) & np.abs(data['pho_eta'] <2.37)) ) &
                                                    ( data['pho_isPhotonEMTight'] == 1)
                                                    )
        lhtfpr, lhttpr, aucTight = rocVars('ATLASmask', setMask)
    elif (args.PartType == "mm"):
        data["ATLASmask"] = (   ( data[ 'muo1_charge']*data['muo2_charge']< 0) & 
                                                    (data["muo1_trigger"]==1)&
                                                    (data["muo1_LHMedium"]==1)&
                                                    (data["muo2_LHMedium"]==1)&
                                                    (data["muo1_pt"]>10000)&
                                                    (data["muo2_pt"]>10000)&
                                                    ( (np.abs( data["muo1_eta"]) <1.37) | ((np.abs(data['muo1_eta'])>1.52) & np.abs(data['muo1_eta'] <2.7)) ) &
                                                    ( (np.abs( data["muo2_eta"]) <1.37) | ((np.abs(data['muo2_eta'])>1.52) & np.abs(data['muo2_eta'] <2.7)) ) &
                                                    ( (np.abs(data['muo1_priTrack_d0'])/data['muo1_priTrack_d0Sig'])<3) & 
                                                    ( (np.abs(data['muo2_priTrack_d0'])/data['muo2_priTrack_d0Sig'])<3) & 
                                                    ( (np.abs(data['muo1_delta_z0_sin_theta'] <0.5))) & 
                                                    ( (np.abs(data['muo2_delta_z0_sin_theta'] <0.5))) 
                                                    # Should there be a cut in pt_varcone20? Atlas does this by removing the #% highest
                                                    )
        lhtfpr, lhttpr, aucTight = rocVars('ATLASmask', setMask)
    elif (args.PartType == "mmg"):
        data["ATLASmask"] = (   ( data[ 'muo1_charge']*data['muo2_charge']< 0) & 
                                                    (data["muo1_trigger"]==1)&
                                                    (data["muo1_LHMedium"]==1)&
                                                    (data["muo2_LHMedium"]==1)&
                                                    (data["muo1_pt"]>10000)&
                                                    (data["muo2_pt"]>10000)&
                                                    (data["invMll"]<82)&
                                                    ( (np.abs( data["muo1_eta"]) <1.37) | ((np.abs(data['muo1_eta'])>1.52) & np.abs(data['muo1_eta'] <2.7)) ) &
                                                    ( (np.abs( data["muo2_eta"]) <1.37) | ((np.abs(data['muo2_eta'])>1.52) & np.abs(data['muo2_eta'] <2.7)) ) &
                                                    ( (np.abs(data['muo1_priTrack_d0'])/data['muo1_priTrack_d0Sig'])<3) & 
                                                    ( (np.abs(data['muo2_priTrack_d0'])/data['muo2_priTrack_d0Sig'])<3) & 
                                                    ( (np.abs(data['muo1_delta_z0_sin_theta'] <0.5))) & 
                                                    ( (np.abs(data['muo2_delta_z0_sin_theta'] <0.5))) & 
                                                    # Should there be a cut in pt_varcone20? Atlas does this by removing the #% highest
                                                    ( (np.abs( data["pho_eta"]) <1.37) | ((np.abs(data['pho_eta'])>1.52) & np.abs(data['pho_eta'] <2.37)) ) &
                                                    ( data['pho_isPhotonEMTight'] == 1)
                                                    )
                                                    
        lhtfpr, lhttpr, aucTight = rocVars('ATLASmask', setMask)
    elif (args.PartType == "gg"):
        data["ATLASmask"] = (   ( data["pho1_isPhotonEMTight"]==1)&
                                                    ( data["pho2_isPhotonEMTight"] == 1)&
                                                    ( data["pho1_et"]>35)&
                                                    ( data["pho2_et"]>25)&
                                                    ( (np.abs( data["pho1_eta"]) <1.37) | ((np.abs(data['pho1_eta'])>1.52) & np.abs(data['pho1_eta'] <2.37)) ) &
                                                    ( (np.abs( data["pho2_eta"]) <1.37) | ((np.abs(data['pho2_eta'])>1.52) & np.abs(data['pho2_eta'] <2.37)) ) &
                                                    #( data["pho1_topoetcone20"]/data["pho1_et"]<0.065)&
                                                    #( data["pho2_topoetcone20"]/data["pho2_et"]<0.065)&
                                                    ( data["pho1_et"]/data["invM"]>0.35)&
                                                    ( data["pho2_et"]/data["invM"]>0.25)&
                                                    ( data["pho1_ptvarcone20"]/data["pho1_et"]<0.05)&
                                                    ( data["pho2_ptvarcone20"]/data["pho2_et"]<0.05)
                                                    )
        data["ATLASmaskLoose"] = (   ( data["pho1_isPhotonEMLoose"]==1)&
                                                    ( data["pho2_isPhotonEMLoose"] == 1)&
                                                    ( data["pho1_et"]>35)&
                                                    ( data["pho2_et"]>25)&
                                                    ( (np.abs( data["pho1_eta"]) <1.37) | ((np.abs(data['pho1_eta'])>1.52) & np.abs(data['pho1_eta'] <2.37)) ) &
                                                    ( (np.abs( data["pho2_eta"]) <1.37) | ((np.abs(data['pho2_eta'])>1.52) & np.abs(data['pho2_eta'] <2.37)) ) &
                                                    #( data["pho1_topoetcone20"]/data["pho1_et"]<0.065)&
                                                    #( data["pho2_topoetcone20"]/data["pho2_et"]<0.065)&
                                                    ( data["pho1_et"]/data["invM"]>0.35)&
                                                    ( data["pho2_et"]/data["invM"]>0.25)&
                                                    ( data["pho1_ptvarcone20"]/data["pho1_et"]<0.05)&
                                                    ( data["pho2_ptvarcone20"]/data["pho2_et"]<0.05)
                                                    )                          

                     
        lhtfpr, lhttpr, aucTight = rocVars('ATLASmask', setMask)
        #lhlfpr, lhltpr, aucLoose = rocVars('ATLASmaskLoose', setMask)                                                   
    if (args.PartType == "phoIso") or (args.PartType == "phoPid"):
        lhlfpr, lhltpr, lhtfpr, lhttpr = lhlfpr[1], lhltpr[1], lhtfpr[1], lhttpr[1]
    elif (args.PartType =="eleIso") or (args.PartType == "muoIso") or (args.PartType =="ee")or (args.PartType =="eeg")or (args.PartType =="mm")or (args.PartType =="mmg") or (args.PartType == "gg"):
        lhtfpr, lhttpr = lhtfpr[1], lhttpr[1]
    else:
        lhlfpr, lhltpr, lhmfpr, lhmtpr, lhtfpr, lhttpr = lhlfpr[1], lhltpr[1], lhmfpr[1], lhmtpr[1], lhtfpr[1], lhttpr[1]
    #print(f"len tprPlot: {len(tprPlot)}, len fprPlot: {len(fprPlot)}, lhltpr: {lhltpr}, lhlfpr: {lhlfpr}")#". Nearest fpr: {find_nearest(fprPlot,lhlfpr)}.")
    #print(f"{tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhlfpr))][0]},{lhlfpr}")
    
    
    if (args.PartType != "eleIso") and (args.PartType != "muoIso")and (args.PartType !="ee")and (args.PartType !="eeg")and (args.PartType !="mm")and (args.PartType !="mmg")and (args.PartType !="gg"):
        if (args.PartType != "phoIso") and (args.PartType != "phoPid") :
            ax.plot(lhmtpr,lhmfpr,'rs', label="Medium LH")
            ax.plot(tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhmfpr))][0],lhmfpr,'bs')
            print(f"ATLAS: For Medium LH, the tpr:{lhmtpr:.4f}, and the fpr:{lhmfpr:.4f}")
            print(f"LGBM: For Medium LH, the tpr:{tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhmfpr))][0]:.4f}, and the fpr:{lhmfpr:.4f}")
        print(f"ATLAS: For Loose LH, the tpr:{lhltpr:.4f}, and the fpr:{lhlfpr:.4f}")
        print(f"LGBM: For Loose LH, the tpr:{tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhlfpr))][0]:.4f}, and the fpr:{lhlfpr:.4f}")
        
        ax.plot(lhltpr,lhlfpr,'g^', label="Loose LH")
        ax.plot(tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhlfpr))][0],lhlfpr,'b^')
    print(f"ATLAS For Tight LH, the tpr:{lhttpr:.4f}, and the fpr:{lhtfpr:.4f}")
    print(f"LGBM: For Tight LH, the tpr:{tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhtfpr))][0]:.4f}, and the fpr:{lhtfpr:.4f}")
    ax.plot(lhttpr,lhtfpr,'kx', label="Tight LH")
    ax.plot(tprPlot[np.where(fprPlot==find_nearest(fprPlot,lhtfpr))][0],lhtfpr,'bx')
    

    # axins.set_xlim(0.1, 0.5)
    # axins.set_ylim(0, 0.03)

    axes.legend(loc="upper left", framealpha=1, frameon=True)
    axes.set_title(f"ROC curve - {datatypePrint[setType]} set")
    axes.set_ylabel('Background efficiency \n(False positive rate)')
    axes.set_xlabel('Signal efficiency \n(True positive rate)')

    fig.tight_layout()
    fig.savefig(output_dir + f"ROCweighted_{datatype[setType]}_v2.png")



"""
# Get weights and assign color
plotColor = ["C0","C1","C2"]

# Initiate figure
fig, ax = plt.subplots(figsize=(3,3.5))

# Cheat to create legend2
ax.plot([-2,-1], [-2,-1], label="ML eIso", color='k')
ax.plot([-2,-1], [-2,-1], label="ML eIso sel.", color='k', linestyle='None', marker='v')
ax.plot([-2,-1], [-2,-1], label="LH wp", color='k', linestyle='--', marker='*')

# Print AUC
print("Set        & Weight           & \multicolumn{2}{c}{ML eIso} & \multicolumn{2}{c}{LH Working point}  \\\\")
print("           &                  & Weighted & Unweighted       & Weighted & Unweighted                 \\\\")

# Loop over training and validation set
for setType in datasetTypes:
    setMask = data["dataset"]==setType

    # Get fpr and tpr for LGBM and wp
    fprPlot, tprPlot, aucPlot, thresholdsPlot = rocVars('predLGBM', setMask, weightName)
    fprLoose, tprLoose, aucLoose, _ = rocVars('ele_LHLoose', setMask, weightName)
    fprMedium, tprMedium, aucMedium, _ = rocVars('ele_LHMedium', setMask, weightName)
    fprTight, tprTight, aucTight, _ = rocVars('ele_LHTight', setMask, weightName)
    fprLH = [fprLoose[1],fprMedium[1],fprTight[1]]
    tprLH = [tprLoose[1],tprMedium[1],tprTight[1]]
    aucLH = auc([1,fprLoose[1],fprMedium[1],fprTight[1],0], [1,tprLoose[1],tprMedium[1],tprTight[1],0])

    # Plot LGBM, wp and LGBM cut
    ax.plot(tprPlot, fprPlot, label=datatypePrint[setType], color=plotColor[setType], alpha=0.8)
    ax.plot(tprLH, fprLH, linestyle='None', marker='*', color=plotColor[setType])
    ax.plot(tprLH, fprLH, linestyle='--', marker='None', color=plotColor[setType], alpha=0.8)

    # Get fpr and tpr for ML selection - cut at threshold
    for i in range(3):
        fprMask, idx, fprLGBM, tprLGBM, thresholdsLGBM = getSameFpr(fprPlot,tprPlot,fprLH[i],thresholdsPlot)
        ax.plot(tprLGBM, fprLGBM, marker='v', color=plotColor[setType])

    # Calculate unweighted auc for printing and print
    fprPlot_unweighted, tprPlot_unweighted, aucPlot_unweighted, thresholdsPlotUnweighted = rocVars('predLGBM', setMask)
    fprLoose_unweighted, tprLoose_unweighted, aucLoose_unweighted, _ = rocVars('ele_LHLoose', setMask)
    fprMedium_unweighted, tprMedium_unweighted, aucMedium_unweighted, _ = rocVars('ele_LHMedium', setMask)
    fprTight_unweighted, tprTight_unweighted, aucTight_unweighted, _ = rocVars('ele_LHTight', setMask)
    aucLH_unweighted = auc([1,fprLoose_unweighted[1],fprMedium_unweighted[1],fprTight_unweighted[1],0], [1,tprLoose_unweighted[1],tprMedium_unweighted[1],tprTight_unweighted[1],0])

    print(f"{datatypePrint[setType]} & {weightName} & {aucPlot:.4f} & {aucPlot_unweighted:.4f} & {aucLH:.4f} & {aucLH_unweighted:.4f} \\\\")


# Set labels
ax.set_ylabel('Background efficiency \n(False positive rate)')
ax.set_xlabel('Signal efficiency \n(True positive rate)')

# Set x and y scales
ax.set_xlim(0.7, 1)
ax.set_yscale('log')
ax.set_ylim(10**(-3), 10**(-1))
# ax.legend(loc="upper left")
ax.text(0.01, 1.01, f"eIso", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

# Create legend
handles, labels = ax.get_legend_handles_labels()
axbox = ax.get_position()
fig.legend(handles, labels, loc="lower center",ncol=2)

# Save
fig.tight_layout(rect=[0,0.2,1,1], h_pad=0.3, w_pad=0.3)
fig.savefig(output_dir + f"ROCcurve.png")

#%%############################################################################
#   Print performance
###############################################################################
print()
print("Printing false and true positive rate table for latex:")
print(f"Fpr and tpr:")
print("Set        & Weight           &                   & \multicolumn{3}{c}{fpr [10^(-2)]} & \multicolumn{4}{c}{tpr}  \\\\")
print("           &                  & LH working points & ML Zee & Difference               & LH working points & ML Zee & Difference & Change     \\\\")
for setType in datasetTypes:
    setMask = data["dataset"]==setType

    # Get fpr and tpr for LGBM and wp
    fprPlot, tprPlot, aucPlot, thresholdsPlot = rocVars('predLGBM', setMask, weightName)
    fprLoose, tprLoose, aucLoose, _ = rocVars('ele_LHLoose', setMask, weightName)
    fprMedium, tprMedium, aucMedium, _ = rocVars('ele_LHMedium', setMask, weightName)
    fprTight, tprTight, aucTight, _ = rocVars('ele_LHTight', setMask, weightName)
    fprLH = [fprLoose[1],fprMedium[1],fprTight[1]]
    tprLH = [tprLoose[1],tprMedium[1],tprTight[1]]
    aucLH = auc([1,fprLoose[1],fprMedium[1],fprTight[1],0], [1,tprLoose[1],tprMedium[1],tprTight[1],0])

    # Get fpr and tpr for ML selection - cut at threshold
    for i, wp in enumerate(["Loose", "Medium", "Tight"]):
        fprMask, idx, fprLGBM, tprLGBM, thresholdsLGBM = getSameFpr(fprPlot,tprPlot,fprLH[i],thresholdsPlot)
        print(f"{datatypePrint[setType]} & {weightName} & {wp} & {fprLH[i]*10**(2):.4f} & {fprLGBM*10**(2):.4f} & {(fprLGBM-fprLH[i])*10**(2):.4f} & {tprLH[i]:.4f} & {tprLGBM:.4f} & {tprLGBM-tprLH[i]:.4f} & {(tprLGBM-tprLH[i])/tprLH[i]*100:.2f}\%\\\\")
"""
print()



print(f'')
print(f'')
print(f"END OF PROGRAM - Total time spent: {timediftostr(t_total_start)}")

