#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used to plot shap values for a model (so it wont be required to retrain the model)
May need updating of the variables, if these are changed in other models.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from scipy.special import logit
from sklearn.metrics import auc, roc_curve, roc_auc_score, mean_absolute_error

from utils import auc_eval, HeatMap_rand, HyperOpt_RandSearch, accuracy, h5ToDf, timediftostr, Build_Folder_Structure, header, footer, print_col_names


print("Packages and functions loaded")
t_total_start = time()

weightNames = [ "regWeight_nEst10", "regWeight_nEst40", "regWeight_nEst100", "regWeight_nEst200",
                "revWeight_nEst10", "revWeight_nEst40", "revWeight_nEst100", "revWeight_nEst200" ]

weightsString = "Choose weights:"
for iWeight, weight in enumerate(weightNames):
    weightsString = weightsString + f" {iWeight} = {weight},"


parser = argparse.ArgumentParser(description='Electron isolation with the LightGBM framework.')
parser.add_argument('paths', type=str, #nargs='+',
                    help='Data file(s) to predict from.')
parser.add_argument('--outdir', action='store', type=str, default="output/TrainingPlots/",
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--model', action='store', type=str, required=True,
                    help='Input model: What directory to load data from.')
parser.add_argument('--weights', action='store', type=int, required=True,
                    help=f'Choose weights: 0 = {weightNames[0]}, 1 = {weightNames[1]}, 2 = {weightNames[2]}')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["eeg", "mmg", "muoIso", "muoPid", "eleIso", "eleIsosimple", "elePid", "phoIso", "phoPid", 'ee', 'mm','phoALL','gg'],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')
args = parser.parse_args()
log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "eleIso") +(args.PartType != "elePid") + (args.PartType != "muoIso") +(args.PartType != "muoPid")+ (args.PartType != "eeg") + (args.PartType != "mmg") + (args.PartType != "phoIso")+(args.PartType != "phoPid")+(args.PartType != "ee"))+(args.PartType != "mm")+(args.PartType != "phoALL")+(args.PartType != "eleIsosimple")!=11:
    log.error("Unknown lepton, use either eleIso, eleIsosimple, elePid, muoIso, muoPid, eeg, mmg, ee, mm, phoPid or phoIso")
    quit()
#%%############################################################################
#   Filenames and directories
###############################################################################
# Name of data file
filename_base = os.path.basename(args.paths)
filename_split = os.path.splitext(filename_base)[0]
filename = filename_split.split("_")[1]

# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"

# Get weight name and type
weightName = weightNames[args.weights]

# Initially I will use both moodels, Helle does however write, that PID give the better result

if args.tag:
    output_dir = args.outdir + args.PartType+"model_LGBM_" + filename + "_" + args.tag + "_" + weightName + "_" + timelabel + "/"
    modelname = args.PartType+"model_LGBM_" + filename + "_" + args.tag + "_" + weightName + "_" + timelabel
    figname = args.PartType+"_LGBM_" + filename + "_" + args.tag + "_" + weightName + "_" + timelabel + "_"
    hypparamname = args.PartType+"hyperParam_LGBM_" + filename + "_" + args.tag + "_" + weightName + "_" + timelabel
elif not args.tag:
    output_dir = args.outdir + args.PartType+"model_LGBM_" + filename + "_" + weightName + "_" + timelabel + "/"
    modelname = args.PartType+"model_LGBM_" + filename + "_" + weightName + "_" + timelabel
    figname = args.PartType+"_LGBM_" + filename + "_" + weightName + "_" + timelabel + "_"
    hypparamname = args.PartType+"hyperParam_LGBM_" + filename + "_" + weightName + "_" + timelabel

# Check if output folder already  exists
if os.path.exists(output_dir):
    log.error(f"Output already exists - please remove yourself. Output dir: {output_dir}")
    quit()

# Figure subdirectory
figure_dir = output_dir + "Images/"
 
# Build the folder structure
Build_Folder_Structure([figure_dir])
#%%############################################################################
#   Set parameters
###############################################################################
# Set random seed
np.random.seed(seed= 42)
# Define variables needed
truth_var = "type"

# Should look up what to do with eeg and mmg.
if args.PartType == "eeg":
    training_var = ['NvtxReco',
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
                'pho_pPid_score']
elif args.PartType == "ee":
    training_var = ['NvtxReco',
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
                'ele2_eIso_score']
elif args.PartType == "mmg":
    training_var = ['NvtxReco',
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
                    'pho_pPid_score']
elif args.PartType == "mm":
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'muo1_delta_z0',
                    'muo2_delta_z0',
                    'muo1_delta_z0_sin_theta',
                    'muo2_delta_z0_sin_theta',
                    'muo1_vertex_z',
                    'muo2_vertex_z',
                    'muo1_charge',
                    'muo2_charge',
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
                    #'ele_expectInnermostPixelLayerHit',
                    #'ele_expectNextToInnermostPixelLayerHit',
                    'ele_core57cellsEnergyCorrection',
                    'ele_nTracks',
                    'ele_numberOfInnermostPixelHits',
                    'ele_numberOfPixelHits',
                    #'ele_numberOfSCTHits',
                    'ele_numberOfTRTHits',
                    'ele_topoetcone20ptCorrection',
                    ]
elif args.PartType == "eleIsosimple": 
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'ele_et',
                    'ele_eta',
                    'ele_ptvarcone20',
                    'ele_topoetcone20',
                    'ele_topoetcone40',
                    'ele_ptvarcone20_rel',
                    #'ele_ptvarcone40_rel',
                    'ele_topoetcone20_rel'
                    #'ele_topoetcone40_rel',
                    #'ele_expectInnermostPixelLayerHit',
                    #'ele_expectNextToInnermostPixelLayerHit',
                    #'ele_core57cellsEnergyCorrection',
                    #'ele_nTracks',
                    #'ele_numberOfInnermostPixelHits',
                    #'ele_numberOfPixelHits',
                    #'ele_numberOfSCTHits',
                    #'ele_numberOfTRTHits',
                    #'ele_topoetcone20ptCorrection',
                    ]
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
                    'ele_TRTPID'
                    ]
elif args.PartType == "muoIso":
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    #'muo_etcon20',
                    'muo_ptvarcone20_rel',
                    'muo_ptvarcone40_rel',
                    'muo_etconecoreConeEnergyCorrection',
                    #'muo_topoetconecoreConeEnergyCorrection',
                    'muo_priTrack_numberOfPixelHits',
                    'muo_priTrack_numberOfSCTHits',
                    'muo_priTrack_numberOfTRTHits'
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
                    'muo_energyLossType'
                    ]             
elif args.PartType == "phoIso":
    training_var = ['correctedScaledAverageMu',
                    'NvtxReco',
                    'pho_et',
                    'pho_ptvarcone20',
                    'pho_topoetcone20',
                    'pho_topoetcone40']
elif args.PartType == "phoPid":
    training_var = ['pho_Rhad',
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
elif args.PartType =="phoALL":
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
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    'pho_z0',
                    'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']

#%%############################################################################
#   Importing data.
###############################################################################
header("Importing and separating data")
t = time()

# Get hdf5 datafile as dataframe
data = h5ToDf(args.paths)

# Print column names to log
print_col_names(data.columns, training_var, truth_var)

# Separate label and training data
X_train = data[training_var][data["dataset"] == 0]
y_train = data[truth_var][data["dataset"] == 0]
X_valid = data[training_var][data["dataset"] == 1]
y_valid = data[truth_var][data["dataset"] == 1]

#Import model
gbm = lgb.Booster(model_file = args.model)

print('Calculating SHAP values...')
shap_values = shap.TreeExplainer(gbm).shap_values(X_train)
    

fig_shap=plt.figure(figsize=(16,12))
shap.summary_plot(shap_values, X_train, plot_type="bar",plot_size=(16,12))
fig_shap.savefig(figure_dir+figname+'featureImportance_SHAP.png')
plt.close(fig_shap)