#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use LightGBM to train boosted decision trees to identify photons
based on variables, doing it for two models - and their data.

This program makes predictions based on trained models and compare those
"""
print("Program running...")

# Import Packages and functionsimport sys
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
import seaborn as sns

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
#   Set parameters for the two models
###############################################################################
# Set random seed
np.random.seed(seed = 42)

# Define variables needed
truth_var = "type"
training_Isovar = ['correctedScaledActualMu',
                    'NvtxReco',
                    'pho_et',
                    'pho_topoetcone20',
                    'pho_topoetcone40',
                    'pho_ptvarcone20'
                    ]
"""# For phoPidmodel_LGBM_phoPid_reg10PPid200521_regWeight_nEst10_2021-05-20
training_Pidvar = ['correctedScaledAverageMu',
                    'pho_DeltaE',
                    'pho_Eratio',
                    'pho_et',
                    'pho_eta',
                    'pho_f1',
                    'pho_fracs1',
                    'pho_ConversionRadius',
                    'pho_ConversionType',
                    'pho_VertexConvEtOverPt',
                    'pho_VertexConvPtRatio',
                    'pho_Reta',
                    'pho_Rhad',
                    'pho_Rhad1',
                    'pho_Rphi',
                    'pho_weta1',
                    'pho_weta2',
                    'pho_wtots1']

"""
#For phoPidmodel_LGBM_phoPid_reg10PPid180521_regWeight_nEst10_2021-05-18
training_Pidvar =  ['pho_eta',
                    'pho_et',
                    'pho_Rhad1',
                    'pho_Rhad',
                    'pho_weta2',
                    'pho_Rphi',
                    'pho_Reta',
                    'pho_Eratio',
                    'pho_wtots1',
                    'pho_DeltaE',
                    'pho_weta1',
                    'pho_fracs1',
                    'pho_f1',
                    'pho_ConversionType',
                    'pho_ConversionRadius',
                    'pho_VertexConvEtOverPt',
                    'pho_VertexConvPtRatio',
                    'pho_r33over37allcalo'
                    ]

def rocVars(true, predict, weight):
    fpr, tpr, thresholds = roc_curve(true, predict, sample_weight = weight)
    aucCalc = auc(fpr, tpr)
    return fpr, tpr, aucCalc

#%%############################################################################
#   Parser
###############################################################################

nJobs = 5

weightNames = [ "regWeight_nEst10", "regWeight_nEst40", "regWeight_nEst100", "revWeight_nEst10", "revWeight_nEst40", "revWeight_nEst100" ]
weightTypes = [ "regular nEst10", "regular nEst40", "regular nEst100", "reverse nEst10", "reverse nEst40", "reverse nEst100" ]

weightsString = "Choose weights:"
for iWeight, weight in enumerate(weightNames):
    weightsString = weightsString + f" {iWeight} = {weight},"

dataPath = "output/ReweightFiles/..."

IsoModelPath = "output/Training/models/phoIsomodel_LGBM_phoIso_reg10PIso180521_regWeight_nEst10_2021-05-18/phoIsomodel_LGBM_phoIso_reg10PIso180521_regWeight_nEst10_2021-05-18.txt"
PidModelPath = "output/Training/models/phoPidmodel_LGBM_phoPid_reg10PPid180521_regWeight_nEst10_2021-05-18/phoPidmodel_LGBM_phoPid_reg10PPid180521_regWeight_nEst10_2021-05-18.txt"
#PidModelPath = "output/Training/models/phoPidmodel_LGBM_phoPid_reg10PPid200521_regWeight_nEst10_2021-05-20"

#%%############################################################################
#   Filenames and directories
###############################################################################
# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"

# Create names
output_dir = "output//CompareROC/pho_LGBM_compareROC_Final_" + timelabel + "/"
figname = "pho_LGBM_compareROC_Final_" + timelabel + "_"

# Check if output folder already exists
if os.path.exists(output_dir):
    log.error(f"Output already exists - please remove yourself. Output dir: {output_dir}")
    quit()

# Build the folder structure
Build_Folder_Structure([output_dir])

#%%############################################################################
#   Print initial log info
###############################################################################
# Log INFO for run
print(f'')
print(f'---------------------------- INFO --------------------------------')
print(f'Output directory:  {output_dir}')
print(f'Datafile:          {dataPath}')
print(f'Models:            {IsoModelPath}')
print(f'                   {PidModelPath}')
print(f'N jobs:            {nJobs}')
print(f'------------------------------------------------------------------')

#%%############################################################################
#   Get data
###############################################################################
print(f"Importing data...")
data = h5ToDf(dataPath)

XIso = data[training_Isovar][:]
XPid = data[training_Pidvar][:]

# Choosing the weight from Reg nest10
weight1 = data[weightNames[0]]


# Import model
print(f"Import model: {weightNames[0]}")
gbmIso = lgb.Booster(model_file = IsoModelPath)
gbmPid = lgb.Booster(model_file = PidModelPath)


#%%############################################################################
#   Prediction
###############################################################################
header("Running Predictions")
t = time()

#Predict on all data and save variables in dataframe
print("Predicting on data")

y_pred_dataIso = gbmIso.predict(XIso, num_iteration = gbmIso.best_iteration, n_jobs = nJobs)
y_pred_dataPid = gbmPid.predict(XPid, num_iteration = gbmPid.best_iteration, n_jobs = nJobs)
data["predLGBMIso"] = y_pred_dataIso
data["predLGBMPid"] = y_pred_dataPid'
# Signal selection on non logit transformed
sigSel = 0.5

# Select signal
print(f"Make LGBM selection with non logit transformed cut: {sigSel}\n")
data["selLGBMIso"] = 0
data["selLGBMPid"] = 0
data.loc[(y_pred_dataIso>sigSel),["selLGBMIso"]] = 1
data.loc[(y_pred_dataPid>sigSel),["selLGBMPid"]] = 1

