#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using LightGBM to train boosted decision trees to identify photons
based on variables, doing it for n models - and their data.
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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from scipy.special import logit
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error

from utils import timediftostr, Build_Folder_Structure, header, footer, print_col_names,h5ToDf

print("Packages and functions loaded")
t_total_start = time()

weightNames = [ "regWeight_nEst10", "regWeight_nEst40", "regWeight_nEst100", "revWeight_nEst10", "revWeight_nEst40", "revWeight_nEst100" ]

weightsString = "Choose weights:"
for iWeight, weight in enumerate(weightNames):
    weightsString = weightsString + f" {iWeight} = {weight},"



#%%############################################################################
#   Set parameters for the two models
###############################################################################
# Define parameters for the different models
"""
In this case will number 1 be a Pid, and number 2 will be Iso.
"""
# Set random seed
np.random.seed(seed=42)

# Define variables needed
truth_var = "type"
""""
training_varIso = ['NvtxReco',
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
                ]
training_varIso2 = ['NvtxReco',
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
training_varIsosimple = ['NvtxReco',
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
"""
# The following lists are different configurations of variables
# They can be used to compare several different models
training_varPid = ['pho_Rhad',
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
training_varPid2 = ['pho_isPhotonEMLoose',
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
"""
training_varPAll = ['pho_isPhotonEMLoose',
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
                    'pho_ConversionType',
                    'pho_ConversionRadius',
                    'pho_VertexConvEtOverPt',
                    'pho_VertexConvPtRatio',
                    'pho_z0',
                    'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']
                    

training_varPAll2 = ['pho_isPhotonEMLoose',
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
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']

training_varPAll3 = ['pho_isPhotonEMLoose',
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
                    'pho_VertexConvPtRatio',
                    'pho_z0',
                    'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']
training_varPAll4 = ['pho_isPhotonEMLoose',
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
training_varPAll5 = ['pho_isPhotonEMLoose',
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
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']                                         
training_varPAll6 = ['pho_isPhotonEMLoose',
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
                    #'pho_ConversionRadius',
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    'pho_z0',
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']  
training_varPAll7 = [#'pho_isPhotonEMLoose',
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
                    #'pho_ConversionRadius',
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    'pho_z0',
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']  
training_varPAll8 = [#'pho_isPhotonEMLoose',
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
                    #'pho_ConversionRadius',
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    #'pho_z0',
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']  
training_varPAll9 = [#'pho_isPhotonEMLoose',
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
                    #'pho_fracs1',
                    #'pho_ConversionType',
                    #'pho_ConversionRadius',
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    #'pho_z0',
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo'] 
training_varPAll10 = [#'pho_isPhotonEMLoose',
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
                    #'pho_weta1',
                    #'pho_fracs1',
                    #'pho_ConversionType',
                    #'pho_ConversionRadius',
                    #'pho_VertexConvEtOverPt',
                    #'pho_VertexConvPtRatio',
                    #'pho_z0',
                    #'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo']                      
"""
def rocVars(true, predict, weight):
    fpr, tpr, thresholds = roc_curve(true, predict, sample_weight = weight)
    aucCalc = auc(fpr,tpr)
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

# Loading different isolation models
iso = "output/Training/models/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso180921_revWeight_nEst40_2021-09-18/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso180921_revWeight_nEst40_2021-09-18.txt"
iso2 = "output/Training/models/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso200921_revWeight_nEst40_2021-09-20/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso200921_revWeight_nEst40_2021-09-20.txt"
isosimple = "output/Training/models/eleIsosimplemodel_LGBM_LeptonLepton_eleRev40Isosimple200921_revWeight_nEst40_2021-09-20/eleIsosimplemodel_LGBM_LeptonLepton_eleRev40Isosimple200921_revWeight_nEst40_2021-09-20.txt"
shapdata = "output/ReweightFiles/combined_LeptonLepton_EleIso170921_2021-09-17_binWidth5_testSize20_validSize20/combined_LeptonLepton_EleIso170921_2021-09-17_test.h5"

# Loading different Identification models
pPidreg40 ="output/Training/models/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid180921_regWeight_nEst40_2021-09-18/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid180921_regWeight_nEst40_2021-09-18.txt"
pPidreg402 = "output/Training/models/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29.txt"
"""
shaptest = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v1_regWeight_nEst40_2021-09-20/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v1_regWeight_nEst40_2021-09-20.txt"
shaptest2 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v2_regWeight_nEst40_2021-09-20/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v2_regWeight_nEst40_2021-09-20.txt"
shaptest3 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v3_regWeight_nEst40_2021-09-20/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest200921v3_regWeight_nEst40_2021-09-20.txt"
shaptest4 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v4_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v4_regWeight_nEst40_2021-09-21.txt"
shaptest5 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v5_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v5_regWeight_nEst40_2021-09-21.txt"
shaptest6 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v6_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v6_regWeight_nEst40_2021-09-21.txt"
shaptest7 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v7_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v7_regWeight_nEst40_2021-09-21.txt"
shaptest8 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v8_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v8_regWeight_nEst40_2021-09-21.txt"
shaptest9 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v9_regWeight_nEst40_2021-09-21/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest210921v9_regWeight_nEst40_2021-09-21.txt"
shaptest10 = "output/Training/models/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest260921v10_regWeight_nEst40_2021-09-26/phoALLmodel_LGBM_phoPidphoPid_phoReg40vartest260921v10_regWeight_nEst40_2021-09-26.txt"
"""
#pPiddata ="output/ReweightFiles/combined_phoPid_phoPidFull_2021-07-20_binWidth5_testSize20_validSize20/combined_phoPid_phoPidFull_2021-07-20_test.h5"
shapdata = "output/ReweightFiles/combined_phoPidphoPid_PhoPid170921v2_2021-09-17_binWidth5_testSize20_validSize20/combined_phoPidphoPid_PhoPid170921v2_2021-09-17_test.h5"

#%%############################################################################
#   Filenames and directories
###############################################################################

# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"
# timelabel = f"{datetime.date(now)}_{int(timestamp)}"

# Create names for output
output_dir = "output/Compare2ROC/LGBM_compareROC_phoPid_" + timelabel + "/"
figname = "LGBM_compareROC_phoPid_" + timelabel + "_"

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
print(f'PID model:  ')
#print(f'         'f'PIDrev10:{pPidreg40}')
#print(f'         'f'PIDrev40:{shaptest}')
print(f'All Isolation models have the same data, and the same for Pid')
print(f'N jobs:            {nJobs}')
print(f'------------------------------------------------------------------')



#%%############################################################################
#   Get data
###############################################################################
print(f"Importing data...")
data1 = h5ToDf(shapdata)
#data2 = h5ToDf(pPiddata)
"""
data2['ele_ptvarcone20_rel'] = data2['ele_ptvarcone20'] / data2['ele_et']
data2['ele_ptvarcone40_rel'] = data2['ele_ptvarcone40_TightTTVALooseCone_pt1000'] / data2['ele_et']
data2['ele_topoetcone20_rel'] = data2['ele_topoetcone20'] / data2['ele_et']
data2['ele_topoetcone40_rel'] = data2['ele_topoetcone40'] / data2['ele_et']

X1 = data1[training_varIso][:]
X2 = data1[training_varIso2][:]
X3 = data1[training_varIsosimple][:]
"""
X = data1[training_varPid][:]
x = data1[training_varPid2][:]
"""
X1 = data1[training_varPAll][:]
X2 = data1[training_varPAll2][:]
X3 = data1[training_varPAll3][:]
X4 = data1[training_varPAll4][:]
X5 = data1[training_varPAll5][:]
X6 = data1[training_varPAll6][:]
X7 = data1[training_varPAll7][:]
X8 = data1[training_varPAll8][:]
X9 = data1[training_varPAll9][:]
X10 = data1[training_varPAll10][:]
""" # Choosing the weight from Reg nest10

#weightpPidreg40 = data1[weightNames[3]]
weihtpshap = data1[weightNames[4]]

# Import model
print(f"Import model: {weightNames[0]}")
"""
gbmIso = lgb.Booster(model_file=iso)
gbmIso2 = lgb.Booster(model_file=iso2)
gbmIsosimple = lgb.Booster(model_file=isosimple)
"""
gbmpPidreg40 = lgb.Booster(model_file = pPidreg40)
gbmpPidreg402 = lgb.Booster(model_file = pPidreg402)
"""
gbmShap = lgb.Booster(model_file=shaptest)
gbmShap2 = lgb.Booster(model_file=shaptest2)
gbmShap3 = lgb.Booster(model_file=shaptest3)
gbmShap4 = lgb.Booster(model_file=shaptest4)
gbmShap5 = lgb.Booster(model_file=shaptest5)
gbmShap6 = lgb.Booster(model_file=shaptest6)
gbmShap7 = lgb.Booster(model_file=shaptest7)
gbmShap8 = lgb.Booster(model_file=shaptest8)
gbmShap9 = lgb.Booster(model_file=shaptest9)
gbmShap10 = lgb.Booster(model_file=shaptest10)
"""

#%%############################################################################
#   Prediction
###############################################################################
header("Running Predictions")
t = time()

#Predict on all data and save variables in dataframe
print("Predicting on data")
"""
y_pred_iso = gbmIso.predict(X1, num_iteration = gbmIso.best_iteration, n_jobs=nJobs)
y_pred_iso2 = gbmIso2.predict(X2, num_iteration = gbmIso2.best_iteration, n_jobs=nJobs)
y_pred_isosimple = gbmIsosimple.predict(X3, num_iteration = gbmIsosimple.best_iteration, n_jobs=nJobs)
data1["predLGBMIso"] = y_pred_iso
data1["predLGBMIso2"] = y_pred_iso2
data1["predLGBMIsosimple"] = y_pred_isosimple
"""
y_pred_pPidreg40 = gbmpPidreg40.predict(X, num_iteration = gbmpPidreg40.best_iteration, n_jobs = nJobs)
y_pred_pPidreg402 = gbmpPidreg402.predict(x, num_iteration = gbmpPidreg402.best_iteration, n_jobs = nJobs)
#If further models should be included
"""
y_pred_pShap = gbmShap.predict(X1, num_iteration = gbmShap.best_iteration, n_jobs=nJobs)
y_pred_pShap2 = gbmShap2.predict(X2, num_iteration = gbmShap2.best_iteration, n_jobs=nJobs)
y_pred_pShap3 = gbmShap3.predict(X3, num_iteration = gbmShap3.best_iteration, n_jobs=nJobs)
y_pred_pShap4 = gbmShap4.predict(X4, num_iteration = gbmShap4.best_iteration, n_jobs=nJobs)
y_pred_pShap5 = gbmShap5.predict(X5, num_iteration = gbmShap5.best_iteration, n_jobs=nJobs)
y_pred_pShap6 = gbmShap6.predict(X6, num_iteration = gbmShap6.best_iteration, n_jobs=nJobs)
y_pred_pShap7 = gbmShap7.predict(X7, num_iteration = gbmShap7.best_iteration, n_jobs=nJobs)
y_pred_pShap8 = gbmShap8.predict(X8, num_iteration = gbmShap8.best_iteration, n_jobs=nJobs)
y_pred_pShap9 = gbmShap9.predict(X9, num_iteration = gbmShap9.best_iteration, n_jobs=nJobs)
y_pred_pShap10 = gbmShap10.predict(X10, num_iteration = gbmShap10.best_iteration, n_jobs=nJobs)
"""
data1["predLGBMPid"] = y_pred_pPidreg40
data1["predLGBMPid2"] = y_pred_pPidreg402
"""
data1["predLGBMshap"] = y_pred_pShap
data1["predLGBMshap2"] = y_pred_pShap2
data1["predLGBMshap3"] = y_pred_pShap3
data1["predLGBMshap4"] = y_pred_pShap4
data1["predLGBMshap5"] = y_pred_pShap5
data1["predLGBMshap6"] = y_pred_pShap6
data1["predLGBMshap7"] = y_pred_pShap7
data1["predLGBMshap8"] = y_pred_pShap8
data1["predLGBMshap9"] = y_pred_pShap9
data1["predLGBMshap10"] = y_pred_pShap10
"""
# Signal selection on non logit transformed
sigSel = 0.5

# Select signal
#print(f"Make LGBM selection with non logit transformed cut: {sigSel}\n")
#data1["selLGBM"] = 0
#data2["selLGBM"] = 0
#data1.loc[(y_pred_data1>sigSel),["selLGBM"]] = 1
#data2.loc[(y_pred_data2>sigSel),["selLGBM"]] = 1


#%%############################################################################
#   ROC curve
###############################################################################
header('Plotting results')
# Based on: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
print("Plotting ROC curve")

def rocVars(DATA, predColumn, setMask, weightnumb):
    true = DATA.loc[setMask][truth_var]
    predict = DATA.loc[setMask][predColumn]
    fpr, tpr, thresholds = roc_curve(true, predict, sample_weight=DATA.loc[setMask][weightNames[weightnumb]])
    aucCalc = auc(fpr, tpr)
    return fpr, tpr, aucCalc

#plotColumns = ["predLGBMreg10","predLGBMreg40","predLGBMreg100","predLGBMrev10","predLGBMrev40","predLGBMrev100"]
plotNamesSHAP = ["ML test 1","ML test 2","ML test 3", "ML test 4","ML test 5", "ML test 6","ML test 7","ML test 8","ML test 9", "ML test 10"]
plotNamesPID = "ML phoPidreg40"
#plotnames = ["Iso", "iso2", "isosimple"]
datasetTypes = 2
setMask1 = data1["dataset"] == datasetTypes
#setMask2 = data2["dataset"] == datasetTypes

fig, ax = plt.subplots(figsize=(8,8))
# Calculating the fpr, tpr and auc.
"""
fprPlot1, tprPlot1, aucPlot1 = rocVars(data1, "predLGBMIso", setMask1, 4)
fprPlot2, tprPlot2, aucPlot2 = rocVars(data1, "predLGBMIso2", setMask1, 4)
fprPlot3, tprPlot3, aucPlot3 = rocVars(data1, "predLGBMIsosimple", setMask1, 4)
ax.plot(tprPlot1,fprPlot1,label=plotnames[0]+f" AUC= {aucPlot1:.3f}")
ax.plot(tprPlot2,fprPlot2,label=plotnames[1]+f" AUC= {aucPlot2:.3f}")
ax.plot(tprPlot3,fprPlot3,label=plotnames[2]+f" AUC= {aucPlot3:.3f}")
"""

"""fprPlot1, tprPlot1, aucPlot1 = rocVars(data1, "predLGBMshap", setMask1, 1)
fprPlot2, tprPlot2, aucPlot2 = rocVars(data1, "predLGBMshap2", setMask1, 1)
fprPlot3, tprPlot3, aucPlot3 = rocVars(data1, "predLGBMshap3", setMask1, 1)
fprPlot4, tprPlot4, aucPlot4 = rocVars(data1, "predLGBMshap4", setMask1, 1)
fprPlot5, tprPlot5, aucPlot5 = rocVars(data1, "predLGBMshap5", setMask1, 1)
fprPlot6, tprPlot6, aucPlot6 = rocVars(data1, "predLGBMshap6", setMask1, 1)
fprPlot7, tprPlot7, aucPlot7 = rocVars(data1, "predLGBMshap7", setMask1, 1)
fprPlot8, tprPlot8, aucPlot8 = rocVars(data1, "predLGBMshap8", setMask1, 1)
fprPlot9, tprPlot9, aucPlot9 = rocVars(data1, "predLGBMshap9", setMask1, 1)
fprPlot10, tprPlot10, aucPlot10 = rocVars(data1, "predLGBMshap10", setMask1, 1)
"""
fprPlot, tprPlot, aucPlot = rocVars(data1, "predLGBMPid", setMask1, 1)
FprPlot, TprPlot, AucPlot = rocVars(data1, "predLGBMPid2", setMask1, 1)
#Plotting the fpr and tpr, writing the auc.
ax.plot(tprPlot,fprPlot,label=plotNamesPID+f" AUC= {aucPlot:.3f}")
ax.plot(TprPlot,FprPlot,label=f"ML phoPidreg40 v2 AUC={AucPlot:.3f}")
"""
ax.plot(tprPlot1,fprPlot1,label=plotNamesSHAP[0]+f" AUC= {aucPlot1:.3f}")
ax.plot(tprPlot2,fprPlot2,label=plotNamesSHAP[1]+f" AUC= {aucPlot2:.3f}")
ax.plot(tprPlot3,fprPlot3,label=plotNamesSHAP[2]+f" AUC= {aucPlot3:.3f}")
ax.plot(tprPlot4,fprPlot4,label=plotNamesSHAP[3]+f" AUC= {aucPlot4:.3f}")
ax.plot(tprPlot5,fprPlot5,label=plotNamesSHAP[4]+f" AUC= {aucPlot5:.3f}")
ax.plot(tprPlot6,fprPlot6,label=plotNamesSHAP[5]+f" AUC= {aucPlot6:.3f}")
ax.plot(tprPlot7,fprPlot7,label=plotNamesSHAP[6]+f" AUC= {aucPlot7:.3f}")
ax.plot(tprPlot8,fprPlot8,label=plotNamesSHAP[7]+f" AUC= {aucPlot8:.3f}")
ax.plot(tprPlot9,fprPlot9,label=plotNamesSHAP[8]+f" AUC= {aucPlot9:.3f}")
ax.plot(tprPlot10,fprPlot10,label=plotNamesSHAP[9]+f" AUC= {aucPlot10:.3f}")
"""
ax.set_ylabel('Background efficiency \n(False positive rate)')
ax.set_xlabel('Signal efficiency \n(True positive rate)')

ax.set_xlim(0.5,1)
ax.set_ylim(0,0.5)

handlesIso, labelsIso = ax.get_legend_handles_labels()
fig.legend(handlesIso, labelsIso, loc = "lower left")
#fig.suptitle(f"ROC curve - for eleIso")
fig.suptitle(f"ROC curve - for phoPid and pho Shap set")
fig.tight_layout(rect=[0,0.05, 1, 0.95], h_pad=0.3, w_pad=0.3)
#fig.savefig(output_dir+f"ROCweighted_eleIso.png")
fig.savefig(output_dir+f"ROCweighted_phoPid_and_phoShap_v1.png")


print(f'')
print(f'')
print(f"END OF PROGRAM - Total time spent: {timediftostr(t_total_start)}")
