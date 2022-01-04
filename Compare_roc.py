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


PartType = "ll"
#%%############################################################################
#   Set parameters for the two models
###############################################################################
"""
In this case will number 1 be a Pid, and number 2 will be Iso.
"""
# Set random seed
np.random.seed(seed=42)

# Define variables needed
truth_var = "type"
if PartType=="p":
    training_varpIso = ['correctedScaledAverageMu',
                    'NvtxReco',
                    'pho_et',
                    'pho_ptvarcone20',
                    'pho_topoetcone20',
                    'pho_topoetcone40']
    training_varpPid = ['pho_Rhad',
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
elif PartType=="e":
    training_varpIso = ['NvtxReco',
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
    training_varpPid = ['ele_d0',
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
elif PartType=="m":  
    training_varpIso = ['NvtxReco',
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
    training_varpPid = ['muo_priTrack_d0',
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
elif PartType == "ll":
    training_varpIso = ['NvtxReco',
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
    training_varpPid = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'muo1_delta_z0',
                    'muo2_delta_z0',
                    'muo1_vertex_z',
                    'muo1_charge',
                    'muo2_charge',
                    'muo1_mPid_score',
                    'muo2_mPid_score',
                    'muo1_mIso_score',
                    'muo2_mIso_score']                            
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
if PartType == "p":
    pIsoreg10 ="output/Training/models/phoIsomodel_LGBM_phoIso_reg10phoIsoFull2_regWeight_nEst10_2021-07-23/phoIsomodel_LGBM_phoIso_reg10phoIsoFull2_regWeight_nEst10_2021-07-23.txt"
    pIsoreg40 ="output/Training/models/phoIsomodel_LGBM_phoIso_reg40phoIsoFull2_regWeight_nEst40_2021-08-13/phoIsomodel_LGBM_phoIso_reg40phoIsoFull2_regWeight_nEst40_2021-08-13.txt" 
    pIsoreg100 ="output/Training/models/phoIsomodel_LGBM_phoIso_reg100phoIsoFull2_regWeight_nEst100_2021-08-13/phoIsomodel_LGBM_phoIso_reg100phoIsoFull2_regWeight_nEst100_2021-08-13.txt"
    pIsorev10 ="output/Training/models/phoIsomodel_LGBM_phoIso_rev10phoIsoFull2_revWeight_nEst10_2021-08-13/phoIsomodel_LGBM_phoIso_rev10phoIsoFull2_revWeight_nEst10_2021-08-13.txt"
    pIsorev40 ="output/Training/models/phoIsomodel_LGBM_phoIso_rev40phoIsoFull2_revWeight_nEst40_2021-08-13/phoIsomodel_LGBM_phoIso_rev40phoIsoFull2_revWeight_nEst40_2021-08-13.txt"
    pIsorev100 ="output/Training/models/phoIsomodel_LGBM_phoIso_rev100phoIsoFull2_revWeight_nEst100_2021-08-13/phoIsomodel_LGBM_phoIso_rev100phoIsoFull2_revWeight_nEst100_2021-08-13.txt"

    pPidreg10 ="output/Training/models/phoPidmodel_LGBM_phoPid_reg10phoPidFull_regWeight_nEst10_2021-08-13/phoPidmodel_LGBM_phoPid_reg10phoPidFull_regWeight_nEst10_2021-08-13.txt"
    pPidreg40 ="output/Training/models/phoPidmodel_LGBM_phoPid_reg40phoPidFull_regWeight_nEst40_2021-08-13/phoPidmodel_LGBM_phoPid_reg40phoPidFull_regWeight_nEst40_2021-08-13.txt"
    pPidreg100 ="output/Training/models/phoPidmodel_LGBM_phoPid_reg100phoPidFull_regWeight_nEst100_2021-08-13/phoPidmodel_LGBM_phoPid_reg100phoPidFull_regWeight_nEst100_2021-08-13.txt"
    pPidrev10 ="output/Training/models/phoPidmodel_LGBM_phoPid_rev10phoPidFull_revWeight_nEst10_2021-08-13/phoPidmodel_LGBM_phoPid_rev10phoPidFull_revWeight_nEst10_2021-08-13.txt"
    pPidrev40 ="output/Training/models/phoPidmodel_LGBM_phoPid_rev40phoPidFull_revWeight_nEst40_2021-08-13/phoPidmodel_LGBM_phoPid_rev40phoPidFull_revWeight_nEst40_2021-08-13.txt"
    pPidrev100 ="output/Training/models/phoPidmodel_LGBM_phoPid_rev100phoPidFull_revWeight_nEst100_2021-08-13/phoPidmodel_LGBM_phoPid_rev100phoPidFull_revWeight_nEst100_2021-08-13.txt"

    pIsodata = "output/ReweightFiles/combined_phoIso_phoIsoFull2_2021-07-21_binWidth5_testSize20_validSize20/combined_phoIso_phoIsoFull2_2021-07-21_test.h5"
    pPiddata ="output/ReweightFiles/combined_phoPid_phoPidFull_2021-07-20_binWidth5_testSize20_validSize20/combined_phoPid_phoPidFull_2021-07-20_test.h5"
elif PartType == "e":
    pIsoreg10 ="output/Training/models/eleIsomodel_LGBM_Lepton_reg10eleIsoFull_regWeight_nEst10_2021-07-23/eleIsomodel_LGBM_Lepton_reg10eleIsoFull_regWeight_nEst10_2021-07-23.txt"
    pIsoreg40 ="output/Training/models/eleIsomodel_LGBM_Lepton_reg40eleIsoFull_regWeight_nEst40_2021-08-18/eleIsomodel_LGBM_Lepton_reg40eleIsoFull_regWeight_nEst40_2021-08-18.txt" 
    pIsoreg100 ="output/Training/models/eleIsomodel_LGBM_Lepton_reg100eleIsoFull_regWeight_nEst100_2021-08-18/eleIsomodel_LGBM_Lepton_reg100eleIsoFull_regWeight_nEst100_2021-08-18.txt"
    pIsorev10 ="output/Training/models/eleIsomodel_LGBM_Lepton_rev10eleIsoFull_revWeight_nEst10_2021-08-18/eleIsomodel_LGBM_Lepton_rev10eleIsoFull_revWeight_nEst10_2021-08-18.txt"
    pIsorev40 ="output/Training/models/eleIsomodel_LGBM_Lepton_rev40eleIsoFull_revWeight_nEst40_2021-08-18/eleIsomodel_LGBM_Lepton_rev40eleIsoFull_revWeight_nEst40_2021-08-18.txt"
    pIsorev100 ="output/Training/models/eleIsomodel_LGBM_Lepton_rev100eleIsoFull_revWeight_nEst100_2021-08-18/eleIsomodel_LGBM_Lepton_rev100eleIsoFull_revWeight_nEst100_2021-08-18.txt"
    
    pPidreg10 ="output/Training/models/elePidmodel_LGBM_Lepton_reg10elePidFull_regWeight_nEst10_2021-07-23/elePidmodel_LGBM_Lepton_reg10elePidFull_regWeight_nEst10_2021-07-23.txt"
    pPidreg40 ="output/Training/models/elePidmodel_LGBM_Lepton_reg40elePidFull_regWeight_nEst40_2021-08-18/elePidmodel_LGBM_Lepton_reg40elePidFull_regWeight_nEst40_2021-08-18.txt"
    pPidreg100 ="output/Training/models/elePidmodel_LGBM_Lepton_reg100elePidFull_regWeight_nEst100_2021-08-18/elePidmodel_LGBM_Lepton_reg100elePidFull_regWeight_nEst100_2021-08-18.txt"
    pPidrev10 ="output/Training/models/elePidmodel_LGBM_Lepton_rev10elePidFull_revWeight_nEst10_2021-08-18/elePidmodel_LGBM_Lepton_rev10elePidFull_revWeight_nEst10_2021-08-18.txt"
    pPidrev40 ="output/Training/models/elePidmodel_LGBM_Lepton_rev40elePidFull_revWeight_nEst40_2021-08-18/elePidmodel_LGBM_Lepton_rev40elePidFull_revWeight_nEst40_2021-08-18.txt"
    pPidrev100 ="output/Training/models/elePidmodel_LGBM_Lepton_rev100elePidFull_revWeight_nEst100_2021-08-18/elePidmodel_LGBM_Lepton_rev100elePidFull_revWeight_nEst100_2021-08-18.txt"

    pIsodata = "output/ReweightFiles/combined_Lepton_eleIsoFull_2021-07-23_binWidth5_testSize20_validSize20/combined_Lepton_eleIsoFull_2021-07-23_test.h5"
    pPiddata ="output/ReweightFiles/combined_Lepton_elePidFull_2021-07-23_binWidth5_testSize20_validSize20/combined_Lepton_elePidFull_2021-07-23_test.h5"
elif PartType == "m":
    pIsoreg10 ="output/Training/models/muoIsomodel_LGBM_Lepton_reg10muoIsoFull_regWeight_nEst10_2021-07-23/muoIsomodel_LGBM_Lepton_reg10muoIsoFull_regWeight_nEst10_2021-07-23.txt"
    pIsoreg40 ="output/Training/models/muoIsomodel_LGBM_Lepton_reg40muoIsoFull_regWeight_nEst40_2021-08-18/muoIsomodel_LGBM_Lepton_reg40muoIsoFull_regWeight_nEst40_2021-08-18.txt" 
    pIsoreg100 ="output/Training/models/muoIsomodel_LGBM_Lepton_reg100muoIsoFull_regWeight_nEst100_2021-08-18/muoIsomodel_LGBM_Lepton_reg100muoIsoFull_regWeight_nEst100_2021-08-18.txt"
    pIsorev10 ="output/Training/models/muoIsomodel_LGBM_Lepton_rev10muoIsoFull_revWeight_nEst10_2021-08-18/muoIsomodel_LGBM_Lepton_rev10muoIsoFull_revWeight_nEst10_2021-08-18.txt"
    pIsorev40 ="output/Training/models/muoIsomodel_LGBM_Lepton_rev40muoIsoFull_revWeight_nEst40_2021-08-18/muoIsomodel_LGBM_Lepton_rev40muoIsoFull_revWeight_nEst40_2021-08-18.txt"
    pIsorev100 ="output/Training/models/muoIsomodel_LGBM_Lepton_rev100muoIsoFull_revWeight_nEst100_2021-08-18/muoIsomodel_LGBM_Lepton_rev100muoIsoFull_revWeight_nEst100_2021-08-18.txt"

    pPidreg10 ="output/Training/models/muoPidmodel_LGBM_Lepton_reg10muoPidFull_regWeight_nEst10_2021-07-23/muoPidmodel_LGBM_Lepton_reg10muoPidFull_regWeight_nEst10_2021-07-23.txt"
    pPidreg40 ="output/Training/models/muoPidmodel_LGBM_Lepton_reg40muoIsoFull_regWeight_nEst40_2021-08-18/muoPidmodel_LGBM_Lepton_reg40muoIsoFull_regWeight_nEst40_2021-08-18.txt"
    pPidreg100 ="output/Training/models/muoPidmodel_LGBM_Lepton_reg100muoIsoFull_regWeight_nEst100_2021-08-18/muoPidmodel_LGBM_Lepton_reg100muoIsoFull_regWeight_nEst100_2021-08-18.txt"
    pPidrev10 ="output/Training/models/muoPidmodel_LGBM_Lepton_rev10muoIsoFull_revWeight_nEst10_2021-08-18/muoPidmodel_LGBM_Lepton_rev10muoIsoFull_revWeight_nEst10_2021-08-18.txt"
    pPidrev40 ="output/Training/models/muoPidmodel_LGBM_Lepton_rev40muoIsoFull_revWeight_nEst40_2021-08-18/muoPidmodel_LGBM_Lepton_rev40muoIsoFull_revWeight_nEst40_2021-08-18.txt"
    pPidrev100 ="output/Training/models/muoPidmodel_LGBM_Lepton_rev100muoIsoFull_revWeight_nEst100_2021-08-18/muoPidmodel_LGBM_Lepton_rev100muoIsoFull_revWeight_nEst100_2021-08-18.txt"

    pIsodata = "output/ReweightFiles/combined_Lepton_muoIsoFull_2021-07-22_binWidth5_testSize20_validSize20/combined_Lepton_muoIsoFull_2021-07-22_test.h5"
    pPiddata ="output/ReweightFiles/combined_Lepton_muoPidFull_2021-07-22_binWidth5_testSize20_validSize20/combined_Lepton_muoPidFull_2021-07-22_test.h5"
elif PartType=="ll":
    pIsoreg10 ="output/Training/models/eemodel_LGBM_ZeeFull2_reg10ZeeFull3_regWeight_nEst10_2021-08-19/eemodel_LGBM_ZeeFull2_reg10ZeeFull3_regWeight_nEst10_2021-08-19.txt"
    pIsoreg40 ="output/Training/models/eemodel_LGBM_ZeeFull2_reg40ZeeFull3_regWeight_nEst40_2021-08-19/eemodel_LGBM_ZeeFull2_reg40ZeeFull3_regWeight_nEst40_2021-08-19.txt" 
    pIsoreg100 ="output/Training/models/eemodel_LGBM_ZeeFull2_reg100ZeeFull3_regWeight_nEst100_2021-08-19/eemodel_LGBM_ZeeFull2_reg100ZeeFull3_regWeight_nEst100_2021-08-19.txt"
    pIsorev10 ="output/Training/models/eemodel_LGBM_ZeeFull2_rev10ZeeFull3_revWeight_nEst10_2021-08-19/eemodel_LGBM_ZeeFull2_rev10ZeeFull3_revWeight_nEst10_2021-08-19.txt"
    pIsorev40 ="output/Training/models/eemodel_LGBM_ZeeFull2_rev40ZeeFull3_revWeight_nEst40_2021-08-19/eemodel_LGBM_ZeeFull2_rev40ZeeFull3_revWeight_nEst40_2021-08-19.txt"
    pIsorev100 ="output/Training/models/eemodel_LGBM_ZeeFull2_rev100ZeeFull3_revWeight_nEst100_2021-08-19/eemodel_LGBM_ZeeFull2_rev100ZeeFull3_revWeight_nEst100_2021-08-19.txt"

    pPidreg10 ="output/Training/models/mmmodel_LGBM_ZmmFull2_reg10ZmmFull3_regWeight_nEst10_2021-08-19/mmmodel_LGBM_ZmmFull2_reg10ZmmFull3_regWeight_nEst10_2021-08-19.txt"
    pPidreg40 ="output/Training/models/mmmodel_LGBM_ZmmFull2_reg40ZmmFull3_regWeight_nEst40_2021-08-19/mmmodel_LGBM_ZmmFull2_reg40ZmmFull3_regWeight_nEst40_2021-08-19.txt"
    pPidreg100 ="output/Training/models/mmmodel_LGBM_ZmmFull2_reg100ZmmFull3_regWeight_nEst100_2021-08-19/mmmodel_LGBM_ZmmFull2_reg100ZmmFull3_regWeight_nEst100_2021-08-19.txt"
    pPidrev10 ="output/Training/models/mmmodel_LGBM_ZmmFull2_rev10ZmmFull3_revWeight_nEst10_2021-08-19/mmmodel_LGBM_ZmmFull2_rev10ZmmFull3_revWeight_nEst10_2021-08-19.txt"
    pPidrev40 ="output/Training/models/mmmodel_LGBM_ZmmFull2_rev40ZmmFull3_revWeight_nEst40_2021-08-19/mmmodel_LGBM_ZmmFull2_rev40ZmmFull3_revWeight_nEst40_2021-08-19.txt"
    pPidrev100 ="output/Training/models/mmmodel_LGBM_ZmmFull2_rev100ZmmFull3_revWeight_nEst100_2021-08-19/mmmodel_LGBM_ZmmFull2_rev100ZmmFull3_revWeight_nEst100_2021-08-19.txt"

    pIsodata = "output/ReweightFiles/combined_ZeeFull2_ZeeFull3_2021-08-19_min50_max150_binWidth5_testSize20_validSize20/combined_ZeeFull2_ZeeFull3_2021-08-19_test.h5"
    pPiddata ="output/ReweightFiles/combined_ZmmFull2_ZmmFull3_2021-08-19_min50_max150_binWidth5_testSize20_validSize20/combined_ZmmFull2_ZmmFull3_2021-08-19_test.h5"
    #For this will Iso be ee and Pid will be mm
#%%############################################################################
#   Filenames and directories
###############################################################################

# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"
# timelabel = f"{datetime.date(now)}_{int(timestamp)}"

# Create names for output
output_dir = "output/CompareROC/"+PartType+"_LGBM_compareROC_Final_" + timelabel + "/"
figname = PartType+"_LGBM_compareROC_Final_" + timelabel + "_"

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
print(f'Isolation model:  ')
print(f'         '+PartType+f'Isoreg10:{pIsoreg10}')
print(f'         '+PartType+f'Isoreg40:{pIsoreg40}')
print(f'         '+PartType+f'Isoreg100:{pIsoreg100}')
print(f'         '+PartType+f'Isorev10:{pIsorev10}')
print(f'         '+PartType+f'Isorev40:{pIsorev40}')
print(f'         '+PartType+f'Isorev100:{pIsorev100}')
print(f'PID model:  ')
print(f'         '+PartType+f'PIDreg10:{pPidreg10}')
print(f'         '+PartType+f'PIDreg40:{pPidreg40}')
print(f'         '+PartType+f'PIDreg100:{pPidreg100}')
print(f'         '+PartType+f'PIDrev10:{pPidrev10}')
print(f'         '+PartType+f'PIDrev40:{pPidrev40}')
print(f'         '+PartType+f'PIDrev100:{pPidrev100}')
print(f'All Isolation models have the same data, and the same for Pid')
print(f'N jobs:            {nJobs}')
print(f'------------------------------------------------------------------')



#%%############################################################################
#   Get data
###############################################################################
print(f"Importing data...")
data1 = h5ToDf(pIsodata)
data2 = h5ToDf(pPiddata)
"""
data2['ele_ptvarcone20_rel'] = data2['ele_ptvarcone20'] / data2['ele_et']
data2['ele_ptvarcone40_rel'] = data2['ele_ptvarcone40_TightTTVALooseCone_pt1000'] / data2['ele_et']
data2['ele_topoetcone20_rel'] = data2['ele_topoetcone20'] / data2['ele_et']
data2['ele_topoetcone40_rel'] = data2['ele_topoetcone40'] / data2['ele_et']
"""
X1 = data1[training_varpIso][:]
X2 = data2[training_varpPid][:]

# Choosing the weight from Reg nest10
weightpIsoreg10 = data1[weightNames[0]]
weightpIsoreg40 = data1[weightNames[1]]
weightpIsoreg100 = data1[weightNames[2]]
weightpIsorev10 = data1[weightNames[3]]
weightpIsorev40 = data1[weightNames[4]]
weightpIsorev100 = data1[weightNames[5]]

weightpPidreg10 = data2[weightNames[0]]
weightpPidreg40 = data2[weightNames[1]]
weightpPidreg100 = data2[weightNames[2]]
weightpPidrev10 = data2[weightNames[3]]
weightpPidrev40 = data2[weightNames[4]]
weightpPidrev100 = data2[weightNames[5]]


# Import model
print(f"Import model: {weightNames[0]}")
gbmpIsoreg10 = lgb.Booster(model_file = pIsoreg10)
gbmpIsoreg40 = lgb.Booster(model_file = pIsoreg40)
gbmpIsoreg100 = lgb.Booster(model_file = pIsoreg100)
gbmpIsorev10 = lgb.Booster(model_file = pIsorev10)
gbmpIsorev40 = lgb.Booster(model_file = pIsorev40)
gbmpIsorev100 = lgb.Booster(model_file = pIsorev100)

gbmpPidreg10 = lgb.Booster(model_file = pPidreg10)
gbmpPidreg40 = lgb.Booster(model_file = pPidreg40)
gbmpPidreg100 = lgb.Booster(model_file = pPidreg100)
gbmpPidrev10 = lgb.Booster(model_file = pPidrev10)
gbmpPidrev40 = lgb.Booster(model_file = pPidrev40)
gbmpPidrev100 = lgb.Booster(model_file = pPidrev100)



#%%############################################################################
#   Prediction
###############################################################################
header("Running Predictions")
t = time()

#Predict on all data and save variables in dataframe
print("Predicting on data")

y_pred_pIsoreg10 = gbmpIsoreg10.predict(X1, num_iteration = gbmpIsoreg10.best_iteration, n_jobs = nJobs)
y_pred_pIsoreg40 = gbmpIsoreg40.predict(X1, num_iteration = gbmpIsoreg40.best_iteration, n_jobs = nJobs)
y_pred_pIsoreg100 = gbmpIsoreg100.predict(X1, num_iteration = gbmpIsoreg100.best_iteration, n_jobs = nJobs)
y_pred_pIsorev10 = gbmpIsorev10.predict(X1, num_iteration = gbmpIsorev10.best_iteration, n_jobs = nJobs)
y_pred_pIsorev40 = gbmpIsorev40.predict(X1, num_iteration = gbmpIsorev40.best_iteration, n_jobs = nJobs)
y_pred_pIsorev100 = gbmpIsorev100.predict(X1, num_iteration = gbmpIsorev100.best_iteration, n_jobs = nJobs)

y_pred_pPidreg10 = gbmpPidreg10.predict(X2, num_iteration = gbmpPidreg10.best_iteration, n_jobs = nJobs)
y_pred_pPidreg40 = gbmpPidreg40.predict(X2, num_iteration = gbmpPidreg40.best_iteration, n_jobs = nJobs)
y_pred_pPidreg100 = gbmpPidreg100.predict(X2, num_iteration = gbmpPidreg100.best_iteration, n_jobs = nJobs)
y_pred_pPidrev10 = gbmpPidrev10.predict(X2, num_iteration = gbmpPidrev10.best_iteration, n_jobs = nJobs)
y_pred_pPidrev40 = gbmpPidrev40.predict(X2, num_iteration = gbmpPidrev40.best_iteration, n_jobs = nJobs)
y_pred_pPidrev100 = gbmpPidrev100.predict(X2, num_iteration = gbmpPidrev100.best_iteration, n_jobs = nJobs)

data1["predLGBMreg10"] = y_pred_pIsoreg10
data1["predLGBMreg40"] = y_pred_pIsoreg40
data1["predLGBMreg100"] = y_pred_pIsoreg100
data1["predLGBMrev10"] = y_pred_pIsorev10
data1["predLGBMrev40"] = y_pred_pIsorev40
data1["predLGBMrev100"] = y_pred_pIsorev100

data2["predLGBMreg10"] = y_pred_pPidreg10
data2["predLGBMreg40"] = y_pred_pPidreg40
data2["predLGBMreg100"] = y_pred_pPidreg100
data2["predLGBMrev10"] = y_pred_pPidrev10
data2["predLGBMrev40"] = y_pred_pPidrev40
data2["predLGBMrev100"] = y_pred_pPidrev100

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

plotColumns = ["predLGBMreg10","predLGBMreg40","predLGBMreg100","predLGBMrev10","predLGBMrev40","predLGBMrev100"]
if (PartType == "e") or (PartType=="p") or(PartType=="m"):
    plotNamesISO = ["ML "+PartType+"Isoreg10", "ML "+PartType+"Isoreg40","ML "+PartType+"Isoreg100","ML "+PartType+"Isorev10","ML "+PartType+"Isorev40", "ML "+PartType+"Isorev100"]
    plotNamesPID = ["ML "+PartType+"Pidreg10", "ML "+PartType+"Pidreg40","ML "+PartType+"Pidreg100","ML "+PartType+"Pidrev10","ML "+PartType+"Pidrev40", "ML "+PartType+"Pidrev100"]

    datasetTypes = 2
    setMask1 = data1["dataset"] == datasetTypes
    setMask2 = data2["dataset"] == datasetTypes

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    for i in range(len(plotColumns)):
        fprPlot1, tprPlot1, aucPlot1 = rocVars(data1, plotColumns[i], setMask1, i)
        fprPlot2, tprPlot2, aucPlot2 = rocVars(data2, plotColumns[i], setMask2, i)

        ax[0].plot(tprPlot1,fprPlot1,label=plotNamesISO[i]+f" AUC= {aucPlot1:.3f}")
        ax[1].plot(tprPlot2,fprPlot2,label=plotNamesPID[i]+f" AUC= {aucPlot2:.3f}")

        ax[0].set_ylabel('Background efficiency \n(False positive rate)')
        ax[0].set_xlabel('Signal efficiency \n(True positive rate)')
        ax[1].set_ylabel('Background efficiency \n(False positive rate)')
        ax[1].set_xlabel('Signal efficiency \n(True positive rate)')
        
        ax[0].set_xlim(0,1)
        ax[0].set_ylim(0,1)
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1)


        


    handlesIso, labelsIso = ax[0].get_legend_handles_labels()
    handlesPid, labelsPid = ax[1].get_legend_handles_labels()
    fig.legend(handlesIso, labelsIso, loc = "lower left")
    fig.legend(handlesPid, labelsPid, loc = "lower right")
    fig.suptitle(f"ROC curve - for "+PartType+"Pid and "+PartType+"Iso set")
    fig.tight_layout(rect=[0,0.05, 1, 0.95], h_pad=0.3, w_pad=0.3)
    fig.savefig(output_dir+f"ROCweighted_"+PartType+"Pid_and_"+PartType+"Iso_v1.png")
elif PartType=="ll":
    plotNamesISO = ["ML Zeereg10", "ML Zeereg40","ML Zeereg100","ML Zeerev10","ML Zeerev40", "ML Zeerev100"]
    plotNamesPID = ["ML Zmmreg10", "ML Zmmreg40","ML Zmmreg100","ML Zmmrev10","ML Zmmrev40", "ML Zmmrev100"]

    datasetTypes = 2
    setMask1 = data1["dataset"] == datasetTypes
    setMask2 = data2["dataset"] == datasetTypes

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    for i in range(len(plotColumns)):
        fprPlot1, tprPlot1, aucPlot1 = rocVars(data1, plotColumns[i], setMask1, i)
        fprPlot2, tprPlot2, aucPlot2 = rocVars(data2, plotColumns[i], setMask2, i)

        ax[0].plot(tprPlot1,fprPlot1,label=plotNamesISO[i]+f" AUC= {aucPlot1:.3f}")
        ax[1].plot(tprPlot2,fprPlot2,label=plotNamesPID[i]+f" AUC= {aucPlot2:.3f}")

        ax[0].set_ylabel('Background efficiency \n(False positive rate)')
        ax[0].set_xlabel('Signal efficiency \n(True positive rate)')
        ax[1].set_ylabel('Background efficiency \n(False positive rate)')
        ax[1].set_xlabel('Signal efficiency \n(True positive rate)')

        ax[0].set_xlim(0.8,1)
        ax[0].set_ylim(0,0.2)
        ax[1].set_xlim(0.96,1)
        ax[1].set_ylim(0,0.06)
        

    handlesIso, labelsIso = ax[0].get_legend_handles_labels()
    handlesPid, labelsPid = ax[1].get_legend_handles_labels()
    fig.legend(handlesIso, labelsIso, loc = "lower left")
    fig.legend(handlesPid, labelsPid, loc = "lower right")
    fig.suptitle(f"ROC curve - for Zee and Zmm set")
    fig.tight_layout(rect=[0,0.05, 1, 0.95], h_pad=0.3, w_pad=0.3)
    fig.savefig(output_dir+f"ROCweighted_Zee_and_Zmm_v1.png")


"""
########### LGBM prediction plotting #########
plt_range = [(0,1), (-8,8)]
bins = 100

print(f"Plotting prediction of background and signal for ePid and eIso")
fig, ax = plt.subplots(2,2, figsize=(8,5), sharex='col')
for i in range(2):
    for j in range(2):
        plt_sig1 = data1.loc[(data1[truth_var]==1) & setMask1]['predLGBM']
        plt_bkg1 = data1.loc[(data1[truth_var]==0) & setMask1]['predLGBM']
        plt_sig2 = data2.loc[(data2[truth_var]==1) & setMask2]['predLGBM']
        plt_bkg2 = data2.loc[(data2[truth_var]==0) & setMask2]['predLGBM']
        line = sigSel
        if j == 1:
            plt_sig1 = logit(plt_sig1)
            plt_bkg1 = logit(plt_bkg1)
            plt_sig2 = logit(plt_sig2)
            plt_bkg2 = logit(plt_bkg2)
            line = logit(sigSel)

        ax[i,j].hist(plt_sig1,bins=bins, range=plt_range[j], label="ePid Signal", histtype='step', color="C0")
        ax[i,j].hist(plt_bkg1,bins=bins, range=plt_range[j], label="ePid Background", histtype='step', color="C3")
        ax[i,j].hist(plt_sig2,bins=bins, range=plt_range[j], label="eIso Signal", histtype='step', color="C1")
        ax[i,j].hist(plt_bkg2,bins=bins, range=plt_range[j], label="eIso Background", histtype='step', color="C2")
        # ax[i,j].axvline(line, label="Selection cut", color="k")

    ax[1,i].set_yscale("log")
    ax[1,i].set_ylim(10**0/2, 10**6)
ax[1,0].set_xlabel(f"LGBM prediction")
ax[1,1].set_xlabel(f"Logit(LGBM prediction)")
ax[0,0].set_ylabel(f"Frequency - non log scale")
ax[1,0].set_ylabel(f"Frequency - log scale")

handles, labels = ax[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",ncol=4)
fig.suptitle(f"LGBM prediction - for ePid and eIso")

fig.tight_layout(rect=[0,0.05,1,0.95], h_pad=0.3, w_pad=0.3)
fig.savefig(output_dir + f"LGBM_prediction_ePid_and_eIso.png")

######## 2D logit plot ###########

# dataPath1 and DataPath2 are both from Pid data 
# dataPath2 and DataPath1 are both from Iso data
# So the ones use Pid fit, while twos use Iso fit
DataPath1 = dataPath2
DataPath2 = dataPath1

Data1 = h5ToDf(DataPath1)
Data2 = h5ToDf(DataPath2)

Data2['ele_ptvarcone20_rel'] = Data2['ele_ptvarcone20'] / Data2['ele_et']
Data2['ele_ptvarcone40_rel'] = Data2['ele_ptvarcone40_TightTTVALooseCone_pt1000'] / Data2['ele_et']
Data2['ele_topoetcone20_rel'] = Data2['ele_topoetcone20'] / Data2['ele_et']
Data2['ele_topoetcone40_rel'] = Data2['ele_topoetcone40'] / Data2['ele_et']

x1 = Data1[training_var1][:]
x2 = Data2[training_var2][:]

Weight1 = Data1[weightNames[0]]
Weight2 = Data2[weightNames[0]]

#predict

Y_pred_data1 = gbm1.predict(x1, num_iteration= gbm1.best_iteration, n_jobs= nJobs)
Y_pred_data2 = gbm2.predict(x2, num_iteration= gbm2.best_iteration, n_jobs= nJobs)

Data1["predLGBM"] = Y_pred_data1
Data2["predLGBM"] = Y_pred_data2
data1["logitpredLGBM"] = logit(y_pred_data1)
data2["logitpredLGBM"] = logit(y_pred_data2)
Data1["logitpredLGBM"] = logit(Y_pred_data1)
Data2["logitpredLGBM"] = logit(Y_pred_data2)
data1["modelType"] = "Pid"
data2["modelType"] = "Iso"
Data1["modelType"] = "Pid"
Data2["modelType"] = "Iso"


# Select signal
Data1["selLGBM"] = 0
Data2["selLGBM"] = 0
Data1.loc[(Y_pred_data1>sigSel),["selLGBM"]] = 1
Data2.loc[(Y_pred_data2>sigSel),["selLGBM"]] = 1
SetMask1 = Data1["dataset"] == datasetTypes
SetMask2 = Data2["dataset"] == datasetTypes

binrange = np.linspace(-8,8,100)
print("Plotting 2d logit plot...")

fig, ax = plt.subplots(2,2,figsize=(12,12))
plt_sig1 = logit(data1.loc[(data1[truth_var]==1) & setMask1]['predLGBM'])
plt_bkg1 = logit(data1.loc[(data1[truth_var]==0) & setMask1]['predLGBM'])
plt_sig2 = logit(data2.loc[(data2[truth_var]==1) & setMask2]['predLGBM'])
plt_bkg2 = logit(data2.loc[(data2[truth_var]==0) & setMask2]['predLGBM'])

Plt_sig1 = logit(Data1.loc[(Data1[truth_var]==1) & SetMask1]['predLGBM'])
Plt_bkg1 = logit(Data1.loc[(Data1[truth_var]==0) & SetMask1]['predLGBM'])
Plt_sig2 = logit(Data2.loc[(Data2[truth_var]==1) & SetMask2]['predLGBM'])
Plt_bkg2 = logit(Data2.loc[(Data2[truth_var]==0) & SetMask2]['predLGBM'])

ax[0,0].hist2d(plt_sig1,Plt_sig2,bins = [binrange, binrange])
ax[1,0].hist2d(plt_sig2,Plt_sig1,bins = [binrange, binrange])
ax[0,1].hist2d(plt_bkg1,Plt_bkg2,bins = [binrange, binrange])
ax[1,1].hist2d(plt_bkg2,Plt_bkg1,bins = [binrange, binrange])

ax[0,0].set_xlabel(f"Logit value for Pid variables")
ax[0,0].set_ylabel(f"Logit value for Iso variables")
ax[0,1].set_xlabel(f"Logit value for Pid variables")
ax[0,1].set_ylabel(f"Logit value for Iso variables")
ax[1,0].set_ylabel(f"Logit value for Pid variables")
ax[1,0].set_xlabel(f"Logit value for Iso variables")
ax[1,1].set_ylabel(f"Logit value for Pid variables")
ax[1,1].set_xlabel(f"Logit value for Iso variables")


ax[0,0].title.set_text("Sig for Pid data with Pid and Iso var")
ax[1,0].title.set_text("Sig for Iso data with Iso and Pid var")
ax[0,1].title.set_text("Bkg for Pid data with Pid and Iso var")
ax[1,1].title.set_text("Bkg for Iso data with Iso and Pid var")

fig.savefig(output_dir+f"2dPlot_logit_trans.png")

######## 2d plots combined #########
print("Plotting the combined 2d plot")
fig, ax = plt.subplots(1,2,figsize=(8,4))
plt11 = np.append(plt_sig1, plt_bkg1)
plt12 = np.append(plt_sig2, plt_bkg2)
plt21 = np.append(Plt_sig2, Plt_bkg2)
plt22 = np.append(Plt_sig1, Plt_bkg1)
ax[0].hist2d(plt11,plt21, bins = [binrange, binrange]) #plt_sig1,Plt_sig2,
ax[1].hist2d(plt12,plt22, bins = [binrange, binrange]) #plt_sig2,Plt_sig1
#ax[0].hist2d(plt_bkg1,Plt_bkg2,bins = [binrange, binrange])
#ax[1].hist2d(plt_bkg2,Plt_bkg1,bins = [binrange, binrange])

ax[0].set_xlabel(f"Logit value for Pid variables")
ax[0].set_ylabel(f"Logit value for Iso variables")
ax[1].set_ylabel(f"Logit value for Pid variables")
ax[1].set_xlabel(f"Logit value for Iso variables")

ax[0].title.set_text("Sig and Bkg for Pid data with Pid and Iso var")
ax[1].title.set_text("Sig and Bkg for Iso data with Iso and Pid var")

fig.savefig(output_dir+f"combined_2dPlot_logit_trans.png")
PLT11 = np.append(plt11,plt22)
PLT22 = np.append(plt21,plt12)

fig, ax = plt.subplots(figsize = (8,8))
ax.hist2d(PLT11,PLT22, bins= [binrange, binrange])

ax.set_xlabel(f"Logit value for Pid variables")
ax.set_ylabel(f"Logit value for Iso variables")

ax.title.set_text("Combined")
fig.savefig(output_dir+f"completely_combined_2dLogit.png")


print(len(plt11),len(plt21),len(plt22),len(plt12))
print(np.sum(data1["type"]==Data2["type"]))
print(np.sum(data2["type"]==Data1["type"]))


######## Something Scatter plot? ###### 

print("Plotting PID scatterplot")
FIG = sns.jointplot(x=data1["logitpredLGBM"],y=Data2["logitpredLGBM"],hue=data1["type"],kind="kde")
FIG.set_axis_labels("Logit prediction Pid", "Logit prediction Iso")
FIG.savefig(output_dir+f"ScatterPlot_PidData.png")
print("Plotting first ISO scatterplot")
FIG = sns.jointplot(x=data2["logitpredLGBM"],y=Data1["logitpredLGBM"],hue=data2["type"],kind="kde")
FIG.set_axis_labels("Logit prediction Iso","Logit prediction Pid")
FIG.savefig(output_dir+f"ScatterPlot_IsoData.png")
print("Plotting second ISO scatterplot")
sigmask, bkgmask = data2["type"]==1, data2["type"]==0
sigmask1, bkgmask1 = Data1["type"]==1, Data1["type"]==0
fig, ax = plt.subplots(1,2,figsize=(12,12))
binnum = np.linspace(-10,10,100)
ax[0].hist(data2["logitpredLGBM"][sigmask],bins=binnum,label="Signal",ec = "red",fill=False)
ax[0].hist(data2["logitpredLGBM"][bkgmask],bins=binnum,label="Background", ec = "blue", fill=False)
ax[0].set_title("Iso data with Iso model")
ax[1].hist(Data1["logitpredLGBM"][sigmask1],bins=binnum,label="Signal",ec = "red",fill=False)
ax[1].hist(Data1["logitpredLGBM"][bkgmask1],bins=binnum,label="Background", ec = "blue", fill=False)
ax[1].set_title("Iso data withb Pid model")
fig.savefig(output_dir+f"ScatterPlot_IsoData1D.png")


"""
print(f'')
print(f'')
print(f"END OF PROGRAM - Total time spent: {timediftostr(t_total_start)}")
