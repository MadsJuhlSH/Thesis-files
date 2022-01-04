#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program will be using LightGBM to a boosted decision tree
to differentiate between signal and background 
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



#%%############################################################################
#   Parser
###############################################################################
parser = argparse.ArgumentParser(description='Zee identification with the LightGBM framework.')
parser.add_argument('path', type=str,
                    help='Data file to train on.')
parser.add_argument('--outdir', action='store', default="output/Training/models/", type=str,
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False,
                    help='Tag the data with an additional name.')
parser.add_argument('--njobs', action='store', type=int, required=False, default=10,
                    help='Amount of njobs (default = 10)')
parser.add_argument('--hypopt', action='store', type=int, required=False, default=0,
                    help="""Run Hyper optimization (default = 0).
                            0: Use standard hyperparameters,
                            1: Run hyperparameter optimization,
                            2: Use preoptimized hyperparameters from inpHypopt.""")
parser.add_argument('--hypParam', action='store', type=str, required=False, default="",
                    help='Input hyperparameters: What directory to load hyperparameters from, used in combination with --hypopt = 2.')
parser.add_argument('--train', action='store', type=int, required=False, default=1,
                    help='Run Train (default = True)')
parser.add_argument('--shap', action='store', type=int, required=False, default=1,
                    help='Run SHAP (default = True)')
parser.add_argument('--weights', action='store', type=int, required=True,
                    help=weightsString)
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["eeg", "mmg", "muoIso", "muoPid", "eleIso", "eleIsosimple", "elePid", "phoIso", "phoPid","phoALL", 'ee', 'mm', "gg"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')


args = parser.parse_args()

if ( (args.hypopt == 2) & (args.hypParam == "")):
    log.error("Path to input hyperparameters missing (--hypParam)")
    quit()

if ( (args.train == 0) & (args.shap == 1)):
    log.info("SHAP values cannot be calculated if training is not chosen (--train = 0)")
    log.info("Setting run shap to 0...")
    args.shap = 0

if args.njobs > 20:
    log.error("Excessive number of jobs (--njobs > 20)")
    quit()

log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "eleIso") +(args.PartType != "elePid") + (args.PartType != "muoIso") +(args.PartType != "muoPid")+ (args.PartType != "eeg") + (args.PartType != "mmg") + (args.PartType != "phoIso")+(args.PartType != "phoPid")+(args.PartType != "ee")+(args.PartType != "mm")+(args.PartType !="phoALL")+(args.PartType !="eleIsosimple")+(args.PartType !="gg"))!=12:
    log.error("Unknown lepton, use either eleIso, eleIsosimple, elePid, muoIso, muoPid, eeg, mmg, ee, mm, phoALL, gg, phoPid or phoIso")
    quit()




#%%############################################################################
#   Filenames and directories
###############################################################################
# Name of data file
filename_base = os.path.basename(args.path)
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
if args.PartType == "eeg":
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
    'pho_r33over37allcalo']"""
elif args.PartType =="phoALL":
    training_var = [#'pho_isPhotonEMLoose',
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
    ['NvtxReco',
                    'correctedScaledAverageMu',
                    'pho_isPhotonEMLoose',
                    #'pho_isPhotonEMThight',
                    'pho_e',
                    'pho_phi',
                    'pho_wtots1',
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
                    'pho_z0',
                    'pho_z0Sig',
                    'pho_core57cellsEnergyCorrection',
                    'pho_r33over37allcalo',
                    'pho_topoetcone20',
                    'pho_topoetcone40',
                    'pho_ptvarcone20']"""
""" mine
    training_var = ['pho_eta',
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
                    """
                        


# Set the fixed parameters for LGBM
params_set = {
    'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
    'objective': 'binary',          # Probability labeĺs in [0,1]
    'boost_from_average': True,
    'verbose': 0,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
    'num_threads': args.njobs,
    }

hyperopt_options = {0 : "Use standard hyperparameters",
                    1 : "Run hyperparameter optimization",
                    2 : f"Use preoptimized hyperparameters from {args.hypParam}"}

# Set dataset type: 0 = train, 1 = valid, 2 = test
datatype = {0 : "train",
            1 : "valid",
            2 : "test"}

#%%############################################################################
#   Print initial log info
###############################################################################
# Log INFO for run
print(f'')
print(f'---------------------------- INFO --------------------------------')
print(f'Datafile:                 {args.path}')
print(f'Output directory:         {output_dir}')
print(f'Hyperparameters:          {hyperopt_options[args.hypopt]}')
print(f'Run Training:             {bool(args.train)}')
print(f'Run SHAP:                 {bool(args.shap)}')
print(f'Number of threads:        {args.njobs}')
print(f'Particle type:            {args.PartType}')
print(f'------------------------------------------------------------------')


#%%############################################################################
#   Importing data.
###############################################################################
header("Importing and separating data")
t = time()

# Get hdf5 datafile as dataframe
data = h5ToDf(args.path)

# Print column names to log
print_col_names(data.columns, training_var, truth_var)

# Separate label and training data
X_train = data[training_var][data["dataset"] == 0]
y_train = data[truth_var][data["dataset"] == 0]
X_valid = data[training_var][data["dataset"] == 1]
y_valid = data[truth_var][data["dataset"] == 1]

# Get list of unique dataset types
datasetTypes = np.sort(data["dataset"].unique())

# Create lgbm dataset
lgb_train = lgb.Dataset(X_train, y_train, weight = data[weightName][data["dataset"]==0])
lgb_valid = lgb.Dataset(X_valid, y_valid, weight = data[weightName][data["dataset"] == 1], reference = lgb_train)

footer("Importing and separating data", t)

#%%############################################################################
#   Perform Hyper Parameter tuning and save parameters
###############################################################################
if (args.hypopt == 0):
    header(f"Hyper Parameter Tuning: {hyperopt_options[args.hypopt]}")
    params_add = {
        'num_leaves' : 30,
        'max_depth' : -1, 
        'min_data_in_leaf' : 30,
        'feature_fraction' : 1.0,
        'bagging_fraction' : 1.0,
        'bagging_freq' : 0}
        
elif (args.hypopt == 1):
    header(f"Hyper Parameter Tuning: {hyperopt_options[args.hypopt]}")
    t = time()

    # Set ranges for random search
    n_iter = 20
    n_fold = 5
    l_rate = 0.1
    n_boost_round = 500
    e_stop_round = 100
    params_grid = {
        'num_leaves' : randint(20,40),                    # Important! Default: 31, set to less than 2^(max_depth)
        'max_depth': randint(-20,20),                       # Important! Default: -1, <= 0 means no limit
        'min_data_in_leaf': randint(10,100),              # Important! Default: 20
        'feature_fraction': [1.0],                        # Default: 1.0, random selection if it is under 1 eg. 80% of features for 0.8, helps with: speed up, over-fitting
        'bagging_fraction': [1.0],                        # Default: 1.0
        'bagging_freq': [0],                              # Default: 0, bags data, should be combined vith bagging_fraction
        }  
    
    # Perform random search
    best, df_cv = HyperOpt_RandSearch(lgb_train,
                             params_set,
                             params_grid,
                             learning_rate=l_rate,
                             n_iter=n_iter,
                             n_fold=n_fold,
                             n_boost_round=n_boost_round,
                             e_stop_round=e_stop_round,
                             verbose=False)

    params_add = df_cv['params'][best]
    print(f"Best iteration: [{best}/{n_iter}]")

    # Save best hyperparameters 
    print('Saving best hyperparameters...')
    np.save(output_dir + hypparamname + '_best.npy', params_add)

    # Save all hyperparameters
    print('Saving all hyperparameters...')
    params_return = np.array(df_cv[['cv_mean', 'cv_stdv', 'num_boost_round','num_leaves','min_data_in_leaf','max_depth']])
    np.savetxt(output_dir + hypparamname + '.txt', params_return)

    # Plot all hyperparameters
    print("Plotting hyperparameters...")
    rand_param = df_cv[['cv_mean', 'cv_stdv', 'num_boost_round','num_leaves','min_data_in_leaf','max_depth']]

    range_num_leaves = np.arange(20, 41, 1).astype(int)
    range_min_data_in_leaf = np.arange(10, 101, 1).astype(int)
    range_max_depth = np.arange(-20,21,1).astype(int)

    fig, ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
    im1 = HeatMap_rand(fig,ax[0],rand_param["cv_mean"],rand_param["cv_stdv"],rand_param["num_leaves"],rand_param["min_data_in_leaf"], range_num_leaves, range_min_data_in_leaf)
    im2 = HeatMap_rand(fig,ax[1],rand_param["cv_mean"],rand_param["cv_stdv"],rand_param["num_leaves"],rand_param["max_depth"], range_num_leaves, range_max_depth)
    ax[0].set_xlabel("min_data_in_leaf")
    ax[1].set_xlabel("max_depth")
    ax[0].set_ylabel("num_leaves")

    fig.colorbar(im1, ax=ax.ravel().tolist())#, format='%.2e')

    fig.suptitle("Random search - negative AUC")
    fig.tight_layout(rect=[0,0,0.78,0.95], h_pad=0.1, w_pad=0.1)
    fig.savefig(figure_dir + figname + "RandSearch.png")
    plt.close(fig)

    footer("Hyper Parameter Tuning",t)
elif (args.hypopt == 2):
    header(f"Hyper Parameter Tuning: {hyperopt_options[args.hypopt]}")

    print(f"Importing parameters from: {args.hypParam}")
    params_add = np.load( args.hypParam, allow_pickle='TRUE').item()

params = params_set
params.update(params_add)

print(f"Parameters:")
keys = list(params.keys())
for i in range(len(keys)):
    print(f"            {keys[i]}: {params[keys[i]]}")

#%%############################################################################
#   Train on entire training set
###############################################################################
if args.train:
    header("Running Training")
    t = time()

    # Set parameters 
    evals_result = {}   # to record eval results for plotting
    l_rate = 0.05
    n_boost_round = 2500
    e_stop_round = 500

    params.update(learning_rate = l_rate)

    # Print parameters
    print(f"Learning rate = {l_rate}, n_boost_round = {n_boost_round}, e_stop_round = {e_stop_round}")

    # Train

    gbm = lgb.train(params,
                    lgb_train, 
                    feval = auc_eval,
                    num_boost_round = n_boost_round,
                    valid_sets  = [lgb_train, lgb_valid],
                    early_stopping_rounds = e_stop_round,
                    evals_result = evals_result)

    footer("Training", t)
    print()

    # Save model
    print('Saving model...')
    gbm.save_model(output_dir + modelname + ".txt")

    # Save plots
    print('Plotting metrics recorded during training...')
    fig_metric, ax_metric = plt.subplots(1,2, figsize=(6,3), sharex = True)
    metrics = ['binary_logloss', 'auc_eval']
    metricNames = ['Binary logloss', 'AUC']
    metricColor = ["C0", "C1"]

    for i,metric in enumerate(metrics):
        metric_train = evals_result['training'][metric]
        metric_valid = evals_result['valid_1'][metric]
        ax_metric[i].plot(metric_train, label="Training", color = metricColor[i])
        ax_metric[i].plot(metric_valid, label="Validation", color = metricColor[i], linestyle = "--")
        ax_metric[i].axvline(gbm.best_iteration, color='k', alpha=0.5, label = "Best iteration")
        ax_metric[i].set_xlabel("Iterations")
        ax_metric[i].set_ylabel(metricNames[i], color = metricColor[i])
        ax_metric[i].legend(loc = "upper right")
        ax_metric[i].tick_params(axis='y', labelcolor=metricColor[i])
    ax_metric[0].set_yscale('log')
    plt.tight_layout()
    fig_metric.savefig(figure_dir + figname + 'trainingMetric_v1.png')
    plt.close(fig_metric)

    print('Plotting metrics recorded during training...')
    fig_metric, ax_metric = plt.subplots(2,1,figsize=(3,4),sharex=True)
    metrics = ['binary_logloss','auc_eval']
    metricNames = ['Binary logloss','AUC']
    ax_metric[1].set_xlabel("Iterations")
    for i, metric in enumerate(metrics):
        metric_train = evals_result['training'][metric]
        metric_valid = evals_result['valid_1'][metric]
        ax_metric[i].plot(metric_train, label="Training", color=metricColor[i])
        ax_metric[i].plot(metric_valid, label="Validation", color=metricColor[i], linestyle="--")
        ax_metric[i].axvline(gbm.best_iteration, color="k", alpha=0.5, label="Best iteration")
        ax_metric[i].legend(loc="upper right")
        ax_metric[i].text(0.01,1.01,metricNames[i], ha='left', va='bottom',transform=ax_metric[i].transAxes, color=metricColor[i])
        ax_metric[i].tick_params(axis='y', labelcolor=metricColor[i])

    fig_metric.tight_layout(rect=[0,0,1,1], h_pad=0.3, w_pad=1.0)
    fig_metric.savefig(figure_dir + figname + 'trainingMetric_v2.png')
    plt.close(fig_metric)


    print('Plotting and saving feature importances...')
    # Get feature importance values and sort
    featImp = pd.DataFrame(gbm.feature_importance(), index = gbm.feature_name(), columns=['feature importance'])
    featImp = featImp.sort_values(by=['feature importance'], ascending = False)

    # Get sorted names and positions for plot
    featImpName = list(featImp.index.values)
    featImp_pos = np.arange(len(featImpName))

    # Plot feature importance
    fig_featImp, ax_featImp = plt.subplots(figsize=(3,4))
    ax_featImp.barh(featImp_pos, featImp['feature importance'], align='center')
    ax_featImp.set_yticks(featImp_pos)
    ax_featImp.set_yticklabels(featImpName,fontsize=9)
    ax_featImp.invert_yaxis()  # labels read top-to-bottom

    fig_featImp.suptitle('LGBM feature importance')
    #fig_featImp.tight_layout(rect=[0,0,1,0.95], h_pad=0.1, w_pad=0.1)
    fig_featImp.savefig(figure_dir + figname + 'featureImportance_LGBM.png')
    plt.close(fig_featImp)

    # Predict on training and validation set
    print()
    print('Predicting on training and validation set')
    y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration, n_jobs=args.njobs)
    y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration, n_jobs=args.njobs)

    print('AUC score of prediction:')
    print(f"        Training:   {roc_auc_score(y_train, y_pred_train):.6f}")
    print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid):.6f}")
    print('AUC score of prediction (weighted):')
    print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data[weightName][data['dataset']==0]):.6f}")
    print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data[weightName][data['dataset']==1]):.6f}")

elif not args.train:
    header("Selected NOT to run Training")
    print(f'')

# #%%############################################################################
# #   Calculate SHAP values
# ###############################################################################
if args.shap:
    header('Running SHAP Feature Importance')
    t = time()

    # Get SHAP values
    print('Calculating SHAP values...')
    shap_values = shap.TreeExplainer(gbm).shap_values(X_train)
    
    #explainer = shap.TreeExplainer(gbm)
    #shap_values = explainer.shap_values(X_train)
    footer("Calculation",t)
    
    # Get mean abs values sorted
    #print('Sorting SHAP values...')
    
    
    #shap_values_df = pd.DataFrame(shap_values, columns = training_var)
    #shap_val = shap_values_df.abs().mean(0)
    #shap_val_sort = shap_val.sort_values(0, ascending = False)
    
    # Get names of values and positions for plot

    fig_shap=plt.figure(figsize=(16,12))
    shap.summary_plot(shap_values, X_train, plot_type="bar",plot_size=(16,12))
    fig_shap.savefig(figure_dir+figname+'featureImportance_SHAP.png')
    plt.close(fig_shap)
    shap_name_sort = list(shap_val_sort.index.values)
    shap_pos = np.arange(len(shap_name_sort))

    # Plot shap values
    print('Plotting SHAP values...')
    fig_shap, ax_shap = plt.subplots(figsize=(3,4))
    ax_shap.barh(shap_pos, shap_val_sort, align='center')
    ax_shap.set_yticks(shap_pos)
    ax_shap.set_yticklabels(shap_name_sort,fontsize=9)
    ax_shap.invert_yaxis()  # labels read top-to-bottom
    fig_shap.suptitle('SHAP values')
    fig_shap.tight_layout(rect=[0,0,1,0.95], h_pad=0.3, w_pad=0.3)
    fig_shap.savefig(figure_dir + figname + 'featureImportance_SHAP.png')
    
    plt.close(fig_shap)

    footer("SHAP Feature Importance",t)

    # Plot shap values
    print('Plotting feature importance together...')
    fig, ax = plt.subplots(1,2,figsize=(6,4))
    ax[0].barh(shap_pos, shap_val_sort, align='center')
    ax[0].set_yticks(shap_pos)
    ax[0].set_yticklabels(shap_name_sort,fontsize=9)
    ax[0].invert_yaxis()  # labels read top-to-bottom
    ax[0].set_title('SHAP')
    ax[1].barh(featImp_pos, featImp['feature importance'], align='center')
    ax[1].set_yticks(featImp_pos)
    ax[1].set_yticklabels(featImpName,fontsize=9)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_title('LightGBM')
    plt.tight_layout()
    fig.savefig(figure_dir + figname + 'featureImportance_together.png')
    plt.close(fig)


print(f'')
print(f'')
print(f"END OF PROGRAM - Total time spent: {timediftostr(t_total_start)}")
