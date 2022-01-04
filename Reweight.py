#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reweight signal/background after the files have been combined
I have not included forward electrons yet.
"""
print("Program running...")
# import warnings
# warnings.filterwarnings('ignore', 'ROOT .+ is currently active but you ')
# warnings.filterwarnings('ignore', 'numpy .+ is currently installed but you ')

import h5py
import numpy as np
import logging as log
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import lightgbm as lgb

from utils import mkdir, h5ToDf
from time import time
from datetime import timedelta
from datetime import datetime
from scipy.special import logit
from sklearn.model_selection import train_test_split
from hep_ml.reweight import GBReweighter

from returnLGBMScore import eIsoScore, ePidScore, mIsoScore, mPidScore, pIsoScore, pPidScore, ZmmScore, ZeeScore

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/ReweightFiles/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--invMmin', action='store', default=50.0, type=float,
                    help='Minimum value of invariant mass. (Default = 50.0)')
parser.add_argument('--invMmax', action='store', default=150.0, type=float,
                    help='Maximum value of invariant mass. (Default = 150.0)')
parser.add_argument('--etmin', action='store', default=4.5, type=float,
                    help='Minimum value of et. (Default = 4.5)')
parser.add_argument('--binWidth', action='store', default=5, type=float,
                    help='Width in GeV. (Default = 5)')
parser.add_argument('--testSize', action='store', default=0.2, type=float,
                    help='Size of test set from 0.0 to 1.0. (Default = 0.2)')
parser.add_argument('--validSize', action='store', default=0.2, type=float,
                    help='Size of validation set from 0.0 to 1.0. Split after separation into test and training set. (Default = 0.2)')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--nJobs', action='store', type=int, required=False, default=5,
                    help='Number of jobs (Default 5)')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["eeg", "mmg", 'ee', 'mm', "muo", "ele", "pho", "gg"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')



args = parser.parse_args()

log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "ele") + (args.PartType != "muo") + (args.PartType != "eeg") + (args.PartType != "mmg") + (args.PartType != "pho")+(args.PartType != "ee")+(args.PartType != "mm")+(args.PartType != "gg"))!=7:
    log.error("Unknown lepton, use either ele, muo, gg, eeg, mmg or pho")
    quit()


#Models required for files with multiple signal particles such as Zee and Zmm
#This would require prior training of these models.
eIsoModel = "output/Training/models/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso240921v2_revWeight_nEst40_2021-09-24/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso240921v2_revWeight_nEst40_2021-09-24.txt"
ePidModel = "output/Training/models/elePidmodel_LGBM_LeptonLepton_eleReg40Pid240921_regWeight_nEst40_2021-09-24/elePidmodel_LGBM_LeptonLepton_eleReg40Pid240921_regWeight_nEst40_2021-09-24.txt"
mIsoModel = "output/Training/models/muoIsomodel_LGBM_Lepton_muoRev10Iso021021_revWeight_nEst10_2021-10-02/muoIsomodel_LGBM_Lepton_muoRev10Iso021021_revWeight_nEst10_2021-10-02.txt"
mPidModel = "output/Training/models/muoPidmodel_LGBM_Lepton_muoReg40Pid180921_regWeight_nEst40_2021-09-18/muoPidmodel_LGBM_Lepton_muoReg40Pid180921_regWeight_nEst40_2021-09-18.txt"
pIsoModel = "output/Training/models/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23.txt"
pPidModel = "output/Training/models/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29.txt"

eemodel = "output/Training/models/eemodel_LGBM_Zee130921Zeeg240921_eeReg40280921_regWeight_nEst40_2021-09-28/eemodel_LGBM_Zee130921Zeeg240921_eeReg40280921_regWeight_nEst40_2021-09-28.txt"
mmmodel = "output/Training/models/mmmodel_LGBM_Zmm130921Zmmg240921_mmReg40021021_regWeight_nEst40_2021-10-03/mmmodel_LGBM_Zmm130921Zmmg240921_mmReg40021021_regWeight_nEst40_2021-10-03.txt"

#============================================================================
# Functions
#============================================================================
# Get reverse and regular weights.

def getReverseWeights(datatype, reweighter, data):
    # Inspired from ElePairReweight.py from mdv971 (Helle Leerberg)
    # predict the weights
    if args.PartType == "eeg":
        total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] > 0.5],
                                                            data["et"][data["type"] > 0.5],
                                                            data["invM"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    if args.PartType == "ee":
        total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] > 0.5],
                                                            data["et"][data["type"] > 0.5],
                                                            data["invM"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "mmg":
            total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] > 0.5],
                                                            data["pt"][data["type"] > 0.5],
                                                            data["invM"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "mm":
            total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] > 0.5],
                                                            data["pt"][data["type"] > 0.5],
                                                            data["invM"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "gg":
            total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] > 0.5],
                                                            data["et"][data["type"] > 0.5],
                                                            data["invM"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "ele":
        total_weight = reweighter.predict_weights(np.array([data["ele_eta"][data["type"] > 0.5],
                                                            data["ele_et"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "muo":
        total_weight = reweighter.predict_weights(np.array([data["muo_eta"][data["type"] > 0.5],
                                                            data["muo_pt"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
    elif args.PartType == "pho":
        total_weight = reweighter.predict_weights(np.array([data["pho_eta"][data["type"] > 0.5],
                                                            data["pho_et"][data["type"] > 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] > 0.5]]).T)
                                                        

    log.info(f"Prediction of weights for {datatype} is done.")

    # Get the ratio of sig and bkg weights to scale the bkg to have the same number of events ( after weighting)
    log.info(f"[Data]   shape: {np.shape(data['type'])}, sum = {np.sum(data['type'])}, sum[<=0.5] = {np.sum(data['type'] <= 0.5)}")
    log.info(f"[Weight] shape: {np.shape(total_weight)}, sum = {np.sum(total_weight)}")

    ratio = np.sum(data["type"] <= 0.5)/np.sum(total_weight)
    log.info(f"Ratio: {ratio}")

    # Set array for weights to 1 (Signal gets weight 1)
    weight = np.ones(len(data["correctedScaledAverageMu"]))

    # Get weights for background
    weight[data["type"] > 0.5] = total_weight * ratio

    # Return the weights normalized to a mean of one, since this is how keras likes it
    return weight, (weight / np.mean(weight))

def getRegularWeights(datatype, reweighter, data):
    # Inspired from ElePairReweight.py from mdv971 (Helle Leerberg)
    # Predict the weights
    if args.PartType == "eeg":
        total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] < 0.5],
                                                            data["et"][data["type"] < 0.5],
                                                            data["invM"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    if args.PartType == "ee":
        total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] < 0.5],
                                                            data["et"][data["type"] < 0.5],
                                                            data["invM"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    if args.PartType == "gg":
        total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] < 0.5],
                                                            data["et"][data["type"] < 0.5],
                                                            data["invM"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)                                                        
    elif args.PartType == "mmg":
            total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] < 0.5],
                                                            data["pt"][data["type"] < 0.5],
                                                            data["invM"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    elif args.PartType == "mm":
            total_weight = reweighter.predict_weights(np.array([data["eta"][data["type"] < 0.5],
                                                            data["pt"][data["type"] < 0.5],
                                                            data["invM"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    elif args.PartType == "ele":
        total_weight = reweighter.predict_weights(np.array([data["ele_eta"][data["type"] < 0.5],
                                                            data["ele_et"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    elif args.PartType == "muo":
        total_weight = reweighter.predict_weights(np.array([data["muo_eta"][data["type"] < 0.5],
                                                            data["muo_pt"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    elif args.PartType == "pho":
        total_weight = reweighter.predict_weights(np.array([data["pho_eta"][data["type"] < 0.5],
                                                            data["pho_et"][data["type"] < 0.5],
                                                            data["correctedScaledAverageMu"][data["type"] < 0.5]]).T)
    log.info(f'Prediction of weights for {datatype} is done')

    # Get the ratio of sig and bkg weights to scale the bkg to have the same number of events ( after weighting )
    log.info(f"[Data]   shape: {np.shape(data['type'])}, sum = {np.sum(data['type'])}, sum[>=0.5] = {np.sum(data['type'] >= 0.5)}")
    log.info(f"[Weight] shape: {np.shape(total_weight)}, sum = {np.sum(total_weight)}")

    ratio = np.sum(data["type"] >= 0.5)/np.sum(total_weight)
    log.info(f"Ratio: {ratio}")

    # Set array for weights to 1 ( Signal gets weight 1)
    weight = np.ones(len(data["correctedScaledAverageMu"]))

    # Get weights for background
    weight[data["type"] < 0.5] = total_weight * ratio

    # Return the weights normalized to a mean of one, since this is how keras likes it
    return weight, (weight/np.mean(weight))

#============================================================================
# Import
#============================================================================
filenames = []
data_name = []
data_list = []
fname = "combined_"

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
    data_get = h5ToDf(path)
    data_get["process"] = name[0]
    data_list.append(data_get)

    # Combine names for new filename
    fname = fname + name[0]

# Create timestamp for name
now = datetime.now()
timelabel = f"{datetime.date(now)}"

print(data_name)

if args.tag:
    fname = fname + "_" + args.tag + "_" + timelabel
elif not args.tag:
    fname = fname + "_" + timelabel

#============================================================================
# Create folder
#============================================================================
if (args.PartType == "eeg") or (args.PartType == "mmg") or (args.PartType == "ee") or (args.PartType == "mm"):
    args.outdir = args.outdir+fname+f"_min{int(args.invMmin)}_max{int(args.invMmax)}_binWidth{int(args.binWidth)}_testSize{int(args.testSize*100)}_validSize{int(args.validSize*100)}/"
else:
    args.outdir = args.outdir+fname+f"_binWidth{int(args.binWidth)}_testSize{int(args.testSize*100)}_validSize{int(args.validSize*100)}/"
if os.path.exists(args.outdir):
    log.error(f"Output already exists - please remove yourself. Output: {args.outdir}")
    quit()
else:
    log.info(f"Creating output folder: {args.outdir}")
    mkdir(args.outdir)

#============================================================================
# Combine files, give labels and cut
#============================================================================
# Concatenate data and shuffle
data_all = pd.concat(data_list, ignore_index = True)
data_all = data_all.sample(frac = 1, random_state = 0).reset_index(drop=True) #Shuffle
#data = data_all.copy()
DataSig = data_all[data_all["type"]==1]
DataBkg = data_all[data_all["type"]==0]
print("The variables in the files are:")
print(data_all.keys())
print(f"There are {DataSig.shape[0]} signal and {DataBkg.shape[0]} background before additional background")
counter = 0
while DataBkg.shape[0]/DataSig.shape[0]<0.9:
    counter +=1
    #print(counter)
    databkg = data_all[data_all["type"]==0]
    DataBkg = pd.concat([DataBkg,databkg], ignore_index=True)
    #print(f"There are now {DataBkg.shape[0]} background events after appending")
    if counter%10==0:
        print(f"Counter for additional background: {counter}")
    if counter == 100:
        break
data = pd.concat([DataSig,DataBkg],ignore_index=True)

if (args.PartType == "eeg") or (args.PartType=="mmg") or (args.PartType == "ee") or (args.PartType == "mm") or (args.PartType == "gg"):
    # Get data in mass range: (invMin, invMax)
    log.info(f"Select invariant mass range (min = {args.invMmin}, max = {args.invMmax})")
    shapeAllBefore = np.shape(data_all)
    shapeSigBefore = np.shape(data[data["type"] == 1])
    shapeBkgBefore = np.shape(data[data["type"] == 0])

    data = data[data.invM > args.invMmin]

    shapeAllAfter1 = np.shape(data_all)
    shapeSigAfter1 = np.shape(data[data["type"] == 1])
    shapeBkgAfter1 = np.shape(data[data["type"] == 0])

    data = data[data.invM < args.invMmax]

    shapeAllAfter2 = np.shape(data_all)
    shapeSigAfter2 = np.shape(data[data["type"] == 1])
    shapeBkgAfter2 = np.shape(data[data["type"] == 0])

    log.info(f"Shape all:        Before: {shapeAllBefore}")
    log.info(f"                  after >{args.invMmin} cut: {shapeAllAfter1}, relative {shapeAllAfter1[0]/shapeAllBefore[0]}")
    log.info(f"                  after <{args.invMmax} cut: {shapeAllAfter2}, relative {shapeAllAfter2[0]/shapeAllAfter1[0]}")
    log.info(f"Shape signal:     Before: {shapeSigBefore}")
    log.info(f"                  after >{args.invMmin} cut: {shapeSigAfter1}, relative {shapeSigAfter1[0]/shapeSigBefore[0]}")
    log.info(f"                  after <{args.invMmax} cut: {shapeSigAfter2}, relative {shapeSigAfter2[0]/shapeSigAfter1[0]}")
    log.info(f"Shape background: Before: {shapeBkgBefore}")
    log.info(f"                  after >{args.invMmin} cut: {shapeBkgAfter1}, relative {shapeBkgAfter1[0]/shapeBkgBefore[0]}")
    log.info(f"                  after <{args.invMmax} cut: {shapeBkgAfter2}, relative {shapeBkgAfter2[0]/shapeBkgAfter1[0]}")
elif (args.PartType == "ele") or (args.PartType=="muo") or (args.PartType == "pho"):
    # Get data in mass range: (invMin, invMax)
    log.info(f"Select invariant et/pt range (min = {args.etmin})")
    shapeAllBefore = np.shape(data_all)
    shapeSigBefore = np.shape(data[data["type"] == 1])
    shapeBkgBefore = np.shape(data[data["type"] == 0])
    
    if args.PartType =="ele":
        data = data[data.ele_et > args.etmin]
    elif args.PartType =="muo":
        data = data[data.muo_pt > args.etmin]
    elif args.PartType =="ele":
        data = data[data.pho_et > args.etmin]
    shapeAllAfter1 = np.shape(data)
    shapeSigAfter1 = np.shape(data[data["type"] == 1])
    shapeBkgAfter1 = np.shape(data[data["type"] == 0])

    
    log.info(f"Shape all:        Before: {shapeAllBefore}")
    log.info(f"                  after >{args.etmin} cut: {shapeAllAfter1}, relative {shapeAllAfter1[0]/shapeAllBefore[0]}")
    log.info(f"Shape signal:     Before: {shapeSigBefore}")
    log.info(f"                  after >{args.etmin} cut: {shapeSigAfter1}, relative {shapeSigAfter1[0]/shapeSigBefore[0]}")
    log.info(f"Shape background: Before: {shapeBkgBefore}")
    log.info(f"                  after >{args.etmin} cut: {shapeBkgAfter1}, relative {shapeBkgAfter1[0]/shapeBkgBefore[0]}")

#============================================================================
# Add columns
#============================================================================

# Add relative cones
if (args.PartType == "eeg") or (args.PartType == "ee"):
    data["ele_deltaZ0"] = np.abs(data["ele1_z0"]-data["ele2_z0"])
    data["ele_deltaZ0sig"] = np.sqrt(data["ele1_z0Sig"]**2+data["ele2_z0Sig"]**2)
    data["ele_deltad0"] = np.abs(data["ele1_d0"]-data["ele2_d0"])
    data["ele_deltad0sig"] = np.sqrt(data["ele1_d0Sig"]**2+data["ele2_d0Sig"]**2)
    data['ele1_ptvarcone40_rel'] = data['ele1_ptvarcone40_TightTTVALooseCone_pt1000']/data['ele1_et']
    data['ele1_topoetcone40_rel'] = data['ele1_topoetcone40']/data['ele1_et']
    data['ele2_ptvarcone40_rel'] = data['ele2_ptvarcone40_TightTTVALooseCone_pt1000']/data['ele2_et']
    data['ele2_topoetcone40_rel'] = data['ele2_topoetcone40']/data['ele2_et']

    data['ele1_ptvarcone20_rel'] = data['ele1_ptvarcone20'] / data['ele1_et']
    data['ele1_topoetcone20_rel'] = data['ele1_topoetcone20'] / data['ele1_et']
    data['ele2_ptvarcone20_rel'] = data['ele2_ptvarcone20'] / data['ele2_et']
    data['ele2_topoetcone20_rel'] = data['ele2_topoetcone20'] / data['ele2_et']
elif (args.PartType == "mmg") or (args.PartType == "mm"):
    data['muo1_ptvarcone40_rel'] = data['muo1_ptvarcone40']/data['muo1_pt']
    #data['muo1_topoetcone40_rel'] = data['ele1_topoetcone40']/data['ele1_et']
    data['muo2_ptvarcone40_rel'] = data['muo2_ptvarcone40']/data['muo2_pt']
    #data['muo2_topoetcone40_rel'] = data['ele2_topoetcone40']/data['ele2_et']

    data['muo1_ptvarcone20_rel'] = data['muo1_ptvarcone20'] / data['muo1_pt']
    #data['muo1_topoetcone20_rel'] = data['ele1_topoetcone20'] / data['ele1_et']
    data['muo2_ptvarcone20_rel'] = data['muo2_ptvarcone20'] / data['muo2_pt']
    #data['muo2_topoetcone20_rel'] = data['ele2_topoetcone20'] / data['ele2_et']
elif args.PartType == "ele":
    data['ele_ptvarcone40_rel'] = data['ele_ptvarcone40_TightTTVALooseCone_pt1000']/data['ele_et']
    data['ele_topoetcone40_rel'] = data['ele_topoetcone40']/data['ele_et']
    data['ele_ptvarcone20_rel'] = data['ele_ptvarcone20'] / data['ele_et']
    data['ele_topoetcone20_rel'] = data['ele_topoetcone20'] / data['ele_et']
elif args.PartType == "muo":
    data['muo_ptvarcone40_rel'] = data['muo_ptvarcone40']/data['muo_pt']
    data['muo_ptvarcone20_rel'] = data['muo_ptvarcone20'] / data['muo_pt']
elif args.PartType == "pho":
    data['pho_topoetcone40_rel'] = data['pho_topoetcone40']/data['pho_et']
    data['pho_ptvarcone20_rel'] = data['pho_ptvarcone20'] / data['pho_et']
    data['pho_topoetcone20_rel'] = data['pho_topoetcone20'] / data['pho_et']
elif args.PartType =="gg":
    data["pho_deltaZ0"] = np.abs(data["pho1_z0"]-data["pho2_z0"])
    data["pho_deltaZ0sig"] = np.sqrt(data["pho1_z0Sig"]**2+data["pho2_z0Sig"]**2)

if (args.PartType == "eeg") or (args.PartType == "ee"):
    # Add ATLAS isolation
    log.info(f"Add ATLAS isolation to data")
    data['ele1_atlasIso_ptvarcone20'] = ( data['ele1_ptvarcone20'] / data['ele1_e'] < 0.15 )
    data['ele2_atlasIso_ptvarcone20'] = ( data['ele2_ptvarcone20'] / data['ele2_e'] < 0.15 )
    data['ele1_atlasIso_topoetcone20'] = ( data['ele1_topoetcone20'] / data['ele1_e'] < 0.20 )
    data['ele2_atlasIso_topoetcone20'] = ( data['ele2_topoetcone20'] / data['ele2_e'] < 0.20 )

    # Add ML isolation
    log.info(f"Add ML_eIso, ML_ePid, ML_pIso and ML_pPid score to data")
    log.info(f"        Electron models: {eIsoModel}")
    log.info(f"                         {ePidModel}")
    

    # Use a model for each of the given models
    eIsoGBM = lgb.Booster(model_file = eIsoModel)
    ePidGBM = lgb.Booster(model_file = ePidModel)
    
    # Add their score to the data 
    data["ele1_eIso_score"] = eIsoScore(eIsoGBM, data, 1, n_jobs=args.nJobs)
    data["ele1_ePid_score"] = ePidScore(ePidGBM, data, 1, n_jobs=args.nJobs)

    data["ele2_eIso_score"] = eIsoScore(eIsoGBM, data, 2, n_jobs = args.nJobs)
    data["ele2_ePid_score"] = ePidScore(ePidGBM, data, 2, n_jobs = args.nJobs)
    if args.PartType == "eeg":
        log.info(f"        Photon models:   {pIsoModel}")
        log.info(f"                         {pPidModel}")
        log.info(f"        ee models:       {eemodel}")
        log.info(f"                         {eemodel}")
        pIsoGBM = lgb.Booster(model_file = pIsoModel)
        pPidGBM = lgb.Booster(model_file = pPidModel)
        eeGBM = lgb.Booster(model_file = eemodel)
        data["pho_pIso_score"] = pIsoScore(pIsoGBM, data,0, n_jobs = args.nJobs)
        data["pho_pPid_score"] = pPidScore(pPidGBM, data,0, n_jobs = args.nJobs)
        data["ee_score"] = ZeeScore(eeGBM,data,n_jobs = args.nJobs)
        
elif (args.PartType == "mmg") or (args.PartType == "mm"):
    # Add ML isolation
    log.info(f"Add ML_mIso, ML_mPid, ML_pIso and ML_pPid score to data")
    log.info(f"        Muon models:     {mIsoModel}")
    log.info(f"                         {mPidModel}")
    
    # Use a model for each of the given models
    mIsoGBM = lgb.Booster(model_file = mIsoModel)
    mPidGBM = lgb.Booster(model_file = mPidModel)
    

    # Add their score to the data 
    data["muo1_mIso_score"] = mIsoScore(mIsoGBM, data, 1, n_jobs=args.nJobs)
    data["muo1_mPid_score"] = mPidScore(mPidGBM, data, 1, n_jobs=args.nJobs)

    data["muo2_mIso_score"] = mIsoScore(mIsoGBM, data, 2, n_jobs = args.nJobs)
    data["muo2_mPid_score"] = mPidScore(mPidGBM, data, 2, n_jobs = args.nJobs)
    if args.PartType == "mmg":
        log.info(f"        Photon models:   {pIsoModel}")
        log.info(f"                         {pPidModel}")
        log.info(f"        mm models:       {mmmodel}")
        log.info(f"                         {mmmodel}")
        pIsoGBM = lgb.Booster(model_file = pIsoModel)
        pPidGBM = lgb.Booster(model_file = pPidModel)
        mmGBM = lgb.Booster(model_file = mmmodel)
        data["pho_pIso_score"] = pIsoScore(pIsoGBM, data,0, n_jobs = args.nJobs)
        data["pho_pPid_score"] = pPidScore(pPidGBM, data,0, n_jobs = args.nJobs)
        data["mm_score"] = ZmmScore(mmGBM, data,n_jobs=args.nJobs)
elif args.PartType == "gg":
    #Add ML model
    log.info(f"Add ML isolation and Pid model")
    log.info(f"        Photon models:   {pIsoModel}")
    log.info(f"                         {pPidModel}")
    pIsoGBM = lgb.Booster(model_file = pIsoModel)
    pPidGBM = lgb.Booster(model_file = pPidModel)
    data["pho1_pIso_score"] = pIsoScore(pIsoGBM, data,1, n_jobs = args.nJobs)
    data["pho1_pPid_score"] = pPidScore(pPidGBM, data,1, n_jobs = args.nJobs)

    data["pho2_pIso_score"] = pIsoScore(pIsoGBM, data,2, n_jobs = args.nJobs)
    data["pho2_pPid_score"] = pPidScore(pPidGBM, data,2, n_jobs = args.nJobs)

#============================================================================
# Split in train, valid and test set
#============================================================================
log.info(f"Split data in training and test with split: {args.testSize}")
data_train, data_test = train_test_split(data, test_size = args.testSize, random_state=0)

TrainNSig = np.shape(data_train[data_train["type"] == 1])[0]
TrainNBkg = np.shape(data_train[data_train["type"] == 0])[0]
TestNSig = np.shape(data_test[data_test["type"] == 1])[0]
TestNBkg = np.shape(data_test[data_test["type"] == 0])[0]

log.info(f"        Shape of training data:  {np.shape(data_train)}")
log.info(f"                Signal:          {TrainNSig} ({( (TrainNSig) / (TrainNSig+TrainNBkg) )*100:.2f}%)")
log.info(f"                Background:      {TrainNBkg} ({( (TrainNBkg) / (TrainNSig+TrainNBkg) )*100:.2f}%)")
log.info(f"        Shape of test data:      {np.shape(data_test)}")
log.info(f"                Signal:          {TestNSig} ({( (TestNSig) / (TestNSig+TestNBkg) )*100:.2f}%)")
log.info(f"                Background:      {TestNBkg} ({( (TestNBkg) / (TestNSig+TestNBkg) )*100:.2f}%)")

# Copy data to avoid SettingWithCopyWarning
data_train = data_train.copy()
data_test = data_test.copy()

# Set dataset type: 0= train, 1 = valid, 2 = test
datatype = {0 : "train",
            1 : "valid",
            2 : "test"}

# Set dataset for test data
data_test["dataset"] = 2

# Split training data into train and valid
log.info(f"Split training data in training and validation with split: {args.validSize}")
data_train["dataset"] = 0
data_train.loc[data_train.sample(frac = args.validSize, random_state = 0).index,"dataset"] = 1

# Create masks
trainMask = (data_train["dataset"] == 0)
validMask = (data_train["dataset"] == 1)

trainNSig = np.shape(data_train[trainMask & (data_train["type"] == 1)])[0]
trainNBkg = np.shape(data_train[trainMask & (data_train["type"] == 0)])[0]
validNSig = np.shape(data_train[validMask & (data_train["type"] == 1)])[0]
validNBkg = np.shape(data_train[validMask & (data_train["type"] == 0)])[0]

# Print
log.info(f"        Shape of training set:   {np.shape(data_train[trainMask])}")
log.info(f"                Signal:          {trainNSig} ({( (trainNSig) / (trainNSig+trainNBkg) )*100:.2f}%)")
log.info(f"                Background:      {trainNBkg} ({( (trainNBkg) / (trainNSig+trainNBkg) )*100:.2f}%)")
log.info(f"        Shape of validation set: {np.shape(data_train[validMask])}")
log.info(f"                Signal:          {validNSig} ({( (validNSig) / (validNSig+validNBkg) )*100:.2f}%)")
log.info(f"                Background:      {validNBkg} ({( (validNBkg) / (validNSig+validNBkg) )*100:.2f}%)")

#============================================================================
# Reweigh - GBReweighter
#============================================================================
log.info(f"Reweigh background data using GBReweighter on training set")

reweightNames = ["nEst10", "nEst40", "nEst100", "nEst200"]

# Set parameters: Default {'n_estimators' : 40, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0}
reweightParams = [ {'n_estimators' : 10, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 40, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 100, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 200, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 }
                 ]

log.info(f"Regular reweights")
for iWeight, weightName in enumerate(reweightNames):
    t = time()
    # Print parameters
    log.info(f"Parameters for GBReweighter:")
    params = reweightParams[iWeight]
    for param in params:
        log.info(f"         {param} : {params[param]}")

      # Setup reweighter: https://arogozhnikov.github.io/hep_ml/reweight.html#
    reweighter  = GBReweighter(n_estimators=params['n_estimators'],
                               learning_rate=params['learning_rate'],
                               max_depth=params['max_depth'],
                               min_samples_leaf=params['min_samples_leaf'],
                               loss_regularization=params['loss_regularization'])
   
   # Create weight estimators and fit them to the data
    log.info(f"Fitting weights...")
    if args.PartType == "eeg":
        reweighter.fit(original = np.array([data_train["eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["et"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["et"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    elif args.PartType == "mmg":
        reweighter.fit(original = np.array([data_train["eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["pt"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["pt"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    if (args.PartType == "ee") or (args.PartType =="gg"): #the same for two photons or two electrons
        reweighter.fit(original = np.array([data_train["eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["et"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["et"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    elif args.PartType == "mm":
        reweighter.fit(original = np.array([data_train["eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["pt"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["pt"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["invM"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    elif args.PartType == "ele":
        reweighter.fit(original = np.array([data_train["ele_eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["ele_et"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["ele_eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["ele_et"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    elif args.PartType == "muo":
        reweighter.fit(original = np.array([data_train["muo_eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["muo_pt"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["muo_eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["muo_pt"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)
    elif args.PartType == "pho":
        reweighter.fit(original = np.array([data_train["pho_eta"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["pho_et"][trainMask & (data_train["type"] < 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] < 0.5)]]).T,
                    target   = np.array([data_train["pho_eta"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["pho_et"][trainMask & (data_train["type"] > 0.5)],
                                        data_train["correctedScaledAverageMu"][trainMask & (data_train["type"] > 0.5)]]).T)


    log.info(f"Fitting of weights is done (time: {timedelta(seconds=time() - t)})")

    # Get weights 
    log.info(f"Get weights for training, validation and test set")
    weight_train, weightNorm_train = getRegularWeights("train", reweighter, data_train[trainMask])
    weight_valid, weightNorm_valid = getRegularWeights("valid", reweighter, data_train[validMask])
    weight_test, weightNorm_test = getRegularWeights("test", reweighter, data_test)

    # Add weights to data 
    log.info(f"Add weights for training, validation and test set to data")
    data_train["regWeight_"+weightName] = 0
    data_train.loc[trainMask, "regWeight_"+weightName] = weight_train
    data_train.loc[validMask, "regWeight_"+weightName] = weight_valid
    data_test["regWeight_"+weightName] = weight_test

    # Add normalized weights to data
    log.info(f"Add normalized weights for training, validation and test set to data")
    data_train["regWeight_"+weightName+"_Norm"] = 0
    data_train.loc[trainMask,"regWeight_"+weightName+"_Norm"] = weightNorm_train
    data_train.loc[validMask,"regWeight_"+weightName+"_Norm"] = weightNorm_valid
    data_test["regWeight_"+weightName+"_Norm"] = weightNorm_test

log.info(f"Reverse reweights")
for iWeight, weightName in enumerate(reweightNames):
    t = time()
    # Print parameters
    log.info(f"Parameters for GBReweighter:")
    params = reweightParams[iWeight]
    for param in params:
        log.info(f"        {param} : {params[param]}")

    # Setup reweighter: https://arogozhnikov.github.io/hep_ml/reweight.html#
    reweighter  = GBReweighter(n_estimators=params['n_estimators'],
                               learning_rate=params['learning_rate'],
                               max_depth=params['max_depth'],
                               min_samples_leaf=params['min_samples_leaf'],
                               loss_regularization=params['loss_regularization'])

    # Create weight estimators and fit them to the data
    log.info(f"Fitting weights...")
    if args.PartType == "eeg":
        reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['et'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['et'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)
    elif args.PartType == "mmg":
        reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['pt'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['pt'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)
    if (args.PartType == "ee") or (args.PartType == "gg"): #Same for two photons and two electrons
        reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['et'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['et'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)
    elif args.PartType == "mm":
        reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['pt'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['pt'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['invM'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)                                            
    elif args.PartType == "ele":
        reweighter.fit(original = np.array([data_train['ele_eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['ele_et'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['ele_eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['ele_et'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)
    elif args.PartType == "muo":
        reweighter.fit(original = np.array([data_train['muo_eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['muo_pt'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['muo_eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['muo_pt'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)
    elif args.PartType == "pho":
        reweighter.fit(original = np.array([data_train['pho_eta'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['pho_et'][trainMask & (data_train["type"] > 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] > 0.5)]]).T,
                    target   = np.array([data_train['pho_eta'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['pho_et'][trainMask & (data_train["type"] <= 0.5)],
                                            data_train['correctedScaledAverageMu'][trainMask & (data_train["type"] <= 0.5)]]).T)                                                                                                                                    
    log.info(f"Fitting of weights is done (time: {timedelta(seconds=time() - t)})")

    # Get weights
    log.info(f"Get weights for training, validation and test set")
    weight_train, weightNorm_train = getReverseWeights("train", reweighter, data_train[trainMask])
    weight_valid, weightNorm_valid = getReverseWeights("valid", reweighter, data_train[validMask])
    weight_test, weightNorm_test  = getReverseWeights("test",  reweighter, data_test)

    # Add weights to data
    log.info(f"Add weights for training, validation and test set to data")
    data_train["revWeight_"+weightName] = 0
    data_train.loc[trainMask,"revWeight_"+weightName] = weight_train
    data_train.loc[validMask,"revWeight_"+weightName] = weight_valid
    data_test["revWeight_"+weightName] = weight_test

    # Add normalized weights to data
    log.info(f"Add normalized weights for training, validation and test set to data")
    data_train["revWeight_"+weightName+"_Norm"] = 0
    data_train.loc[trainMask,"revWeight_"+weightName+"_Norm"] = weightNorm_train
    data_train.loc[validMask,"revWeight_"+weightName+"_Norm"] = weightNorm_valid
    data_test["revWeight_"+weightName+"_Norm"] = weightNorm_test

#============================================================================
# Save to hdf5
#============================================================================
column_names = data_train.columns
log.info("Column names:\n{}".format(column_names))

filename_train = args.outdir+fname+"_train.h5"
filename_test = args.outdir+fname+"_test.h5"

log.info("Saving training data to {}".format(filename_train))
with h5py.File(filename_train, "w") as hf:
    for var in column_names:
        if var == "process":
            continue
        else:
            hf.create_dataset( f'{var}', data=np.array(data_train[var]), chunks = True, maxshape= (None,), compression='lzf')

log.info("Saving test data to {}".format(filename_test))
with h5py.File(filename_test, "w") as hf:
    for var in column_names:
        if var == "process":
            continue
        else:
            hf.create_dataset( f'{var}', data=np.array(data_test[var]), chunks = True, maxshape= (None,), compression='lzf')

#============================================================================
# Plotting and printing weights for training set
#============================================================================
masks = [trainMask, validMask]
maskNames = ["train", "valid"]
maskLabel = ["Training set", "Validation set"]

weightTypes = ["regWeight", "revWeight"]
weightTypeNames = ["Regular", "Reverse"]

weightLinestyle = ["dotted", "dashed", "dashdot", "solid"]

for iType, weightType in enumerate(weightTypes):
    for iWeight, weightName in enumerate(reweightNames):
        # Plot histogram of weights
        log.info(f"Plotting histogram of weights for {weightTypeNames[iType]} weights with {weightType} {weightName}")
        fig, ax = plt.subplots(1,2,figsize=(8,3), sharey = True, gridspec_kw = {'wspace':0, 'hspace':0})

        for iMask, mask in enumerate(masks):
            log.info(f"Plotting data for {maskNames[iMask]}")
            weightxRange = [(10**(-2),10**2),(10**(-3),10**3),(10**(-20),10**3),(10**(-30),10**3)]
            # Get dataset of weights
            weightsAll = data_train[["type",weightType+"_"+weightName]][mask].copy()

            # Sort weighted values - Highest weight get index 0, secondhighest index 1 etc
            log.info(f"         Sorting weights")
            weightSorted = weightsAll.sort_values(by=[weightType+"_"+weightName], ascending = False)
            weightIndex = np.arange(len(weightSorted))

            # Signal and background separation
            weightSortedSig = weightSorted[weightSorted["type"]==1]
            weightSortedBkg = weightSorted[weightSorted["type"]==0]

            # Number of weights above 100 and below 0.01
            nBkgOver100 = np.sum( weightSortedBkg[weightType+"_"+weightName] > 100 )
            nBkgUnder01 = np.sum( weightSortedBkg[weightType+"_"+weightName] < 0.01 )
            nBkgTotal = len(weightSortedBkg[weightType+"_"+weightName])
            nSigTotal = len(weightSortedSig[weightType+"_"+weightName])

            # Print weights
            log.info(f"        Ten particles with highest weights: \n{weightSortedBkg[:10]}")
            log.info(f"        Ten particles with lowest weights: \n{weightSortedBkg[nBkgTotal-10:]}")

            # Set logarithmic bins for histogram
            xRange = weightxRange[iWeight]
            nBins = 100
            logbins = np.logspace(np.log10(xRange[0]),np.log10(xRange[1]),nBins)

            # Plot weights
            SigData = weightSortedSig[weightType+"_"+weightName]
            SigLabel = f"Sig weights"
            BkgData = weightSortedBkg[weightType+"_"+weightName]
            BkgLabel = f"Bkg weights"
            ax[iMask].hist(SigData, label=SigLabel, color='C0', bins=logbins, histtype='step', alpha=0.9, linestyle='solid')
            ax[iMask].hist(BkgData, label=BkgLabel, color='C3', bins=logbins, histtype='step', alpha=0.9, linestyle='solid')

            # Set plot design
            ax[iMask].set_title(f"particle weights ({weightName})\n{maskLabel[iMask]}")
            ax[iMask].set_xlabel(f"GBM reweighter weight")
            ax[iMask].set_xscale("log")
            ax[iMask].set_yscale("log")
        
        # Set remaining design
        ax[1].legend(loc="upper left", bbox_to_anchor=(1.04,1), borderaxespad=0)
        ax[0].set_ylabel(f"Frequency")
        fig.tight_layout(rect=[0,0,1,1])

        # Save and close figure
        fig.savefig(args.outdir+fname+f'_weightsHistogram_{weightType+"_"+weightName}.png')
        plt.close(fig)

#============================================================================
# Plotting histogram of weighted features
#============================================================================

log.info(f"Plotting data for {maskNames[iMask]}")

# Columns to be plotted 
if (args.PartType == "eeg") or (args.PartType == "ee") or (args.PartType == "gg"):
    colTypes = ['correctedScaledAverageMu', "eta", "et", 'invM' ]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$", "$M_{ee}$" ]
elif (args.PartType == "mmg") or (args.PartType == "mm"):
    colTypes = ['correctedScaledAverageMu', "eta", "pt", 'invM' ]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$P_T$", "$M_{\mu\mu}$" ]
elif args.PartType == "ele":
    colTypes = ['correctedScaledAverageMu', 'ele_eta', 'ele_et']
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$"]
elif args.PartType == "pho":
    colTypes = ['correctedScaledAverageMu', 'pho_eta', 'pho_et']
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$"]
elif args.PartType == "muo":
    colTypes = ['correctedScaledAverageMu', "muo_eta", "muo_pt"]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$P_T$"]


# Set plotting parameters
if (args.PartType == "eeg") or (args.PartType == "mmg") or (args.PartType == "ee") or (args.PartType == "mm") or (args.PartType == "gg"):
    xRange = [(0,100), (-7,7), (0,1000), (args.invMmin, args.invMmax)]
    yRangeTrain = [(0,6.5*10**4), (10**0/2,10**4), (10**0/2,5*10**5), (0,8*10**4)]
    yRangeValid = [(0,2*10**4), (10**0/2,5*10**3), (10**0/2,2.5*10**5), (0,2*10**4)]
    binwidth = [5,0.05,10,args.binWidth]
    binUnits = ["",""," GeV", " GeV"]
    nBins = [int((xRange[0][1]-xRange[0][0])/binwidth[0]),int((xRange[1][1]-xRange[1][0])/binwidth[1]), int((xRange[2][1]-xRange[2][0])/binwidth[2]),  int((args.invMmax-args.invMmin)/binwidth[3])]
    logScale = [False, True, True, False]

    weightColor = ['C1', 'C2', 'C4', 'C5']

    placement = [(0,0),(1,0),(0,1),(1,1)]
else: 
    xRange = [(0,100), (-7,7), (0,200)]
    yRangeTrain = [(0,1*10**5), (10**0/2,5*10**4), (10**0/2,5*10**5)]
    yRangeValid = [(0,4*10**4), (10**0/2,7*10**3), (10**0/2,2.5*10**5)]
    binwidth = [5,0.05,10]
    binUnits = ["",""," GeV"]
    nBins = [int((xRange[0][1]-xRange[0][0])/binwidth[0]),int((xRange[1][1]-xRange[1][0])/binwidth[1]), int((xRange[2][1]-xRange[2][0])/binwidth[2])]
    logScale = [False, True, True]

    weightColor = ['C1', 'C2', 'C4']

    placement = [(0,0),(1,0),(0,1)]


# Plot weighted eta, et and <mu> for regular and reverse reweighing
for iType, weightType in enumerate(weightTypes):
    for iMask, mask in enumerate(masks):
        print(f"Plotting {weightTypeNames[iType]} weighted distributions for {maskNames[iMask]}")

        # Initiate figure
        fig, ax = plt.subplots(2,2,figsize=(8,5))

        for i, colType in enumerate(colTypes):
            sigData = data_train[colType][mask & (data_train["type"]==1)]
            bkgData = data_train[colType][mask & (data_train["type"]==0)]
            # Plot data in separate figure
            ax[placement[i]].hist(sigData, label="Signal",     color='C0', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
            ax[placement[i]].hist(bkgData, label="Background", color='C3', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)

            for iWeight, weightName in enumerate(reweightNames[:-1]):
                print(f"        Adding weighted data: {weightType+'_'+weightName}")
                if iType==0: # Regular weights
                    bkgWeights = data_train[weightType+'_'+weightName][mask & (data_train["type"]==0)]
                    bkgLabel = f"Weights {weightName}"
                    ax[placement[i]].hist(bkgData, weights=bkgWeights, label=bkgLabel, color=weightColor[iWeight], linestyle=weightLinestyle[iWeight], range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
                elif iType==1: # Reverse weights
                    sigWeights = data_train[weightType+'_'+weightName][mask & (data_train["type"]==1)]
                    sigLabel = f"Weights {weightName}"
                    ax[placement[i]].hist(sigData, weights=sigWeights, label=sigLabel, color=weightColor[iWeight], linestyle=weightLinestyle[iWeight], range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)

            # Set plot design for separate figure
            ax[placement[i]].set_xlabel(f"{colNames[i]}")
            if (logScale[i]): ax[placement[i]].set_yscale("log")
            if iMask==0:
                ax[placement[i]].set_ylim(yRangeTrain[i])
            elif iMask==1:
                ax[placement[i]].set_ylim(yRangeValid[i])
            ax[placement[i]].set_ylabel(f"Frequency / {binwidth[i]}{binUnits[i]}")

        # Set remaining design for separate figure
        ax[0,1].legend(loc="upper right" ) #, bbox_to_anchor=(1.04, 1), borderaxespad=0) #, framealpha=1, fontsize=9, edgecolor='k')
        fig.tight_layout(rect=[0,0,1,0.98], h_pad=0.3, w_pad=0.3)
        plt.text(0.01, 0.99, f"Zee - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)

        # Save and close separate figure
        fig.savefig(args.outdir+fname+f'_weightedHistogram_{weightType}_{maskNames[iMask]}.png')
        plt.close(fig)
if (args.PartType == "eeg") or (args.PartType == "mmg"):
    #============================================================================
    # Plotting stacked histogram of processes
    #============================================================================
    processes = data_name
    log.info(f"Processes in data: {processes}")

    for iMask, mask in enumerate(masks):
        log.info(f"Plotting stacked histograms of processes for together for {maskNames[iMask]}")

        # Set plotting parameters
        xRange = (args.invMmin,args.invMmax)
        nBins = int((args.invMmax-args.invMmin)/args.binWidth)
        yRange = (0.5*10**0, 10**(5))

        weightNames = [None]+reweightNames

        # Setup figure
        fig, ax = plt.subplots(2,4, figsize=(8,4), sharey=True, sharex=True)

        # Loop over the subplots
        for subfigi in range(2):
            for subfigj in range(4):
                weightName = weightNames[subfigj]

                # Get data based on mask (train/valid), sig/bkg and process
                processesData = []
                processesWeights = []
                colors = []
                labels = []
                for iProcess, process in enumerate(processes):
                    processData = data_train['invM'][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                    processesData.append(processData)
                    if weightName!=None:
                        processWeights = data_train[weightTypes[subfigi]+'_'+weightName][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                        processesWeights.append(processWeights)
                    colors.append(f'C{iProcess}')
                    labels.append(process)

                # Plot
                if weightName==None:
                    histogram = ax[subfigi,subfigj].hist(processesData, range=xRange, bins=nBins, color=colors, label=labels, stacked=True)
                    # ax[subfigi,subfigj].set_title(f"Unweighted",fontsize=12)
                    ax[subfigi,subfigj].text(0.01, 1.01, f"Unweighted", horizontalalignment='left', verticalalignment='bottom', transform=ax[subfigi,subfigj].transAxes, fontsize=10)
                else:
                    histogram = ax[subfigi,subfigj].hist(processesData, weights=processesWeights, range=xRange, bins=nBins, color=colors, label=labels, stacked=True)
                    ax[subfigi,subfigj].text(0.01, 1.01, f"{weightTypeNames[subfigi]} weighted \n{weightName}", horizontalalignment='left', verticalalignment='bottom', transform=ax[subfigi,subfigj].transAxes, fontsize=10)


                ax[subfigi,subfigj].set_yscale("log")
                ax[subfigi,subfigj].set_ylim(yRange)
                ax[subfigi,subfigj].set_xlim(xRange)
                if args.PartType == "eeg":
                    ax[-1,subfigj].set_xlabel(r"$M_{ee\gamma}$")
                elif args.PartType == "mmg":
                    ax[-1,subfigj].set_xlabel(r"$M_{\mu\mu\gamma}$")
            ax[1,0].set_ylabel(f"Signal \nFrequency / {args.binWidth} GeV")
            ax[0,0].set_ylabel(f"Background \nFrequency / {args.binWidth} GeV")

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",ncol=4)

        # Save
        fig.tight_layout(rect=[0,0.1,1,0.98], h_pad=0.3, w_pad=0.3)
        plt.text(0.01, 0.99, f"Zeeg {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        fig.savefig(args.outdir+fname+f'_SigBkgByProcess_combined_{maskNames[iMask]}.png')
        plt.close(fig)

    #============================================================================
    # Plotting Zllg peak
    #============================================================================
    log.info(f"Plotting Zllg peak")
    for iMask, mask in enumerate(masks):
        # Initiate figure
        fig, ax = plt.subplots(figsize=(3,3))
        bins = int( ((args.invMmax-args.invMmin)/args.binWidth) *2 )

        # Run over training and validation set
        ZeeData    = data_train['invM'][mask & (data_train["type"]==1) & (data_train["process"]=='Zee')]

        # Plot data in separate figure
        ax.hist(ZeeData,    label="Signal - Zee",    color='C0', range=(args.invMmin,args.invMmax), bins=bins, histtype='step', alpha=0.9)

        # Set plot design for separate figure
        ax.set_xlabel(f"invM")
        if iMask==0:
            ax.set_ylim(0,25000)
        elif iMask==1:
            ax.set_ylim(0,6000)
        plt.locator_params(axis='y', nbins=5)

        # Set remaining design for separate figure
        ax.legend(loc="upper right")#, bbox_to_anchor=(1.04, 1), borderaxespad=0) #, framealpha=1, fontsize=9, edgecolor='k')
        ax.set_ylabel(f"Frequency / {np.around(bins,2)} GeV")
        ax.text(0.01, 1.01, f"Zee {maskLabel[iMask]}", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        fig.tight_layout()

        # Save and close separate figure
        fig.savefig(args.outdir+fname+f'_Histogram_ZeePeak_{maskNames[iMask]}.png')
        plt.close(fig)
else:
    processes = data_name
    log.info(f"Processes in data: {processes}")

    for iMask, mask in enumerate(masks):
        log.info("Plotting stacked histograms of processes for together for {maskNames[iMask]}")

        # Set plotting parameters
        xRange = (10,80)
        nBins = int((80-10)/args.binWidth)
        yRange = (0.5*10**0, 10**(5))

        weightNames = [None]+reweightNames

        # Setup figure
        fig, ax = plt.subplots(2,4, figsize= (8,4), sharey = True, sharex = True)

        #Loop over the subplots
        for subfigi in range(2):
            for subfigj in range(4):
                weightNames = weightNames[subfigj]

                # Get data based on mask (train/valid), sig/bkg and process
                processesData = []
                processesWeights = []
                colors = []
                labels = []
                for iProcess, process in enumerate(processes):
                    if args.PartType == "ele":
                        processData = data_train['ele_et'][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                    elif args.PartType == "muo":
                        processData = data_train['muo_pt'][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                    elif args.PartType == "pho":
                        processData = data_train['pho_et'][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                    processesData.append(processData)
                    if weightName!=None:
                        processWeights = data_train[weightTypes[subfigi]+'_'+weightName][mask & (data_train["type"]==subfigi) & (data_train["process"]==process)]
                        processesWeights.append(processWeights)
                    colors.append(f'C{iProcess}')
                    labels.append(process)

                # Plot
                if weightName==None:
                    histogram = ax[subfigi,subfigj].hist(processesData, range=xRange, bins=nBins, color=colors, label=labels, stacked=True)
                    # ax[subfigi,subfigj].set_title(f"Unweighted",fontsize=12)
                    ax[subfigi,subfigj].text(0.01, 1.01, f"Unweighted", horizontalalignment='left', verticalalignment='bottom', transform=ax[subfigi,subfigj].transAxes, fontsize=10)
                else:
                    histogram = ax[subfigi,subfigj].hist(processesData, weights=processesWeights, range=xRange, bins=nBins, color=colors, label=labels, stacked=True)
                    ax[subfigi,subfigj].text(0.01, 1.01, f"{weightTypeNames[subfigi]} weighted \n{weightName}", horizontalalignment='left', verticalalignment='bottom', transform=ax[subfigi,subfigj].transAxes, fontsize=10)


                ax[subfigi,subfigj].set_yscale("log")
                ax[subfigi,subfigj].set_ylim(yRange)
                ax[subfigi,subfigj].set_xlim(xRange)

                ax[-1,subfigj].set_xlabel("$M_{e}$")
            ax[1,0].set_ylabel(f"Signal \nFrequency / {args.binWidth} GeV")
            ax[0,0].set_ylabel(f"Background \nFrequency / {args.binWidth} GeV")

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",ncol=4)

        # Save
        fig.tight_layout(rect=[0,0.1,1,0.98], h_pad=0.3, w_pad=0.3)
        plt.text(0.01, 0.99, f"Zee {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        fig.savefig(args.outdir+fname+f'_SigBkgByProcess_combined_{maskNames[iMask]}.png')
        plt.close(fig)
#============================================================================
# End
#============================================================================
log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")

