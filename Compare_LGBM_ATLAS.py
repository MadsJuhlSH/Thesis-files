"""
Comparison between the selection using LGBM and the 
selection using ATLAS cuts
"""
print("Program running...")
# Import packages and functions
import sys
import os
from os import path
import argparse
import logging as log
from time import time
from datetime import datetime, timedelta
#from Pho_search.Hgg_LGBMATLAS_sel import HggGBM
import lightgbm as lgb
import ROOT
from ROOT import RooChebychev, RooFit , RooCBShape, RooRealVar, gPad, RooArgList, RooFFTConvPdf, RooGaussian
from scipy.stats import chi2

import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math as m
import seaborn as sns
from returnLGBMScore import HggScore, eIsoScore, ePidScore, mIsoScore, mPidScore, pIsoScore, pPidScore, ZeeScore, ZmmScore, ZeegScore, ZmmgScore
from utils import timediftostr, Build_Folder_Structure, header, footer, print_col_names, h5ToDf, accuracy, HyperOpt_RandSearch, auc_eval, HeatMap_rand

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

print("Packages and functions loaded")
# Start "timer"
t_start = time()
#%%############################################################################
#   Parser
###############################################################################
parser = argparse.ArgumentParser(description='Electron isolation with the LightGBM framework.')
#parser.add_argument('path', type=str,
#                    help='Data file to train on.')
parser.add_argument('--outdir', action='store', default="output/LGBM_vs_ATLAS/", type=str,
                    help='Output directory.')
parser.add_argument('--tag', action='store', type=str, required=False,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', type=int, required=False, default=10,
                    help='Amount of njobs (default = 10)')
parser.add_argument('--ptcut', action= 'store', type=int, required=False, default = 0,choices=[0,1],
                    help = "If 1, there will be a cut in ptvarcone20")
#parser.add_argument('--ATLAScut', action= 'store', type=int, required=False, default = 1,choices=[0,1],
#                    help = "If 1, There have only been used photon cuts, so implementing electron cuts. Assumed electron cuts implemented")
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["ele", "muo", "Hgg", "Data"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')
parser.add_argument('--Data', action = 'store', type=int, default=0, choices=[0,1],
                    help="If 1 it is data, changes plot name." )

args = parser.parse_args()

#The different models used

eIsoModel = "output/Training/models/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso240921v2_revWeight_nEst40_2021-09-24/eleIsomodel_LGBM_LeptonLepton_eleRev40Iso240921v2_revWeight_nEst40_2021-09-24.txt"
ePidModel = "output/Training/models/elePidmodel_LGBM_LeptonLepton_eleReg40Pid240921_regWeight_nEst40_2021-09-24/elePidmodel_LGBM_LeptonLepton_eleReg40Pid240921_regWeight_nEst40_2021-09-24.txt"
mIsoModel = "output/Training/models/muoIsomodel_LGBM_Lepton_muoRev10Iso021021_revWeight_nEst10_2021-10-02/muoIsomodel_LGBM_Lepton_muoRev10Iso021021_revWeight_nEst10_2021-10-02.txt"
mPidModel = "output/Training/models/muoPidmodel_LGBM_Lepton_muoReg40Pid180921_regWeight_nEst40_2021-09-18/muoPidmodel_LGBM_Lepton_muoReg40Pid180921_regWeight_nEst40_2021-09-18.txt"
pIsoModel = "output/Training/models/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23.txt"
pPidModel = "output/Training/models/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29.txt"

eeModel = "output/Training/models/eemodel_LGBM_Zee130921Zeeg240921_eeReg40280921_regWeight_nEst40_2021-09-28/eemodel_LGBM_Zee130921Zeeg240921_eeReg40280921_regWeight_nEst40_2021-09-28.txt"
mmModel = "output/Training/models/mmmodel_LGBM_Zmm130921Zmmg240921_mmReg40021021_regWeight_nEst40_2021-10-03/mmmodel_LGBM_Zmm130921Zmmg240921_mmReg40021021_regWeight_nEst40_2021-10-03.txt"

eegModel = "output/Training/models/eegmodel_LGBM_Zeeg200921Zee280921_eegReg40051021_regWeight_nEst40_2021-10-05/eegmodel_LGBM_Zeeg200921Zee280921_eegReg40051021_regWeight_nEst40_2021-10-05.txt"
mmgModel = "output/Training/models/mmgmodel_LGBM_Zmmg200921_mmgReg40051021_regWeight_nEst40_2021-10-05/mmgmodel_LGBM_Zmmg200921_mmgReg40051021_regWeight_nEst40_2021-10-05.txt"

HggModel = "output/Training/models/ggmodel_LGBM_hgg160921_ggReg10300921_regWeight_nEst10_2021-09-30/ggmodel_LGBM_hgg160921_ggReg10300921_regWeight_nEst10_2021-09-30.txt"

#Loading the different models
# The models depends on which interaction is being done. The PartType variable.

if (args.PartType == "ele"):
    # Add ML isolation
    log.info(f"Loading ML_eIso, ML_ePid, ML_pIso and ML_pPid.")
    log.info(f"        Electron models: {eIsoModel}")
    log.info(f"                         {ePidModel}")
    log.info(f"        Photon models:   {pIsoModel}")
    log.info(f"                         {pPidModel}")
    log.info(f"             ee model:   {eeModel}")
    log.info(f"             eeg model:  {eegModel}")

    # Use a model for each of the given models
    lIsoGBM = lgb.Booster(model_file = eIsoModel)
    lPidGBM = lgb.Booster(model_file = ePidModel)
    pIsoGBM = lgb.Booster(model_file = pIsoModel)
    pPidGBM = lgb.Booster(model_file = pPidModel)
    ZllGBM = lgb.Booster(model_file = eeModel)
    ZllgGBM = lgb.Booster(model_file = eegModel)
    
elif (args.PartType == "muo") or (args.PartType=="Data"):
    # Add ML isolation
    log.info(f"Loading ML_mIso, ML_mPid, ML_pIso and ML_pPid.")
    log.info(f"        Muon models:     {mIsoModel}")
    log.info(f"                         {mPidModel}")
    log.info(f"        Photon models:   {pIsoModel}")
    log.info(f"                         {pPidModel}")
    log.info(f"        mm model:        {mmModel}")
    log.info(f"        mmg model:       {mmgModel}")

    # Use a model for each of the given models
    lIsoGBM = lgb.Booster(model_file = mIsoModel)
    lPidGBM = lgb.Booster(model_file = mPidModel)
    pIsoGBM = lgb.Booster(model_file = pIsoModel)
    pPidGBM = lgb.Booster(model_file = pPidModel)
    ZllGBM = lgb.Booster(model_file = mmModel)
    ZllgGBM = lgb.Booster(model_file = mmgModel)

elif args.PartType == "Hgg":
    # Add ML models
    log.info(f"Loading ML_pIso and ML_pPid and ML_pp.")
    log.info(f"        Photon models:   {pIsoModel}")
    log.info(f"                         {pPidModel}")
    log.info(f"        pp model:        {HggModel}")

    # Use a model for each of the given models
    pIsoGBM = lgb.Booster(model_file = pIsoModel)
    pPidGBM = lgb.Booster(model_file = pPidModel)
    HggGBM = lgb.Booster(model_file = HggModel)


# The different Data files the models are used on
if args.PartType =="ele":
    TruthFile ="output/SignalFiles/Llgam/Zeeg291021_phoOrigin3/Zeeg291021_PhoOrigin3.h5"
elif args.PartType == "Data":
    TruthFile = "output/SignalFiles/data_Llgam/datammg011121/datammg011121.h5"
elif args.PartType =="muo":
    TruthFile ="output/SignalFiles/Llgam/Zmmg291021_phoOrigin3/Zmmg291021_PhoOrigin3.h5"
elif args.PartType == "Hgg":
    TruthFile = "output/SignalFiles/Hgamgam/hgg291021/hgg291021.h5"
    
# Second check
#%%############################################################################
#   Filenames and directories
###############################################################################
# Create timestamp for model name
now = datetime.now()
timestamp = datetime.timestamp(now)
timelabel = f"{datetime.date(now)}"
output_dir = args.outdir + args.tag + "_" + timelabel + "/"
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
print(f'Output directory:         {output_dir}')
print(f'Number of threads:        {args.njobs}')
print(f'------------------------------------------------------------------')
#%%############################################################################
#   Importing data.
###############################################################################
header("Importing and separating data")
t = time()
if (args.PartType == "ele") or (args.PartType =="muo"):
    mininvm, maxinvm = 60,140
    minpeakinvm, maxpeakinvm = 87,95
elif args.PartType == "Data":
    mininvm, maxinvm = 60,160
    minpeakinvm, maxpeakinvm = 87,95
else: 
    mininvm, maxinvm = 110,160
    minpeakinvm, maxpeakinvm = 121,129
# Get hdf5 datafile as dataframe
#Loading ATLAS file
print(f"Loading Truth file and making ATLAS cuts")

dataatlas = h5ToDf(TruthFile)
#######################################################
# Do the ATLAS selections
#######################################################
#These are the cuts used by ATLAS
if args.PartType == "ele":
    maskATLAS = (   (dataatlas["ele1_trigger"]==1)&
                    (dataatlas["ele1_LHLoose"]==1)&
                    (dataatlas["ele2_LHLoose"]==1)&
                    (dataatlas["ele1_et"]>10)&
                    (dataatlas["ele2_et"]>10)&
                    (dataatlas["invMll"]<82)&
                    (dataatlas["ele1_charge"]*dataatlas["ele2_charge"]<0)&
                    ((np.abs(dataatlas["ele1_eta"])<1.37) | ((np.abs(dataatlas["ele1_eta"])>1.52) & (np.abs(dataatlas["ele1_eta"])<2.47))) &
                    ((np.abs(dataatlas["ele2_eta"])<1.37) | ((np.abs(dataatlas["ele2_eta"])>1.52) & (np.abs(dataatlas["ele2_eta"])<2.47))) &
                    (np.abs(dataatlas["ele1_d0"]/dataatlas["ele1_d0Sig"])<5)&
                    (np.abs(dataatlas["ele2_d0"]/dataatlas["ele2_d0Sig"])<5)&
                    (np.abs(dataatlas["ele1_delta_z0_sin_theta"])<0.5)&
                    (np.abs(dataatlas["ele2_delta_z0_sin_theta"])<0.5)&
                    ((np.abs(dataatlas["pho_eta"])<1.37) | ((np.abs(dataatlas["pho_eta"])>1.52) & (np.abs(dataatlas["pho_eta"])<2.37)))&
                    (dataatlas["pho_isPhotonEMTight"]==1)
    )
elif (args.PartType == "muo"):
    maskATLAS = (   (dataatlas["muo1_trigger"]==1)&
                    (dataatlas["muo1_LHMedium"]==1)&
                    (dataatlas["muo2_LHMedium"]==1)&
                    (dataatlas["muo1_pt"]>10000)&
                    (dataatlas["muo2_pt"]>10000)&
                    (dataatlas["invMll"]<82)&
                    (dataatlas["muo1_charge"]*dataatlas["muo2_charge"]<0)&
                    ((np.abs(dataatlas["muo1_eta"])<1.37) | ((np.abs(dataatlas["muo1_eta"])>1.52) & (np.abs(dataatlas["muo1_eta"])<2.7))) &
                    ((np.abs(dataatlas["muo2_eta"])<1.37) | ((np.abs(dataatlas["muo2_eta"])>1.52) & (np.abs(dataatlas["muo2_eta"])<2.7))) &
                    (np.abs(dataatlas["muo1_priTrack_d0"]/dataatlas["muo1_priTrack_d0Sig"])<3)&
                    (np.abs(dataatlas["muo2_priTrack_d0"]/dataatlas["muo2_priTrack_d0Sig"])<3)&
                    (np.abs(dataatlas["muo1_delta_z0_sin_theta"])<0.5)&
                    (np.abs(dataatlas["muo2_delta_z0_sin_theta"])<0.5)&
                    ((np.abs(dataatlas["pho_eta"])<1.37) | ((np.abs(dataatlas["pho_eta"])>1.52) & (np.abs(dataatlas["pho_eta"])<2.37)))&
                    (dataatlas["pho_isPhotonEMTight"]==1)
    )
elif args.PartType == "Data":
    maskATLAS = (   (dataatlas["muo1_trigger"]==1)&
                    (dataatlas["muo1_LHMedium"]==1)&
                    (dataatlas["muo2_LHMedium"]==1)&
                    (dataatlas["muo1_pt"]>10000)&
                    (dataatlas["muo2_pt"]>10000)&
                    (dataatlas["invMll"]<82)&
                    (dataatlas["muo1_charge"]*dataatlas["muo2_charge"]<0)&
                    ((np.abs(dataatlas["muo1_eta"])<1.37) | ((np.abs(dataatlas["muo1_eta"])>1.52) & (np.abs(dataatlas["muo1_eta"])<2.7))) &
                    ((np.abs(dataatlas["muo2_eta"])<1.37) | ((np.abs(dataatlas["muo2_eta"])>1.52) & (np.abs(dataatlas["muo2_eta"])<2.7))) &
                    (np.abs(dataatlas["muo1_priTrack_d0"]/dataatlas["muo1_priTrack_d0Sig"])<3)&
                    (np.abs(dataatlas["muo2_priTrack_d0"]/dataatlas["muo2_priTrack_d0Sig"])<3)&
                    (np.abs(dataatlas["muo1_delta_z0_sin_theta"])<0.5)&
                    (np.abs(dataatlas["muo2_delta_z0_sin_theta"])<0.5)&
                    #(dataatlas["pho_et"]/dataatlas["invM"])&
                    ((np.abs(dataatlas["pho_eta"])<1.37) | ((np.abs(dataatlas["pho_eta"])>1.52) & (np.abs(dataatlas["pho_eta"])<2.37)))&
                    (dataatlas["pho_isPhotonEMTight"]==1)
    )
elif args.PartType == "Hgg":
    
    
    
    maskATLAS = (   (dataatlas["pho1_isPhotonEMTight"]==1) &
                    (dataatlas["pho2_isPhotonEMTight"]==1) &
                    (dataatlas["pho1_et"]>35) &
                    (dataatlas["pho2_et"]>25) &
                    ((np.abs(dataatlas["pho1_eta"])<1.37) | ((np.abs(dataatlas["pho1_eta"])>1.52) & (np.abs(dataatlas["pho1_eta"])<2.37)))&
                    ((np.abs(dataatlas["pho2_eta"])<1.37) | ((np.abs(dataatlas["pho2_eta"])>1.52) & (np.abs(dataatlas["pho2_eta"])<2.37)))&
                    (np.abs(dataatlas["pho1_pt"]/dataatlas["invM"])>0.35) &
                    (np.abs(dataatlas["pho2_pt"]/dataatlas["invM"])>0.25) &
                    (np.abs(dataatlas["pho1_ptvarcone20"]/dataatlas["pho1_et"])<0.05) &
                    (np.abs(dataatlas["pho2_ptvarcone20"]/dataatlas["pho2_et"])<0.05) 
                    #The following two lines should be applied as well.
                    #But I don't have the etcone20 saved in my data.
                    #(np.abs(dataatlas["pho1_topoetcone20"]/dataatlas["pho1_et"])<0.065) &
                    #(np.abs(dataatlas["pho2_topoetcone20"]/dataatlas["pho2_et"])<0.065)
    )
    
    
    
dataATLAS = dataatlas[maskATLAS]
#Applying cuts to see how much signal and background can be found in the peak region for ATLAS
if args.PartType != "Data":
    nATLAS = len(dataATLAS[(dataATLAS["type"]==1)&(dataATLAS["invM"]<maxinvm) & (dataATLAS["invM"]>mininvm)])
    nATLASpeak = len(dataATLAS[(dataATLAS["type"]==1)&(dataATLAS["invM"]<maxpeakinvm) & (dataATLAS["invM"]>minpeakinvm)])

    nATLASbkg = len(dataATLAS[(dataATLAS["type"]==0)&(dataATLAS["invM"]<maxinvm) & (dataATLAS["invM"]>mininvm)])
    nATLASpeakbkg = len(dataATLAS[(dataATLAS["type"]==0)&(dataATLAS["invM"]<maxpeakinvm) & (dataATLAS["invM"]>minpeakinvm)])
#########################
# The the pt cut of ATLAS
#########################
#ATLAS
#Applying Efficiency cuts for ATLAS data.
if args.PartType !="Hgg":
    print("Removing top 1% ptvarcone, from ATLAS selections")
    # sorting all the ptvarcone20
    if args.PartType =="ele":
        ATLASptvarcone = np.concatenate((dataATLAS['ele1_ptvarcone20'],dataATLAS['ele2_ptvarcone20']))
    elif (args.PartType =="muo") or (args.PartType == "Data"):
        ATLASptvarcone = np.concatenate((dataATLAS['muo1_ptvarcone20'],dataATLAS['muo2_ptvarcone20']))
    ATLASptvarcone = np.sort(ATLASptvarcone)
    
    nremove = np.int(ATLASptvarcone.shape[0]*0.01)
    ATLASptvarconecut = (ATLASptvarcone[-(nremove+1)])
    if args.PartType == "ele":
        def mask_ZllgammaATLAS(cut):
            return (( dataATLAS['ele1_ptvarcone20'] < cut ) &
                    ( dataATLAS['ele2_ptvarcone20'] < cut ) 
                    )
    elif (args.PartType == "muo") or (args.PartType == "Data"):
        def mask_ZllgammaATLAS(cut):
            return (( dataATLAS['muo1_ptvarcone20'] < cut ) &
                    ( dataATLAS['muo2_ptvarcone20'] < cut ) 
                    )
    #Applying cuts and updating the number variable.
    dataATLAS = dataATLAS[mask_ZllgammaATLAS(ATLASptvarconecut)]
    if args.PartType != "Data":
        nATLAS = len(dataATLAS[(dataATLAS["type"]==1)&(dataATLAS["invM"]<maxinvm) & (dataATLAS["invM"]>mininvm)])
        nATLASpeak = len(dataATLAS[(dataATLAS["type"]==1)&(dataATLAS["invM"]<maxpeakinvm) & (dataATLAS["invM"]>minpeakinvm)])

        nATLASbkg = len(dataATLAS[(dataATLAS["type"]==0)&(dataATLAS["invM"]<maxinvm) & (dataATLAS["invM"]>mininvm)])
        nATLASpeakbkg = len(dataATLAS[(dataATLAS["type"]==0)&(dataATLAS["invM"]<maxpeakinvm) & (dataATLAS["invM"]>minpeakinvm)])
    else: 
        nATLAS = len(dataATLAS[(dataATLAS["invM"]<maxinvm) & (dataATLAS["invM"]>mininvm)])
        nATLASpeak = len(dataATLAS[(dataATLAS["invM"]<maxpeakinvm) & (dataATLAS["invM"]>minpeakinvm)])
        nATLASouter = len(dataATLAS[((dataATLAS["invM"]>60)& (dataATLAS["invM"]<80)) | ((dataATLAS["invM"]>100)& (dataATLAS["invM"]<140))])

"""
#Seeing the distribution for the real data - just testing things
if args.PartType == "Data":
    fig,ax = plt.subplots(figsize=(12,12))
    ax.hist(dataATLAS["invM"],bins=np.linspace(mininvm,maxinvm,maxinvm-mininvm+1),label = "Data distribution")
    ax.legend(loc= "best")
    ax.set_xlabel("invM")
    ax.set_ylabel("Frequency")
    ax.set_title("InvM distribution for ATLAS selection of data")
    fig.savefig(figure_dir+"Data_ATLASdistribution.png")
    #quit()
"""
#######################################################
# ATLAS selections on the hybrid models (my ll-pair model with ATLAS photon model and the opposite)
#######################################################
if args.PartType !="Hgg":

    datalep = h5ToDf(TruthFile)
    datapho = h5ToDf(TruthFile)
if args.PartType == "ele":
    # Using only the photon cuts
    maskATLASpho = (((np.abs(datalep["pho_eta"])<1.37) | ((np.abs(datalep["pho_eta"])>1.52) & (np.abs(datalep["pho_eta"])<2.37)))&
                    (datalep["pho_isPhotonEMTight"]==1)
    )
    # Using only the electron cuts
    maskATLASlep = (   (datapho["invMll"]<82)&
                    (datapho["ele1_trigger"]==1)&
                    (datapho["ele1_LHLoose"]==1)&
                    (datapho["ele2_LHLoose"]==1)&
                    (datapho["ele1_et"]>10)&
                    (datapho["ele2_et"]>10)&
                    (datapho["ele1_charge"]*datapho["ele2_charge"]<0)&
                    ((np.abs(datapho["ele1_eta"])<1.37) | ((np.abs(datapho["ele1_eta"])>1.52) & (np.abs(datapho["ele1_eta"])<2.47))) &
                    ((np.abs(datapho["ele2_eta"])<1.37) | ((np.abs(datapho["ele2_eta"])>1.52) & (np.abs(datapho["ele2_eta"])<2.47))) &
                    (np.abs(datapho["ele1_d0"]/datapho["ele1_d0Sig"])<5)&
                    (np.abs(datapho["ele2_d0"]/datapho["ele2_d0Sig"])<5)&
                    (np.abs(datapho["ele1_delta_z0_sin_theta"])<0.5)&
                    (np.abs(datapho["ele2_delta_z0_sin_theta"])<0.5)
                    )
elif (args.PartType == "muo") or (args.PartType == "Data"):
    # Using only the photon cuts
    maskATLASpho = (((np.abs(datalep["pho_eta"])<1.37) | ((np.abs(datalep["pho_eta"])>1.52) & (np.abs(datalep["pho_eta"])<2.37)))&
                    (datalep["pho_isPhotonEMTight"]==1)
    )
    # Using only the muon cuts
    maskATLASlep = (   (datapho["invMll"]<82)&
                    (datapho["muo1_trigger"]==1)&
                    (datapho["muo1_LHMedium"]==1)&
                    (datapho["muo2_LHMedium"]==1)&
                    (datapho["muo1_pt"]>10000)&
                    (datapho["muo2_pt"]>10000)&
                    (datapho["muo1_charge"]*datapho["muo2_charge"]<0)&
                    ((np.abs(datapho["muo1_eta"])<1.37) | ((np.abs(datapho["muo1_eta"])>1.52) & (np.abs(datapho["muo1_eta"])<2.7))) &
                    ((np.abs(datapho["muo2_eta"])<1.37) | ((np.abs(datapho["muo2_eta"])>1.52) & (np.abs(datapho["muo2_eta"])<2.7))) &
                    (np.abs(datapho["muo1_priTrack_d0"]/datapho["muo1_priTrack_d0Sig"])<3)&
                    (np.abs(datapho["muo2_priTrack_d0"]/datapho["muo2_priTrack_d0Sig"])<3)&
                    (np.abs(datapho["muo1_delta_z0_sin_theta"])<0.5)&
                    (np.abs(datapho["muo2_delta_z0_sin_theta"])<0.5)
    )

if args.PartType != "Hgg":
    # Applying cuts for models where some ATLAS cuts are used - not for H-> gamma gamma model.  
    dataLEP = datalep[maskATLASpho]
    dataPho = datapho[maskATLASlep]
#Loading the file used for the ML models
print(f"Loading the Truth file, so the different cuts can be determined")
dataTruth = h5ToDf(TruthFile)


# Creating some variables used in the models
if (args.PartType == "ele"):
    
    dataTruth['ele1_ptvarcone40_rel'] = (dataTruth['ele1_ptvarcone40_TightTTVALooseCone_pt1000'].div(dataTruth['ele1_et'])).values.tolist()
    dataTruth['ele1_topoetcone40_rel'] = (dataTruth['ele1_topoetcone40'].div(dataTruth['ele1_et'])).values.tolist()
    dataTruth['ele2_ptvarcone40_rel'] = (dataTruth['ele2_ptvarcone40_TightTTVALooseCone_pt1000'].div(dataTruth['ele2_et'])).values.tolist()
    dataTruth['ele2_topoetcone40_rel'] = (dataTruth['ele2_topoetcone40'].div(dataTruth['ele2_et'])).values.tolist()

    dataTruth['ele1_ptvarcone20_rel'] = (dataTruth['ele1_ptvarcone20'].div(dataTruth['ele1_et'])).values.tolist()
    dataTruth['ele1_topoetcone20_rel'] = (dataTruth['ele1_topoetcone20'].div(dataTruth['ele1_et'])).values.tolist()
    dataTruth['ele2_ptvarcone20_rel'] = (dataTruth['ele2_ptvarcone20'].div(dataTruth['ele2_et'])).values.tolist()
    dataTruth['ele2_topoetcone20_rel'] = (dataTruth['ele2_topoetcone20'].div(dataTruth['ele2_et'])).values.tolist()
    dataTruth["ele_deltaZ0"] = np.abs((dataTruth["ele1_z0"].sub(dataTruth["ele2_z0"])).values.tolist())
    dataTruth["ele_deltaZ0sig"] = np.sqrt(((dataTruth["ele1_z0Sig"]**2).add(dataTruth["ele2_z0Sig"]**2)).values.tolist())
    dataTruth["ele_deltad0"] = np.abs((dataTruth["ele1_d0"].sub(dataTruth["ele2_d0"])).values.tolist())
    dataTruth["ele_deltad0sig"] = np.sqrt(((dataTruth["ele1_d0Sig"]**2).add(dataTruth["ele2_d0Sig"]**2)).values.tolist())

    dataLEP['ele1_ptvarcone40_rel'] = (dataLEP['ele1_ptvarcone40_TightTTVALooseCone_pt1000'].div(dataLEP['ele1_et'])).values.tolist()
    dataLEP['ele1_topoetcone40_rel'] = (dataLEP['ele1_topoetcone40'].div(dataLEP['ele1_et'])).values.tolist()
    dataLEP['ele2_ptvarcone40_rel'] = (dataLEP['ele2_ptvarcone40_TightTTVALooseCone_pt1000'].div(dataLEP['ele2_et'])).values.tolist()
    dataLEP['ele2_topoetcone40_rel'] = (dataLEP['ele2_topoetcone40'].div(dataLEP['ele2_et'])).values.tolist()

    dataLEP['ele1_ptvarcone20_rel'] = (dataLEP['ele1_ptvarcone20'].div(dataLEP['ele1_et'])).values.tolist()
    dataLEP['ele1_topoetcone20_rel'] = (dataLEP['ele1_topoetcone20'].div(dataLEP['ele1_et'])).values.tolist()
    dataLEP['ele2_ptvarcone20_rel'] = (dataLEP['ele2_ptvarcone20'].div(dataLEP['ele2_et'])).values.tolist()
    dataLEP['ele2_topoetcone20_rel'] = (dataLEP['ele2_topoetcone20'].div(dataLEP['ele2_et'])).values.tolist()
    dataLEP["ele_deltaZ0"] = np.abs((dataLEP["ele1_z0"].sub(dataLEP["ele2_z0"])).values.tolist())
    dataLEP["ele_deltaZ0sig"] = np.sqrt(((dataLEP["ele1_z0Sig"]**2).add(dataLEP["ele2_z0Sig"]**2)).values.tolist())
    dataLEP["ele_deltad0"] = np.abs((dataLEP["ele1_d0"].sub(dataLEP["ele2_d0"])).values.tolist())
    dataLEP["ele_deltad0sig"] = np.sqrt(((dataLEP["ele1_d0Sig"]**2).add(dataLEP["ele2_d0Sig"]**2)).values.tolist())
elif (args.PartType == "muo") or (args.PartType == "Data"):
    dataTruth['muo1_ptvarcone40_rel'] = (dataTruth['muo1_ptvarcone40'].div(dataTruth['muo1_pt'])).values.tolist()
    dataTruth['muo2_ptvarcone40_rel'] = (dataTruth['muo2_ptvarcone40'].div(dataTruth['muo2_pt'])).values.tolist()

    dataTruth['muo1_ptvarcone20_rel'] = (dataTruth['muo1_ptvarcone20'].div(dataTruth['muo1_pt'])).values.tolist()
    dataTruth['muo2_ptvarcone20_rel'] = (dataTruth['muo2_ptvarcone20'].div(dataTruth['muo2_pt'])).values.tolist()

    dataLEP['muo1_ptvarcone40_rel'] = (dataLEP['muo1_ptvarcone40'].div(dataLEP['muo1_pt'])).values.tolist()
    dataLEP['muo2_ptvarcone40_rel'] = (dataLEP['muo2_ptvarcone40'].div(dataLEP['muo2_pt'])).values.tolist()

    dataLEP['muo1_ptvarcone20_rel'] = (dataLEP['muo1_ptvarcone20'].div(dataLEP['muo1_pt'])).values.tolist()
    dataLEP['muo2_ptvarcone20_rel'] = (dataLEP['muo2_ptvarcone20'].div(dataLEP['muo2_pt'])).values.tolist()
#It has to be converted to a dataframe, for the scores to be calculated
if args.PartType != "Hgg":
    dataTruth["pho_pIso_score"] = pIsoScore(pIsoGBM, dataTruth,0, n_jobs = args.njobs)
    dataTruth["pho_pPid_score"] = pPidScore(pPidGBM, dataTruth,0, n_jobs = args.njobs)

    dataPho["pho_pIso_score"] = pIsoScore(pIsoGBM, dataPho,0, n_jobs = args.njobs)
    dataPho["pho_pPid_score"] = pPidScore(pPidGBM, dataPho,0, n_jobs = args.njobs)
if (args.PartType == "ele"):
    dataTruth["ele1_eIso_score"] = eIsoScore(lIsoGBM, dataTruth, 1, n_jobs = args.njobs)
    dataTruth["ele1_ePid_score"] = ePidScore(lPidGBM, dataTruth, 1, n_jobs=args.njobs)
    dataTruth["ele2_eIso_score"] = eIsoScore(lIsoGBM, dataTruth, 2, n_jobs = args.njobs)
    dataTruth["ele2_ePid_score"] = ePidScore(lPidGBM, dataTruth, 2, n_jobs = args.njobs)
    dataTruth["ee_score"] = ZeeScore(ZllGBM, dataTruth, n_jobs = args.njobs)
    dataTruth["eeg_score"] = ZeegScore(ZllgGBM, dataTruth, n_jobs = args.njobs)

    dataLEP["ele1_eIso_score"] = eIsoScore(lIsoGBM, dataLEP, 1, n_jobs = args.njobs)
    dataLEP["ele1_ePid_score"] = ePidScore(lPidGBM, dataLEP, 1, n_jobs=args.njobs)
    dataLEP["ele2_eIso_score"] = eIsoScore(lIsoGBM, dataLEP, 2, n_jobs = args.njobs)
    dataLEP["ele2_ePid_score"] = ePidScore(lPidGBM, dataLEP, 2, n_jobs = args.njobs)
    dataLEP["ee_score"] = ZeeScore(ZllGBM, dataLEP, n_jobs = args.njobs)
    #dataLEP["eeg_score"] = ZeegScore(ZllgGBM, dataLEP, n_jobs = args.njobs)
elif (args.PartType == "muo") or (args.PartType == "Data"):
    dataTruth["muo1_mIso_score"] = mIsoScore(lIsoGBM, dataTruth, 1, n_jobs = args.njobs)
    dataTruth["muo1_mPid_score"] = mPidScore(lPidGBM, dataTruth, 1, n_jobs=args.njobs)
    dataTruth["muo2_mIso_score"] = mIsoScore(lIsoGBM, dataTruth, 2, n_jobs = args.njobs)
    dataTruth["muo2_mPid_score"] = mPidScore(lPidGBM, dataTruth, 2, n_jobs = args.njobs)
    dataTruth["mm_score"] = ZmmScore(ZllGBM, dataTruth, n_jobs = args.njobs)
    dataTruth["mmg_score"] = ZmmgScore(ZllgGBM, dataTruth, n_jobs = args.njobs)

    dataLEP["muo1_mIso_score"] = mIsoScore(lIsoGBM, dataLEP, 1, n_jobs = args.njobs)
    dataLEP["muo1_mPid_score"] = mPidScore(lPidGBM, dataLEP, 1, n_jobs=args.njobs)
    dataLEP["muo2_mIso_score"] = mIsoScore(lIsoGBM, dataLEP, 2, n_jobs = args.njobs)
    dataLEP["muo2_mPid_score"] = mPidScore(lPidGBM, dataLEP, 2, n_jobs = args.njobs)
    dataLEP["mm_score"] = ZmmScore(ZllGBM, dataLEP, n_jobs = args.njobs)
    #dataLEP["mmg_score"] = ZmmgScore(ZllgGBM, dataLEP, n_jobs = args.njobs)
elif args.PartType == "Hgg":
    dataTruth["pho_deltaZ0"] = np.abs(dataTruth["pho1_z0"]-dataTruth["pho2_z0"])
    dataTruth["pho_deltaZ0sig"] = np.sqrt(dataTruth["pho1_z0Sig"]**2+dataTruth["pho2_z0Sig"]**2)
    dataTruth["pho1_pIso_score"] = pIsoScore(pIsoGBM, dataTruth,1, n_jobs = args.njobs)
    dataTruth["pho1_pPid_score"] = pPidScore(pPidGBM, dataTruth,1, n_jobs = args.njobs)
    dataTruth["pho2_pIso_score"] = pIsoScore(pIsoGBM, dataTruth,2, n_jobs = args.njobs)
    dataTruth["pho2_pPid_score"] = pPidScore(pPidGBM, dataTruth,2, n_jobs = args.njobs)
    dataTruth["gg_score"] = HggScore(HggGBM,dataTruth,n_jobs=args.njobs)
    
#introducing the invMll<82GeV restriction
if args.PartType != "Hgg":
    print("Requiring invMll<82GeV")
    DATATruth = dataTruth[(dataTruth["invMll"]<82) & (dataTruth["invM"]>mininvm) & (dataTruth["invM"]<maxinvm)].copy()
    DATALEP = dataLEP[(dataLEP["invMll"]<82) & (dataLEP["invM"]>mininvm) & (dataLEP["invM"]<maxinvm)].copy()
    DATAPho = dataPho[(dataPho["invMll"]<82) & (dataPho["invM"]>mininvm) & (dataPho["invM"]<maxinvm)].copy()
if args.PartType == "Hgg":
    DATATruth = dataTruth[(dataTruth["invM"]>mininvm) &(dataTruth["invM"]<maxinvm )].copy()
    databoth =DATATruth.copy()

#The following code will apply the different models, and plot the cuts in the distribution of scores.

if args.PartType == "Data":
    #############################
    # Testing DATA: Same cuts as Zmmg MC
    #############################
    # This test results in poor results
    databoth = DATATruth.copy()
    databoth2 = databoth.copy()
    databoth3 = databoth.copy()
    def llcut(cut):
        return (databoth["mm_score"]>cut)
    databoth = databoth[llcut(2.622)]
    def piso(cut):
        return (databoth["pho_pIso_score"]>cut)
    databoth = databoth[piso(-7.214)]
    def ppid(cut):
        return (databoth["pho_pPid_score"]>cut)
    databoth = databoth[ppid(1)]

    dataFML = DATATruth.copy()
    dataFML2 = dataFML.copy()
    def llgcut(cut):
        return (dataFML["mmg_score"]>cut)
    dataFML = dataFML[llgcut(0.399)]
    DATALEP2 = DATALEP.copy()
    def llscore(cut):
        return (DATALEP["mm_score"]>cut)
    DATALEP = DATALEP[llscore(5.367)]
    #Applying ptvarcone20 efficiency cut.
    phoptvarcone = np.concatenate((DATAPho['muo1_ptvarcone20'],DATAPho['muo2_ptvarcone20']))
    phoptvarcone = np.sort(phoptvarcone)

    nremove = np.int(phoptvarcone.shape[0]*0.01)
    phoptvarconecut = (phoptvarcone[-(nremove+1)])
    
    def mask_Zllgammapho(cut):
        return (( DATAPho['muo1_ptvarcone20'] < cut ) &
                ( DATAPho['muo2_ptvarcone20'] < cut ) 
                )
    DATAPho = DATAPho[mask_Zllgammapho(phoptvarconecut)]
    DATAPho2 = DATAPho.copy()

    def piso(cut):
        return (DATAPho>cut)
    DATAPho = DATAPho[piso(-7.212)]

    def ppid(cut):
        return (DATAPho>cut)
    DATAPho = DATAPho[ppid(0.745)]

    ####################################################################################################################
    # Testing DATA: Same "signal in outer parts as ATLAS" - assumed background 
    ####################################################################################################################

    ##################
    # Removing 5% ll-score
    ##################
    print(f"Running selections using both lep pair and photon scores")
    
    print(f"Removing the lowest 5% of ll-score from signal (and including that cut for background)")
    sortleppair = np.sort(databoth2["mm_score"])
    nremove = np.int(sortleppair.shape[0]*0.05)
    leppaircut = sortleppair[nremove-1]
    def llscore(cut):
        return (databoth2["mm_score"]>cut)
    print(f"Printing the 5% ll-score cut")

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(databoth2[llscore(leppaircut)])/len(databoth2)-1)*100
    ax.hist(databoth2["mm_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.set_xlabel("mm_score", fontsize=20)
    ax.set_title(f"Cut in mm_score, using 95% signal efficiency", fontsize=20)
    ax.axvline(x=leppaircut,color = "r", linestyle = "dashed", linewidth = 2, label = f"lep-pair cut: {leppaircut:.3f}")
    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    fig.savefig(figure_dir+"both_mmscorecut")
    databoth2 = databoth2[llscore(leppaircut)]

    ##################
    # Removing 1% piso-score (Photon Isolation Score)
    ##################
    print(f"Removing the lowest 1% of piso-score from signal (and including that cut for background)")
    sortpiso = np.sort(databoth2["pho_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (databoth2["pho_pIso_score"]>cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(databoth2[pisoscore(pisocut)])/len(databoth2)-1)*100
    ax.hist(databoth2["pho_pIso_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"both_pIsoscorecut")
    databoth2 = databoth2[pisoscore(pisocut)]

    ##################
    # matching backgrounds using pPid-score
    ##################
    print(f"Now setting pPid score, so there is the same number of events in the outer area (60-80 and 100-140)")
    
    sortppid = np.sort(databoth2[((databoth2["invM"]>60)& (databoth2["invM"]<80)) | ((databoth2["invM"]>100)& (databoth2["invM"]<140))]["pho_pPid_score"])

    nboth2outer = len(sortppid)
    nremove = np.abs(nATLASouter-nboth2outer)
    print(f"{nremove} backgrounds will be removed from the both peak")
    sortppidcut = (sortppid[nremove-1])
    print(f"The cut in pPid for the model using both Lep pair and pho model has been set to {sortppidcut}, the min value is {sortppid[0]}, with the highest value being {sortppid[-1]}, with the mean {np.mean(sortppid)}.")
    def ppidscore(cut):
        return (databoth2["pho_pPid_score"]>cut)
    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(databoth2[ppidscore(sortppidcut)])/len(databoth2)-1)*100
    ax.hist(databoth2["pho_pPid_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.axvline(x=sortppidcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PPid cut: {sortppidcut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score", fontsize=20)
    fig.savefig(figure_dir+"both_pPidscorecut")
    databoth2 = databoth2[ppidscore(sortppidcut)]

    ################################################################################################
    # Make the Full ML cuts
    ################################################################################################
    print("Running selections using the full model")


    sortllg = np.sort(dataFML2[((dataFML2["invM"]>60)& (dataFML2["invM"]<80)) | ((dataFML2["invM"]>100)& (dataFML2["invM"]<140))]["mmg_score"])
    #Start med at se mht peak, bagefter kan man køre med fulde run, hvis der er meget lidt baggrund
    nFML2outer = len(sortllg)
    nremove = np.abs(nATLASouter-nFML2outer)
    sortllgcut = (sortllg[nremove-1])
    print(f"The cut in eeg score for the full model has been set to {sortllgcut}, the min value is {sortllg[0]}, with the highest value being {sortllg[-1]}, with the mean {np.mean(sortllg)}.")
    def llgscore(cut):
        return (dataFML2["mmg_score"]>cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (sum(llgscore(sortllgcut))/len(dataFML)-1)*100
    ax.hist(dataFML2["mmg_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.set_xlabel("mmg_score", fontsize=20)
    ax.set_title(f"Cut in mmg_score", fontsize=20)
    ax.axvline(x=sortllgcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"mmg cut: {sortllgcut:.3f}")

    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    fig.savefig(figure_dir+"FML_mmgscorecut")
    dataFML2 = dataFML2[llgscore(sortllgcut)]
    ################################################################################################
    # Make the lep pair cuts (ATLAS photon)
    ################################################################################################
    ##################
    # Matching backgrounds
    ##################
    sortleppair = np.sort(DATALEP2[((DATALEP2["invM"]>60)& (DATALEP2["invM"]<80)) | ((DATALEP2["invM"]>100)& (DATALEP2["invM"]<140))]["mm_score"])
    print(f"Matching backgrounds for the lep pair models")
    nLEPouter = len(sortleppair)
    nremove = np.abs(nATLASouter-nLEPouter)
    leppaircut = sortleppair[nremove-1]
    def llscore(cut):
        return (DATALEP2["mm_score"]>cut)


    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(DATALEP2[llscore(leppaircut)])/len(DATALEP2)-1)*100
    ax.hist(DATALEP2["mm_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.set_xlabel("mm_score", fontsize=20)
    ax.set_title(f"Cut in mm_score, for background equal to ATLAS", fontsize=20)
    ax.axvline(x=leppaircut,color = "r", linestyle = "dashed", linewidth = 2, label = f"lep-pair cut: {leppaircut:.3f}")
    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    fig.savefig(figure_dir+"LEP_eescorecut")
    DATALEP2 = DATALEP2[llscore(leppaircut)]



    ################################################################################################
    # Make the photon cuts (ATLAS lep)
    ################################################################################################
    ##################
    # Removing top 1% ptvarcone, when copied it has already gotten 1% of ptvarcone removed
    ##################

    ##################
    # Removing 1% pIso
    ##################
    print(f"Removing the lowest 1% of piso-score from signal (and including that cut for background)")
    sortpiso = np.sort(DATAPho2["pho_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (DATAPho2["pho_pIso_score"]>cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(DATAPho2[pisoscore(pisocut)])/len(DATAPho2)-1)*100
    ax.hist(DATAPho2["pho_pIso_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"pho_pIsoscorecut")
    DATAPho2 = DATAPho2[pisoscore(pisocut)]

    ##################
    # matching backgrounds using pPid-score
    ##################
    print(f"Now setting pPid score, so there is the same background in the peak area")
    sortppid = np.sort(DATAPho2[((DATAPho2["invM"]>60)& (DATAPho2["invM"]<80)) | ((DATAPho2["invM"]>100)& (DATAPho2["invM"]<140))]["pho_pPid_score"])
    nphoouter = len(sortppid)
    nremove = np.abs(nATLASouter-nphoouter)
    sortppidcut = (sortppid[nremove-1])
    print(f"The cut in pPid for the model using both Lep pair and pho model has been set to {sortppidcut}, the min value is {sortppid[0]}, with the highest value being {sortppid[-1]}, with the mean {np.mean(sortppid)}.")
    def ppidscore(cut):
        return (DATAPho2["pho_pPid_score"]>cut)
    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = (len(DATAPho2[ppidscore(sortppidcut)])/len(DATAPho2)-1)*100
    ax.hist(DATAPho2["pho_pPid_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.axvline(x=sortppidcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PPid cut: {sortppidcut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score", fontsize=20)
    fig.savefig(figure_dir+"pho_pPidscorecut")
    DATAPho2 = DATAPho2[ppidscore(sortppidcut)]

    
if (args.PartType !="Hgg") and (args.PartType != "Data"):
    ################################################################################################
    # Make the both cuts
    ################################################################################################

    ##################
    # Removing 5% ll-score
    ##################
    print(f"Running selections using both lep pair and photon scores")
    databoth = DATATruth.copy()
    print(f"Removing the lowest 5% of ll-score from signal (and including that cut for background)")
    if args.PartType == "ele":
        sortleppair = np.sort(databoth[databoth["type"]==1]["ee_score"])
    elif args.PartType == "muo":
        sortleppair = np.sort(databoth[databoth["type"]==1]["mm_score"])
    nremove = np.int(sortleppair.shape[0]*0.05)
    leppaircut = sortleppair[nremove-1]
    def llscore(cut):
        if args.PartType == "ele":
            return (databoth["ee_score"]<cut)
        elif args.PartType == "muo":
            return (databoth["mm_score"]<cut)
    print(f"Printing the 5% ll-score cut")

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(databoth[(llscore(leppaircut))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(llscore(leppaircut))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    # After having applied the cuts, the cuts are plotted
    if args.PartType == "ele":
        ax.hist(databoth[databoth["type"]==1]["ee_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(databoth[databoth["type"]==0]["ee_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("ee_score", fontsize=20)
        ax.set_title(f"Cut in ee_score, using 95% signal efficiency", fontsize=20)
    elif args.PartType == "muo":
        ax.hist(databoth[databoth["type"]==1]["mm_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(databoth[databoth["type"]==0]["mm_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("mm_score", fontsize=20)
        ax.set_title(f"Cut in mm_score, using 95% signal efficiency", fontsize=20)
    ax.axvline(x=leppaircut,color = "r", linestyle = "dashed", linewidth = 2, label = f"lep-pair cut: {leppaircut:.3f}")
    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    if args.PartType == "ele":
        fig.savefig(figure_dir+"both_eescorecut")
    elif args.PartType == "muo":
        fig.savefig(figure_dir+"both_mmscorecut")
    databoth["type"][llscore(leppaircut)] = 2



    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0  after removing 5% signal (ee) in the both part")

    ##################
    # Removing 1% piso-score
    ##################
    print(f"Removing the lowest 1% of piso-score from signal (and including that cut for background)")
    sortpiso = np.sort(databoth[databoth["type"]==1]["pho_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (databoth["pho_pIso_score"]<cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    ax.hist(databoth["pho_pIso_score"][databoth["type"]==1],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(databoth["pho_pIso_score"][databoth["type"]==0],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"both_pIsoscorecut")
    databoth["type"][pisoscore(pisocut)]= 2

    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after removing 1% iso in the both part")

    ##################
    # matching backgrounds using pPid-score
    ##################
    print(f"Now setting pPid score, so there is the same background in the peak area")
    nbothpeakbkg = len(databoth[(databoth["type"]==0)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
    sortppid = np.sort(databoth[(databoth["type"]==0)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)]["pho_pPid_score"])
    nremove = np.abs(nATLASpeakbkg-nbothpeakbkg)
    print(f"{nremove} backgrounds will be removed from the both peak")
    sortppidcut = (sortppid[nremove-1])
    print(f"The cut in pPid for the model using both Lep pair and pho model has been set to {sortppidcut}, the min value is {sortppid[0]}, with the highest value being {sortppid[-1]}, with the mean {np.mean(sortppid)}.")
    def ppidscore(cut):
        return (databoth["pho_pPid_score"]<cut)
    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(databoth[(ppidscore(sortppidcut))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(ppidscore(sortppidcut))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    ax.hist(databoth[databoth["type"]==1]["pho_pPid_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(databoth[databoth["type"]==0]["pho_pPid_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=sortppidcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PPid cut: {sortppidcut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score", fontsize=20)
    fig.savefig(figure_dir+"both_pPidscorecut")
    databoth["type"][ppidscore(sortppidcut)]=2 
    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is finally {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 in the both part") 
    print(f"There should be {nATLASpeakbkg} background in the peak")





    ################################################################################################
    # Make the Full ML cuts
    ################################################################################################
    dataFML = DATATruth.copy()
    print("Running selections using the full model")

    nFMLpeakbkg = len(dataFML[(dataFML["type"]==0)&(dataFML["invM"]<maxpeakinvm) & (dataFML["invM"]>minpeakinvm)])


    #Start med at se mht peak, bagefter kan man køre med fulde run, hvis der er meget lidt baggrund
    print(f"There are {nATLASpeakbkg} background in the peak area for ATLAS, and {nFMLpeakbkg} for the full model")
    if args.PartType == "ele":
        sortllg = np.sort(dataFML[(dataFML["type"]==0)&(dataFML["invM"]<maxpeakinvm)&(dataFML["invM"]>minpeakinvm)]["eeg_score"])
    elif args.PartType == "muo":
        sortllg = np.sort(dataFML[(dataFML["type"]==0)&(dataFML["invM"]<maxpeakinvm)&(dataFML["invM"]>minpeakinvm)]["mmg_score"])
    nremove = np.abs(nATLASpeakbkg-nFMLpeakbkg)
    sortllgcut = (sortllg[nremove-1])
    print(f"The cut in eeg score for the full model has been set to {sortllgcut}, the min value is {sortllg[0]}, with the highest value being {sortllg[-1]}, with the mean {np.mean(sortllg)}.")
    def llgscore(cut):
        if args.PartType == "ele":
            return (dataFML["eeg_score"]<cut)
        elif args.PartType == "muo":
            return (dataFML["mmg_score"]<cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)


    print("The length of the cut binary: ", len(llgscore(sortllgcut)), ". The length of signal: ", sum(dataFML["type"]==1) , ". The length of background: " , sum(dataFML["type"]==0))
    sigdiff = -sum(llgscore(sortllgcut)*(dataFML["type"]==1))/sum(dataFML["type"]==1)*100
    bkgdiff = -sum(llgscore(sortllgcut)*(dataFML["type"]==0))/sum(dataFML["type"]==0)*100
    if args.PartType == "ele":
        ax.hist(dataFML[dataFML["type"]==1]["eeg_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(dataFML[dataFML["type"]==0]["eeg_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("eeg_score", fontsize=20)
        ax.set_title(f"Cut in eeg_score", fontsize=20)
        ax.axvline(x=sortllgcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"eeg cut: {sortllgcut:.3f}")
    elif args.PartType == "muo":
        ax.hist(dataFML[dataFML["type"]==1]["mmg_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(dataFML[dataFML["type"]==0]["mmg_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("mmg_score", fontsize=20)
        ax.set_title(f"Cut in mmg_score", fontsize=20)
        ax.axvline(x=sortllgcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"mmg cut: {sortllgcut:.3f}")

    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    fig.savefig(figure_dir+"FML_llgscorecut")
    dataFML["type"][llgscore(sortllgcut)]=2    
    testfmlpeakbkg = len(dataFML[(dataFML["type"]==0)&(dataFML["invM"]<maxpeakinvm) & (dataFML["invM"]>minpeakinvm)])
    print(f"There are now {testfmlpeakbkg} FML bkg in peak, and {nATLASpeakbkg} ATLAS peak in bkg")
    ################################################################################################
    # Make the lep pair cuts (ATLAS photon)
    ################################################################################################
    ##################
    # Matching backgrounds
    ##################
    nLEPpeakbkg = len(DATALEP[(DATALEP["type"]==0)&(DATALEP["invM"]<maxpeakinvm) & (DATALEP["invM"]>minpeakinvm)])
    print(f"Matching backgrounds for the lep pair models")
    if args.PartType == "ele":
        sortleppair = np.sort(DATALEP[(DATALEP["type"]==0)&(DATALEP["invM"]<maxpeakinvm) & (DATALEP["invM"]>minpeakinvm)]["ee_score"])
    elif args.PartType == "muo": 
        sortleppair = np.sort(DATALEP[(DATALEP["type"]==0)&(DATALEP["invM"]<maxpeakinvm) & (DATALEP["invM"]>minpeakinvm)]["mm_score"])
    nremove = np.abs(nATLASpeakbkg-nLEPpeakbkg)
    leppaircut = sortleppair[nremove-1]
    def llscore(cut):
        if args.PartType == "ele":
            return (DATALEP["ee_score"]<cut)
        elif args.PartType == "muo":
            return (DATALEP["mm_score"]<cut)


    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(DATALEP[(llscore(leppaircut))&(DATALEP["type"]==1)]))/len(DATALEP[DATALEP["type"]==1])*100
    bkgdiff = -(len(DATALEP[(llscore(leppaircut))&(DATALEP["type"]==0)]))/len(DATALEP[DATALEP["type"]==0])*100
    if args.PartType == "ele":
        ax.hist(DATALEP[DATALEP["type"]==1]["ee_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(DATALEP[DATALEP["type"]==0]["ee_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("ee_score", fontsize=20)
        ax.set_title(f"Cut in ee_score, for background equal to ATLAS", fontsize=20)
    elif args.PartType == "muo":
        ax.hist(DATALEP[DATALEP["type"]==1]["mm_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
        ax.hist(DATALEP[DATALEP["type"]==0]["mm_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
        ax.set_xlabel("mm_score", fontsize=20)
        ax.set_title(f"Cut in mm_score, for background equal to ATLAS", fontsize=20)
    ax.axvline(x=leppaircut,color = "r", linestyle = "dashed", linewidth = 2, label = f"lep-pair cut: {leppaircut:.3f}")
    ax.legend(loc='best', fontsize=20)

    ax.set_ylabel("Frequency", fontsize=20)

    fig.savefig(figure_dir+"LEP_llscorecut")
    DATALEP["type"][llscore(leppaircut)] = 2



    ################################################################################################
    # Make the photon cuts (ATLAS lep)
    ################################################################################################
    ##################
    # Removing top 1% ptvarcone
    ##################
    if args.PartType =="ele":
        phoptvarcone = np.concatenate((DATAPho['ele1_ptvarcone20'],DATAPho['ele2_ptvarcone20']))
    elif args.PartType =="muo":
        phoptvarcone = np.concatenate((DATAPho['muo1_ptvarcone20'],DATAPho['muo2_ptvarcone20']))
    phoptvarcone = np.sort(phoptvarcone)

    nremove = np.int(phoptvarcone.shape[0]*0.01)
    phoptvarconecut = (phoptvarcone[-(nremove+1)])
    if args.PartType == "ele":
        def mask_Zllgammapho(cut):
            return (( DATAPho['ele1_ptvarcone20'] < cut ) &
                    ( DATAPho['ele2_ptvarcone20'] < cut ) 
                    )
    elif args.PartType == "muo":
        def mask_Zllgammapho(cut):
            return (( DATAPho['muo1_ptvarcone20'] < cut ) &
                    ( DATAPho['muo2_ptvarcone20'] < cut ) 
                    )
    DATAPho = DATAPho[mask_Zllgammapho(phoptvarconecut)]

    ##################
    # Removing 1% pIso
    ##################
    print(f"Removing the lowest 1% of piso-score from signal (and including that cut for background)")
    sortpiso = np.sort(DATAPho[DATAPho["type"]==1]["pho_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (DATAPho["pho_pIso_score"]<cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(DATAPho[(pisoscore(pisocut))&(DATAPho["type"]==1)]))/len(DATAPho[DATAPho["type"]==1])*100
    bkgdiff = -(len(DATAPho[(pisoscore(pisocut))&(DATAPho["type"]==0)]))/len(DATAPho[DATAPho["type"]==0])*100
    ax.hist(DATAPho["pho_pIso_score"][DATAPho["type"]==1],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(DATAPho["pho_pIso_score"][DATAPho["type"]==0],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"pho_pIsoscorecut")
    DATAPho["type"][pisoscore(pisocut)]= 2

    nboth0 = len(DATAPho[DATAPho["type"]==0])
    nboth1 = len(DATAPho[DATAPho["type"]==1])
    nboth2 = len(DATAPho[DATAPho["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after removing 1% iso in the both part")


    ##################
    # matching backgrounds using pPid-score
    ##################
    print(f"Now setting pPid score, so there is the same background in the peak area")
    nphopeakbkg = len(DATAPho[(DATAPho["type"]==0)&(DATAPho["invM"]<maxpeakinvm)&(DATAPho["invM"]>minpeakinvm)])
    sortppid = np.sort(DATAPho[(DATAPho["type"]==0)&(DATAPho["invM"]<maxpeakinvm)&(DATAPho["invM"]>minpeakinvm)]["pho_pPid_score"])
    nremove = np.abs(nATLASpeakbkg-nphopeakbkg)
    print(f"Need {nremove} removed, got {len(sortppid)}")
    sortppidcut = (sortppid[nremove-1])
    print(f"The cut in pPid for the model using both Lep pair and pho model has been set to {sortppidcut}, the min value is {sortppid[0]}, with the highest value being {sortppid[-1]}, with the mean {np.mean(sortppid)}.")
    def ppidscore(cut):
        return (DATAPho["pho_pPid_score"]<cut)
    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(DATAPho[(ppidscore(sortppidcut))&(DATAPho["type"]==1)]))/len(DATAPho[DATAPho["type"]==1])*100
    bkgdiff = -(len(DATAPho[(ppidscore(sortppidcut))&(DATAPho["type"]==0)]))/len(DATAPho[DATAPho["type"]==0])*100
    ax.hist(DATAPho[DATAPho["type"]==1]["pho_pPid_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(DATAPho[DATAPho["type"]==0]["pho_pPid_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=sortppidcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PPid cut: {sortppidcut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score", fontsize=20)
    fig.savefig(figure_dir+"pho_pPidscorecut")
    DATAPho["type"][ppidscore(sortppidcut)]=2 
elif args.PartType =="Hgg":
    ################################################################################################
    # Matching Hgg model with ATLAS selection
    ################################################################################################
    print(f"Now setting gg_score, so there is the same background in peak area")
    
    ntruthbkg = len(DATATruth[(DATATruth["type"]==0)&(DATATruth["invM"]<maxpeakinvm)&(DATATruth["invM"]>minpeakinvm)])
    print(f"There are {nATLASbkg} ATLAS bkg in the peak, and {ntruthbkg} of mine")
    fig, ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(mininvm,maxinvm,maxinvm-mininvm+1)
    #Plotting some signal and background, to see initial distribution.
    ax.hist(DATATruth[DATATruth["type"]==1]["invM"],bins=nbins,histtype="step",label=f"Signal True")
    ax.hist(DATATruth[DATATruth["type"]==0]["invM"],bins=nbins,histtype="step",label=f"Bkg True")
    ax.hist(dataATLAS[dataATLAS["type"]==1]["invM"],bins=nbins,histtype="step",label=f"Signal ATLAS")
    ax.hist(dataATLAS[dataATLAS["type"]==0]["invM"],bins=nbins,histtype="step",label=f"Bkg ATLAS")
    ax.legend(loc='best',fontsize=20)
    fig.savefig(figure_dir+"gginvMtest")
    
    #Finding the gg_score cut and applying it
    sortgg = np.sort(DATATruth[(DATATruth["type"]==0)&(DATATruth["invM"]<maxpeakinvm)&(DATATruth["invM"]>minpeakinvm)]["gg_score"])
    nremove = np.abs(nATLASpeakbkg-ntruthbkg)
    sortggcut = sortgg[nremove-1]
    print(f"The cut in gg_score for the full ML model has been set to {sortggcut}, the min value is {sortgg[0]}, with the highest value being {sortgg[-1]}, with the mean {np.mean(sortgg)}.")
    def ggscore(cut):
        return (DATATruth["gg_score"]<cut)
    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -sum(ggscore(sortggcut)*(DATATruth["type"]==1))/sum(dataTruth["type"]==1)*100
    bkgdiff = -sum(ggscore(sortggcut)*(DATATruth["type"]==0))/sum(dataTruth["type"]==0)*100
    ax.hist(DATATruth[DATATruth["type"]==1]["gg_score"],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(DATATruth[DATATruth["type"]==0]["gg_score"],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=sortggcut,color = "r", linestyle = "dashed", linewidth = 2, label = r"$\gamma\gamma$ cut")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel(r"$\gamma\gamma$_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(r"Cut in $\gamma\gamma$_score", fontsize=20)
    fig.savefig(figure_dir+"ggscorecut")
    DATATruth["type"][ggscore(sortggcut)]=2 
    
    ################################################################################################
    # Using photon isolation and identification
    ################################################################################################
    print(" ")
    print(f"There are a total of {nATLAS} signal for ATLAS and {nATLASbkg} background for ATLAS.")
    print(" ")
    ##############################
    # Applying 99% signal efficiency on p1_iso
    ##############################
    print(f"Removing the lowest 1% of p1iso-score from signal (and including that cut for background)")
    sortpiso = np.sort(databoth[databoth["type"]==1]["pho1_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (databoth["pho1_pIso_score"]<cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    ax.hist(databoth["pho1_pIso_score"][databoth["type"]==1],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(databoth["pho1_pIso_score"][databoth["type"]==0],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho1_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho1_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"pho1_pIsoscorecut")
    databoth["type"][pisoscore(pisocut)]= 2

    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after removing 1% iso in the both part")



    ##############################
    # Applying 99% signal efficiency on p2_iso
    ##############################
    print(f"Removing the lowest 1% of p2iso-score from signal (and including that cut for background)")
    sortpiso = np.sort(databoth[databoth["type"]==1]["pho2_pIso_score"])
    nremove = np.int(sortpiso.shape[0]*0.01)
    pisocut = sortpiso[nremove-1]
    def pisoscore(cut):
        return (databoth["pho2_pIso_score"]<cut)

    fig,ax = plt.subplots(figsize=(12,12))
    nbins = np.linspace(-10,10,101)
    sigdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(pisoscore(pisocut))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    ax.hist(databoth["pho2_pIso_score"][databoth["type"]==1],bins=nbins,histtype="step",label=f"Signal difference: {sigdiff:.1f}%")
    ax.hist(databoth["pho2_pIso_score"][databoth["type"]==0],bins=nbins,histtype="step",label=f"Bkg difference: {bkgdiff:.1f}%")
    ax.axvline(x=pisocut,color = "r", linestyle = "dashed", linewidth = 2, label = f"PIso cut: {pisocut:.3f}")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho2_pIso_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho2_pIso_score, using 99% signal efficiency", fontsize=20)
    fig.savefig(figure_dir+"pho2_pIsoscorecut")
    databoth["type"][pisoscore(pisocut)]= 2

    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after removing 1% iso in the both part")
    databoth2 = databoth.copy()
    ##############################
    # Making grid search, to see what ppid to pick, same background as ATLAS, but optimize signal
    ##############################
    
    print(f"Running optimization of ppid, doing search, trying to optimize signal given same background as ATLAS")
    ntest = 0
    counter = 0
    p1pid, p2pid = [], []
    r_min, r_max =-10.0,10.0
    #Require either 5000 combinations which has the same amount of background as ATLAS, or having found 500 combinations 200 times - so it wont do an infinite loop.
    while ntest<5000:
        counter +=1
        p1p, p2p = r_min+rand(500)*(r_max-r_min),r_min+rand(500)*(r_max-r_min)
        for i in range(len(p1p)):
            nbkg = len(databoth[(databoth["type"]==0)&(databoth["pho1_pPid_score"]>p1p[i])&(databoth["pho2_pPid_score"]>p2p[i])&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
            if nbkg == nATLASpeakbkg:
                p1pid.append(p1p[i])
                p2pid.append(p2p[i])
                ntest +=1
        if counter == 200:
            break
    print(f"{ntest} candidates were found")
    bestscores = 0
    nbestsig = 0
    #Finding the best of the candidates
    for i in range(len(p1pid)):
        nsigi = len(databoth[(databoth["type"]==1)&(databoth["pho1_pPid_score"]>p1pid[i])&(databoth["pho2_pPid_score"]>p2pid[i])&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
        if nsigi > nbestsig:
            bestscores = i
            nbestsig = len(databoth[(databoth["type"]==1)&(databoth["pho1_pPid_score"]>p1pid[bestscores])&(databoth["pho2_pPid_score"]>p2pid[bestscores])&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
            print(f"New best score found, with {nbestsig} signal")
            
    
    nsig = len(databoth[(databoth["type"]==1)&(databoth["pho1_pPid_score"]>p1pid[bestscores])&(databoth["pho2_pPid_score"]>p2pid[bestscores])])
    nbkg = len(databoth[(databoth["type"]==0)&(databoth["pho1_pPid_score"]>p1pid[bestscores])&(databoth["pho2_pPid_score"]>p2pid[bestscores])])
    print(f"The best result had: {nsig} signal, {nbkg} background.")
    #Applying the cut. Then plotting
    def pidcut(cutindex):
        return ((databoth["pho1_pPid_score"]<p1pid[cutindex])|(databoth["pho2_pPid_score"]<p2pid[cutindex]))
    
    fig,ax = plt.subplots(figsize=(12,12))
    sigdiff = -(len(databoth[(pidcut(bestscores))&(databoth["type"]==1)]))/len(databoth[databoth["type"]==1])*100
    bkgdiff = -(len(databoth[(pidcut(bestscores))&(databoth["type"]==0)]))/len(databoth[databoth["type"]==0])*100
    ax.hist(databoth["pho1_pPid_score"][databoth["type"]==1],bins=nbins,histtype="step",label=f"p1pid Signal")
    ax.hist(databoth["pho1_pPid_score"][databoth["type"]==0],bins=nbins,histtype="step",label=f"p1pid Bkg")
    ax.hist(databoth["pho2_pPid_score"][databoth["type"]==1],bins=nbins,histtype="step",label=f"p2pid Signal")
    ax.hist(databoth["pho2_pPid_score"][databoth["type"]==0],bins=nbins,histtype="step",label=f"p2pid Bkg")
    ax.axvline(x=p1pid[bestscores],color = "r", linestyle = "dashed", linewidth = 2, label = f"P1Pid cut: {p1pid[bestscores]:.3f}")
    ax.axvline(x=p2pid[bestscores],color = "k", linestyle = "dashed", linewidth = 2, label = f"P2Pid cut: {p2pid[bestscores]:.3f}")

    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score, Signal diff: {sigdiff:.3f}%, bkg diff: {bkgdiff:.3f}%", fontsize=20)
    fig.savefig(figure_dir+"pho_ppidscorecut")
    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 before matching backgrounds")
    databoth["type"][pidcut(bestscores)]= 2

    nboth0 = len(databoth[databoth["type"]==0])
    nboth1 = len(databoth[databoth["type"]==1])
    nboth2 = len(databoth[databoth["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after matching backgrounds")

    ##############################
    # Alternatively setting the PID to the same value
    ##############################
    print(f"Now setting pPid scores, so there is the same background in the peak area")
    searchval = np.linspace(-10,7.5,17500+1) #Set so there is 3 decimals certainty
    sigval, ival = [], []
    sig,bkg = [], []
    #The previous section was a grid, this time it is a line, where 17.5k values are determined.
    for i in searchval:
        sig.append(sum((databoth2["type"]==1)&(databoth2["pho1_pPid_score"]>i)&(databoth2["pho2_pPid_score"]>i)))
        bkg.append(sum((databoth2["type"]==0)&(databoth2["pho1_pPid_score"]>i)&(databoth2["pho2_pPid_score"]>i)))
        if  sum((databoth2["type"]==0)&(databoth2["pho1_pPid_score"]>i)&(databoth2["pho2_pPid_score"]>i)&(databoth2["invM"]<maxpeakinvm) & (databoth2["invM"]>minpeakinvm))== nATLASpeakbkg:
            sigval.append(sum((databoth2["type"]==1)&(databoth2["pho1_pPid_score"]>i)&(databoth2["pho2_pPid_score"]>i)))
            ival.append(i)
    fig,ax= plt.subplots(3,figsize=(12,12))
    ax[0].hist(sig,bins=100,histtype = "step",label="number of signal")
    ax[1].hist(bkg,bins=100,histtype = "step",label="number of background")
    ax[2].hist(sigval,bins=100,histtype = "step",label="elligible signal")
    ax[0].legend(loc='best',fontsize=20)
    ax[1].legend(loc='best',fontsize=20)
    ax[2].legend(loc='best',fontsize=20)

    ax[0].set_title(f"There is {nATLAS} ATLAS signal")
    ax[1].set_title(f"There is {nATLASbkg} ATLAS background")
    ax[2].set_title(f"Signal, with as much background as ATLAS peak")
    fig.savefig(figure_dir+"sig_and_bkg_forlinsearch")
    
    bestval = np.where(sigval == np.amax(sigval))[0]
    
    

    bestcut = ival[bestval[0]]
    #Applying cut and plotting.
    def ppidscore(cut):
        return ((databoth2["pho1_pPid_score"]<cut) | (databoth2["pho2_pPid_score"]<cut))
    
    fig,ax = plt.subplots(figsize=(12,12))
    sigdiff = -(len(databoth2[(ppidscore(bestcut))&(databoth2["type"]==1)]))/len(databoth2[databoth2["type"]==1])*100
    bkgdiff = -(len(databoth2[(ppidscore(bestcut))&(databoth2["type"]==0)]))/len(databoth2[databoth2["type"]==0])*100
    ax.hist(databoth2["pho1_pPid_score"][databoth2["type"]==1],bins=nbins,histtype="step",label=f"p1pid Signal")
    ax.hist(databoth2["pho1_pPid_score"][databoth2["type"]==0],bins=nbins,histtype="step",label=f"p1pid Bkg")
    ax.hist(databoth2["pho2_pPid_score"][databoth2["type"]==1],bins=nbins,histtype="step",label=f"p2pid Signal")
    ax.hist(databoth2["pho2_pPid_score"][databoth2["type"]==0],bins=nbins,histtype="step",label=f"p2pid Bkg")
    ax.axvline(x=bestcut,color = "r", linestyle = "dashed", linewidth = 2, label = f"pPid cut: {bestcut:.3f}")

    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("pho_pPid_score", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(f"Cut in pho_pPid_score, Signal diff: {sigdiff:.3f}%, bkg diff: {bkgdiff:.3f}%", fontsize=20)
    fig.savefig(figure_dir+"pho_sameppidscorecut")
    nboth0 = len(databoth2[databoth2["type"]==0])
    nboth1 = len(databoth2[databoth2["type"]==1])
    nboth2 = len(databoth2[databoth2["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 before matching backgrounds")
    databoth2["type"][ppidscore(bestcut)]= 2
    nboth0 = len(databoth2[databoth2["type"]==0])
    nboth1 = len(databoth2[databoth2["type"]==1])
    nboth2 = len(databoth2[databoth2["type"]==2])
    print(f"there is {nboth2} type 2, {nboth1} type 1 and {nboth0} type 0 after matching backgrounds")

    




#Change in signal
############################
# The different numbers and improvements
############################
if (args.PartType == "ele") or (args.PartType == "muo"):
    nboth = len(databoth[databoth["type"]==1])
    nbothpeak = len(databoth[(databoth["type"]==1)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
    nbothbkg = len(databoth[databoth["type"]==0])
    nbothpeakbkg = len(databoth[(databoth["type"]==0)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])

    nFML = len(dataFML[dataFML["type"]==1])
    nFMLpeak = len(dataFML[(dataFML["type"]==1)&(dataFML["invM"]<maxpeakinvm)&(dataFML["invM"]>minpeakinvm)])
    nFMLbkg = len(dataFML[dataFML["type"]==0])
    nFMLpeakbkg = len(dataFML[(dataFML["type"]==0)&(dataFML["invM"]<maxpeakinvm)&(dataFML["invM"]>minpeakinvm)])

    nLEP = len(DATALEP[DATALEP["type"]==1])
    nLEPpeak = len(DATALEP[(DATALEP["type"]==1)&(DATALEP["invM"]<maxpeakinvm)&(DATALEP["invM"]>minpeakinvm)])
    nLEPbkg = len(DATALEP[DATALEP["type"]==0])
    nLEPpeakbkg = len(DATALEP[(DATALEP["type"]==0)&(DATALEP["invM"]<maxpeakinvm)&(DATALEP["invM"]>minpeakinvm)])

    nPho = len(DATAPho[DATAPho["type"]==1])
    nPhopeak = len(DATAPho[(DATAPho["type"]==1)&(DATAPho["invM"]<maxpeakinvm)&(DATAPho["invM"]>minpeakinvm)])
    nPhobkg = len(DATAPho[DATAPho["type"]==0])
    nPhopeakbkg = len(DATAPho[(DATAPho["type"]==0)&(DATAPho["invM"]<maxpeakinvm)&(DATAPho["invM"]>minpeakinvm)])
elif args.PartType == "Data":
    nboth = len(databoth)
    nbothpeak = len(databoth[(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])

    nFML = len(dataFML)
    nFMLpeak = len(dataFML[(dataFML["invM"]<maxpeakinvm)&(dataFML["invM"]>minpeakinvm)])

    nLEP = len(DATALEP)
    nLEPpeak = len(DATALEP[(DATALEP["invM"]<maxpeakinvm)&(DATALEP["invM"]>minpeakinvm)])

    nPho = len(DATAPho)
    nPhopeak = len(DATAPho[(DATAPho["invM"]<maxpeakinvm)&(DATAPho["invM"]>minpeakinvm)])

    nboth2 = len(databoth2)
    nboth2peak = len(databoth2[(databoth2["invM"]<maxpeakinvm)&(databoth2["invM"]>minpeakinvm)])

    nFML2 = len(dataFML2)
    nFML2peak = len(dataFML2[(dataFML2["invM"]<maxpeakinvm)&(dataFML2["invM"]>minpeakinvm)])

    nLEP2 = len(DATALEP2)
    nLEP2peak = len(DATALEP2[(DATALEP2["invM"]<maxpeakinvm)&(DATALEP2["invM"]>minpeakinvm)])

    nPho2 = len(DATAPho2)
    nPho2peak = len(DATAPho2[(DATAPho2["invM"]<maxpeakinvm)&(DATAPho2["invM"]>minpeakinvm)])
elif args.PartType == "Hgg":
    nTruth = len(DATATruth[DATATruth["type"]==1])
    nTruthpeak = len(DATATruth[(DATATruth["type"]==1)&(DATATruth["invM"]<maxpeakinvm)&(DATATruth["invM"]>minpeakinvm)])
    nTruthbkg = len(DATATruth[DATATruth["type"]==0])
    nTruthpeakbkg = len(DATATruth[(DATATruth["type"]==0)&(DATATruth["invM"]<maxpeakinvm)&(DATATruth["invM"]>minpeakinvm)])

    nboth = len(databoth[databoth["type"]==1])
    nbothpeak = len(databoth[(databoth["type"]==1)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])
    nbothbkg = len(databoth[databoth["type"]==0])
    nbothpeakbkg = len(databoth[(databoth["type"]==0)&(databoth["invM"]<maxpeakinvm)&(databoth["invM"]>minpeakinvm)])

    nboth2 = len(databoth2[databoth2["type"]==1])
    nboth2peak = len(databoth2[(databoth2["type"]==1)&(databoth2["invM"]<maxpeakinvm)&(databoth2["invM"]>minpeakinvm)])
    nboth2bkg = len(databoth2[databoth2["type"]==0])
    nboth2peakbkg = len(databoth2[(databoth2["type"]==0)&(databoth2["invM"]<maxpeakinvm)&(databoth2["invM"]>minpeakinvm)])




#Calculating the improvement for the models, compare to ATLAS

if args.PartType !="Hgg":
    improvment = (nboth/nATLAS -1)*100
    improvmentPho = (nPho/nATLAS-1)*100
    improvmentLEP = (nLEP/nATLAS-1)*100
    improvFULL = (nFML/nATLAS-1)*100
    improvmentpeak = (nbothpeak/nATLASpeak -1)*100
    improvmentPhopeak = (nPhopeak/nATLASpeak-1)*100
    improvmentLEPpeak = (nLEPpeak/nATLASpeak-1)*100
    improvFULLpeak = (nFMLpeak/nATLASpeak-1)*100
    if args.PartType == "Data":
        improvment2 = (nboth2/nATLAS -1)*100
        improvmentPho2 = (nPho2/nATLAS-1)*100
        improvmentLEP2 = (nLEP2/nATLAS-1)*100
        improvFULL2 = (nFML2/nATLAS-1)*100
        improvmentpeak2 = (nboth2peak/nATLASpeak -1)*100
        improvmentPhopeak2 = (nPho2peak/nATLASpeak-1)*100
        improvmentLEPpeak2 = (nLEP2peak/nATLASpeak-1)*100
        improvFULLpeak2 = (nFML2peak/nATLASpeak-1)*100
else: 
    improvmentTruth = (nTruth/nATLAS-1)*100
    improvmentTruthpeak = (nTruthpeak/nATLASpeak-1)*100

    improvmentboth = (nboth/nATLAS-1)*100
    improvmentbothpeak = (nbothpeak/nATLASpeak-1)*100

    improvmentboth2 = (nboth2/nATLAS-1)*100
    improvmentboth2peak = (nboth2peak/nATLASpeak-1)*100
    



#Change in backgrounds

if (args.PartType == "ele") or (args.PartType == "muo"):
    improvmentbkg = (nbothbkg/nATLASbkg -1)*100
    improvmentPhobkg = (nPhobkg/nATLASbkg-1)*100
    improvmentLEPbkg = (nLEPbkg/nATLASbkg-1)*100
    improvFULLbkg = (nFMLbkg/nATLASbkg-1)*100
    improvmentpeakbkg = (nbothpeakbkg/nATLASpeakbkg -1)*100
    improvmentPhopeakbkg = (nPhopeakbkg/nATLASpeakbkg-1)*100
    improvmentLEPpeakbkg = (nLEPpeakbkg/nATLASpeakbkg-1)*100
    improvFULLpeakbkg = (nFMLpeakbkg/nATLASpeakbkg-1)*100
elif args.PartType == "Hgg": 
    improvmentTruthbkg = (nTruthbkg/nATLASbkg-1)*100
    improvmentTruthpeakbkg = (nTruthpeakbkg/nATLASpeakbkg-1)*100

    improvmentbothbkg = (nbothbkg/nATLASbkg-1)*100
    improvmentbothpeakbkg = (nbothpeakbkg/nATLASpeakbkg-1)*100
    
    improvmentboth2bkg = (nboth2bkg/nATLASbkg-1)*100
    improvmentboth2peakbkg = (nboth2peakbkg/nATLASpeakbkg-1)*100


#Printing in the log how the improvements were.

print(f"There are {nATLAS} signal in the ATLAS file.")
print(f"There are {nATLASpeak} signal in the ATLAS file, in the range {minpeakinvm}-{maxpeakinvm}GeV")
if args.PartType !="Hgg":
    print("Using same cuts as MC")
    print(f"With both lepton and photon model there are {nboth}. Improvement of:                                {improvment:.1f}%")
    print(f"With both lepton and photon model there are {nbothpeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentpeak:.1f}%")
    print(f"With my photon model there are {nPho}. Improvement of:                                             {improvmentPho:.1f}%")
    print(f"With my photon model there are {nPhopeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentPhopeak:.1f}%")
    print(f"With my lepton model there are {nLEP}. Improvement of:                                             {improvmentLEP:.1f}%")
    print(f"With my lepton model there are {nLEPpeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentLEPpeak:.1f}%")
    print(f"With the full ML model there are {nFML}. Improvement of:                                            {improvFULL:.1f}%")
    print(f"With the full ML model there are {nFMLpeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                {improvFULLpeak:.1f}%")
    if args.PartType == "Data":
        print("Calculates the value by making sure the 'outer' part has the same amount of signal - assumed background.") 
        print(f"With both lepton and photon model there are {nboth2}. Improvement of:                                {improvment2:.1f}%")
        print(f"With both lepton and photon model there are {nboth2peak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentpeak2:.1f}%")
        print(f"With my photon model there are {nPho2}. Improvement of:                                             {improvmentPho2:.1f}%")
        print(f"With my photon model there are {nPho2peak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentPhopeak2:.1f}%")
        print(f"With my lepton model there are {nLEP2}. Improvement of:                                             {improvmentLEP2:.1f}%")
        print(f"With my lepton model there are {nLEP2peak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentLEPpeak2:.1f}%")
        print(f"With the full ML model there are {nFML2}. Improvement of:                                            {improvFULL2:.1f}%")
        print(f"With the full ML model there are {nFML2peak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                {improvFULLpeak2:.1f}%")
elif args.PartType =="Hgg":
    print(f"With the Higgs to two photons model there are {nTruth}. Improvement of:                                {improvmentTruth:.1f}%")
    print(f"With the Higgs to two photons model there are {nTruthpeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentTruthpeak:.1f}%")
    print(f"With the single photon models there are {nboth}. Improvement of:                                {improvmentboth:.1f}%")
    print(f"With the single photon models there are {nbothpeak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentbothpeak:.1f}%")
    print(f"With the single photon models using same ppid score there are {nboth2}. Improvement of:                                {improvmentboth2:.1f}%")
    print(f"With the single photon models using same ppid score there are {nboth2peak}, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentboth2peak:.1f}%")
print("")
print("")
if args.PartType != "Data":
    print(f"There are {nATLASbkg} background in the ATLAS file.")
    print(f"There are {nATLASpeakbkg} background in the ATLAS file, in the range {minpeakinvm}-{maxpeakinvm}GeV")
if (args.PartType == "ele") or (args.PartType == "muo"):
    print(f"With both lepton and photon model there are {nbothbkg} background. Improvement of:                                {improvmentbkg:.1f}%")
    print(f"With both lepton and photon model there are {nbothpeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:    {improvmentpeakbkg:.1f}%")
    print(f"With my photon model there are {nPhobkg} background. Improvement of:                                             {improvmentPhobkg:.1f}%")
    print(f"With my photon model there are {nPhopeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentPhopeakbkg:.1f}%")
    print(f"With my lepton model there are {nLEPbkg} background. Improvement of:                                             {improvmentLEPbkg:.1f}%")
    print(f"With my lepton model there are {nLEPpeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentLEPpeakbkg:.1f}%")
    print(f"With the full ML model there are {nFMLbkg} background. Improvement of:                                            {improvFULLbkg:.1f}%")
    print(f"With the full ML model there are {nFMLpeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                {improvFULLpeakbkg:.1f}%")
elif args.PartType == "Hgg":
    print(f"With the Higgs to two photons model there are {nTruthbkg} background. Improvement of:                                             {improvmentTruthbkg:.1f}%")
    print(f"With the Higgs to two photons model there are {nTruthpeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentTruthpeakbkg:.1f}%")
    print(f"With the single photon models  there are {nbothbkg} background. Improvement of:                                             {improvmentbothbkg:.1f}%")
    print(f"With the single photon models there are {nbothpeakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentbothpeakbkg:.1f}%")
    print(f"With the single photon models using same ppid score there are {nboth2bkg} background. Improvement of:                                             {improvmentboth2bkg:.1f}%")
    print(f"With the single photon models using same ppid score there are {nboth2peakbkg} background, in the range {minpeakinvm}-{maxpeakinvm}GeV. Improvement of:                 {improvmentboth2peakbkg:.1f}%")


#Plotting the final distributions.

nbins = np.linspace(mininvm,maxinvm,maxinvm-mininvm+1)
print("Plotting the invM to compare")
fig, ax = plt.subplots(figsize=(12,12))
if args.PartType != "Data":
    ax.hist(dataATLAS["invM"][dataATLAS["type"]==1],bins=nbins, histtype ="step",label = "Atlas signal")
if (args.PartType == "ele") or (args.PartType == "muo"):
    #ax.hist(dataeMgA["invM"][Sigemga],bins=nbins,histtype="step",label=f"{args.PartType} pair model")
    #ax.hist(dataeAgM["invM"][Sigeagm],bins=nbins,histtype="step",label=f"photon model")
    ax.hist(dataFML["invM"][dataFML["type"]==1], bins= nbins, histtype= "step", label=f"Full ML model")
    ax.hist(databoth["invM"][databoth["type"]==1],bins=nbins,histtype="step",label=f"both models")
    ax.hist(DATALEP["invM"][DATALEP["type"]==1], bins= nbins, histtype= "step", label=f"{args.PartType} pair model")
    ax.hist(DATAPho["invM"][DATAPho["type"]==1],bins=nbins,histtype="step",label=f"photon model")
elif args.PartType == "Data":
    ax.hist(dataATLAS["invM"],bins=nbins, histtype ="step",label = "Atlas signal")
    ax.hist(dataFML["invM"], bins= nbins, histtype= "step", label=f"Full ML model")
    ax.hist(databoth["invM"],bins=nbins,histtype="step",label=f"both models")
    ax.hist(DATALEP["invM"], bins= nbins, histtype= "step", label=f"muon pair model")
    ax.hist(DATAPho["invM"],bins=nbins,histtype="step",label=f"photon model")
elif args.PartType == "Hgg":
    ax.hist(DATATruth["invM"][DATATruth["type"]==1], bins = nbins, histtype = "step", label=r"$H\rightarrow \gamma\gamma$ model")
    ax.hist(databoth["invM"][databoth["type"]==1], bins = nbins, histtype = "step", label=r"Different pPid scores")
    ax.hist(databoth2["invM"][databoth2["type"]==1], bins = nbins, histtype = "step", label=r"Same pPid scores")

ax.legend(loc='best', fontsize=20)
ax.set_xlabel("invM", fontsize=20)
ax.set_ylabel("Frequency", fontsize=20)
if args.Data==1:
    plottype ="Data"
elif args.Data==0:
    plottype = "MC"
if args.PartType == "ele":
    ax.set_title(r"Signal for models and ATLAS cuts, in MC $Z\rightarrow ee\gamma$ "+plottype+".", fontsize=20)
elif args.PartType =="muo":
    ax.set_title(r"Signal for models and ATLAS cuts, in MC $Z\rightarrow \mu\mu\gamma$ "+plottype+".", fontsize=20)
elif args.PartType == "Data":
    ax.set_title(r"Signal, using cuts from MC, in Data $Z\rightarrow \mu\mu\gamma$.", fontsize=20)
elif args.PartType == "Hgg":
    ax.set_title(r"Signal for $H\rightarrow \gamma\gamma$ models and ATLAS selection, in "+plottype+".", fontsize=20)
fig.savefig(figure_dir+"_signal_invM_comparison")
plt.close()

if args.PartType != "Data":
    fig, ax = plt.subplots(figsize=(12,12))
    #ax.hist(dataLGBM["invM"][SIGLGBMbkg],bins=nbins,histtype="step",label=f"LGBM signal")
    ax.hist(dataATLAS["invM"][dataATLAS["type"]==0],bins=nbins, histtype ="step",label = "Atlas background")
    if args.PartType !="Hgg":
        #ax.hist(dataeMgA["invM"][Sigemgabkg],bins=nbins,histtype="step",label=f"{args.PartType} pair model")
        #ax.hist(dataeAgM["invM"][Sigeagmbkg],bins=nbins,histtype="step",label=f"photon model")
        ax.hist(dataFML["invM"][dataFML["type"]==0], bins= nbins, histtype= "step", label=f"Full ML model")
        ax.hist(databoth["invM"][databoth["type"]==0],bins=nbins,histtype="step",label=f"both models")
        ax.hist(DATALEP["invM"][DATALEP["type"]==0], bins= nbins, histtype= "step", label=f"{args.PartType} pair model")
        ax.hist(DATAPho["invM"][DATAPho["type"]==0],bins=nbins,histtype="step",label=f"photon model")
    elif args.PartType =="Hgg":
        ax.hist(DATATruth["invM"][DATATruth["type"]==0], bins = nbins, histtype = "step", label=r"$H\rightarrow \gamma\gamma$ model")
        ax.hist(databoth["invM"][databoth["type"]==0], bins = nbins, histtype = "step", label=r"$Different pPid scores")
        ax.hist(databoth2["invM"][databoth2["type"]==0], bins = nbins, histtype = "step", label=r"$Same pPid scores")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("invM", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    if args.Data==1:
        plottype ="Data"
    elif args.Data==0:
        plottype = "MC"
    if args.PartType == "ele":
        ax.set_title(r"Bakground for models and ATLAS cuts, in MC $Z\rightarrow ee\gamma$ "+plottype+".", fontsize=20)
    elif args.PartType =="muo":
        ax.set_title(r"Background for models and ATLAS cuts, in MC $Z\rightarrow\mu\mu\gamma$ "+plottype+".", fontsize=20)
    elif args.PartType == "Hgg":
        ax.set_title(r"Background for $H\rightarrow \gamma\gamma$ models and ATLAS selection, in "+plottype+".", fontsize=20)
    fig.savefig(figure_dir+"_background_invM_comparison")
    plt.close()
elif args.PartType == "Data":
    fig, ax = plt.subplots(figsize=(12,12))
    ax.hist(dataATLAS["invM"],bins=nbins, histtype ="step",label = "Atlas signal")
    ax.hist(dataFML2["invM"], bins= nbins, histtype= "step", label=f"Full ML model")
    ax.hist(databoth2["invM"],bins=nbins,histtype="step",label=f"both models")
    ax.hist(DATALEP2["invM"], bins= nbins, histtype= "step", label=f"muon pair model")
    ax.hist(DATAPho2["invM"],bins=nbins,histtype="step",label=f"photon model")
    ax.legend(loc='best', fontsize=20)
    ax.set_xlabel("invM", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title(r"Signal for models and ATLAS cuts, in Data $Z\rightarrow \mu\mu\gamma$.", fontsize=20)
    fig.savefig(figure_dir+"_own_selection_invM_comparison")
    plt.close()

sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
print("Program done.")