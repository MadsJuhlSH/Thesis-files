#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wednesday 6 Jan
@author: Sara Dahl Andersen

Running my Z model for muons and outputting the score

nohup python -u muoGamPredict.py --tag 20210122 output/ZFilesWithVars/20210122_2/combined_20210121.h5 --model ../Zmmg/output/ZModels/20210103_85GeV/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210122_2 output/ZFilesWithVars/20210122/combined_20201112.h5 --model ../Zmmg/output/ZModels/20201222_40-80GeV/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210127 output/ZFilesWithVars/20210127/combined_20210126.h5 --model ../Zmmg/output/ZModels/20210103_85GeV/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210205 output/ZFilesWithVars/20210127/combined_20210126.h5 --model ../Zmmg/output/ZModels/20210205_2/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210205_onlyZmmg output/ZFilesWithVars/20210127/combined_20210126.h5 --model ../Zmmg/output/ZModels/20210205_onlyZmmg/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210205_onlyZmmBkg output/ZFilesWithVars/20210127/combined_20210126.h5 --model ../Zmmg/output/ZModels/20210205_onlyZmmBkg/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210209_onlyZmmBkg_chargeCut output/ZFilesWithVars/20210209_chargeCut/combined_20210126.h5 --model ../Zmmg/output/ZModels/20210205_onlyZmmBkg/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210212 output/ZFilesWithVars/20210212/combined_2021021220210212.h5 --model ../Zmmg/output/ZModels/20210212/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210212_2 output/ZFilesWithVars/20210212_2/combined_2021021220210212.h5 --model ../Zmmg/output/ZModels/20210212_2/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210216 output/ZFilesWithVars/20210212_2/combined_2021021220210212.h5 --model ../Zmmg/output/ZModels/20210216/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210216_noZmm output/ZFilesWithVars/20210212_2/combined_2021021220210212.h5 --model ../Zmmg/output/ZModels/20210216_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210216_noZmm2 output/ZFilesWithVars/20210212_2/combined_2021021220210212.h5 --model ../Zmmg/output/ZModels/20210216_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210216 output/ZFilesWithVars/20210216/combined_20210216.h5 --model ../Zmmg/output/ZModels/20210216/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210216_noZmm output/ZFilesWithVars/20210216/combined_20210216.h5 --model ../Zmmg/output/ZModels/20210216_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210219_noZmm output/ZFilesWithVars/20210216/combined_20210216.h5 --model ../Zmmg/output/ZModels/20210216_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210311_noZmm output/ZFilesWithVars/20210216/combined_20210216.h5 --model ../Zmmg/output/ZModels/20210216_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210312_noZmm output/ZFilesWithVars/20210312/combined_20210312.h5 --model ../Zmmg/output/ZModels/20210312_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210316_noZmm_test output/ZFilesWithVars/20210312/combined_20210312.h5 --model ../Zmmg/output/ZModels/20210312_noZmm/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown


nohup python -u muoGamPredict.py --tag 20210401 output/ZFilesWithVars/20210401/combined_20210312.h5 --model ../Zmmg/output/ZModels/20210401/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210401_opt output/ZFilesWithVars/20210401/combined_20210312.h5 --model ../Zmmg/output/ZModels/20210401_opt/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210401_opt output/ZFilesWithVars/20210401/combined_20210312.h5 --model ../Zmmg/output/ZModels/20210401_opt/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210402_opt2 output/ZFilesWithVars/20210402/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210401_opt/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown


nohup python -u muoGamPredict.py --tag 20210406_wBkg output/ZFilesWithVars/20210406/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210406/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210406_wBkg_wPtvarconeCut output/ZFilesWithVars/20210406/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210406/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210414 output/ZFilesWithVars/20210414_40_83/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210414/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210414_opt output/ZFilesWithVars/20210414_40_83/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210414_opt/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210415_invM_40_83_phoet_10 output/ZFilesWithVars/20210415_invM_40_83_phoet_10/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210415_invM_83_phoet_10 output/ZFilesWithVars/20210415_invM_83_phoet_10/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210415_invM_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210415_invM_40_83_phoet_10_wZ output/ZFilesWithVars/20210415_invM_40_83_phoet_10/combined_20210402.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210420_invM_40_83_phoet_10_wZ output/ZFilesWithVars/20210420_invM_40_83_phoet_10/combined_20210419.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210422_invM_40_83_phoet_10 output/ZFilesWithVars/20210420_invM_40_83_phoet_10/combined_20210419.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown

nohup python -u muoGamPredict.py --tag 20210426 output/ZFilesWithVars/20210420_invM_40_83_phoet_10/combined_20210419.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210427 output/ZFilesWithVars/20210420_invM_40_83_phoet_10/combined_20210419.h5 --model ../Zmmg/output/ZModels/20210415_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown
nohup python -u muoGamPredict.py --tag 20210430_newZ output/ZFilesWithVars/20210430_invM_40_83_phoet_10_newZ/combined_20210419.h5 --model ../Zmmg/output/ZModels/20210430_invM_40_83_phoet_10/lgbmZmmg.txt 2>&1 &> output/logZPredict.txt & disown


"""
print("Program running...")

import os
import argparse
import logging as log

from time import time
from datetime import timedelta
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import randint

from utils import mkdir
from hep_ml.reweight import GBReweighter

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shap
import lightgbm as lgb
from Zee_functions_HKLE import h5ToDf, accuracy, HyperOpt_RandSearch, auc_eval, HeatMap_rand, PlotRandomSearch
from scipy.special import logit, expit
from peakfit import PeakFit_likelihood

plt.style.use("normalfig")


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/Datafitting/", type=str,
                    help='Output directory.')
parser.add_argument('path', type=str, nargs='+',
                    help='HDF5 file(s) to use for Z model.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--model', action='store', type=str,
                    help='Trained LGBM model to predict from')


args = parser.parse_args()

# Validate arguments
if not args.path:
    log.error("No HDF5 file was specified.")
    quit()

if args.njobs > 20:
    log.error("The requested number of jobs ({}) is excessive (>20). Exiting.".format(args.njobs))
    quit()


# Make and set the output directory to tag, if it doesn't already exist
# Will stop if the output already exists since re-running is either not needed or unwanted
# If it's wanted, delete the output first yourself
args.outdir = args.outdir+args.tag+f"/"
if os.path.exists(args.outdir):
    log.error(f"Output already exists - please remove yourself. Output: {args.outdir}")
    quit()
else:
    log.info(f"Creating output folder: {args.outdir}")
    mkdir(args.outdir)

# File number counter (incremented first in loop)
counter = -1

# ================================================ #
#                   Functions                      #
# ================================================ #

def h5ToDf(filename):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    log.info(f"Import data from: {filename}")
    with h5py.File(filename, "r") as hf :
        d = {}
        for name in list(hf.keys()):
            d[name] = np.array(hf[name][:])
        df = pd.DataFrame(data=d)
    return(df)

def mask_LGBM(data, sigSel):
    return (logit(data["predLGBM"]) > sigSel)

def getSameBkg(data, maskFunction, selStart):
    Sel = selStart
    i = 0

    nBkgATLAS = len(data[(data["isATLAS"] == 1) & (data["invM"] > 110)])
    nSigATLAS = len(data[(data["isATLAS"] == 1) & (data["invM"] >  91.2-10) & (data["invM"] <  91.2+10)])

    print(f"    ATLAS selection:  nBkg = {nBkgATLAS}, nSig = {nSigATLAS}")

    nBkgCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 110)])
    nSigCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    print(f"   Initial selection data at selection {selStart}:    nBkg = {nBkgCompare}, nSig = {nSigCompare}")


    while nBkgCompare > nBkgATLAS:
        # Increase selection
        Sel = Sel + 0.01
        nBkgBefore = nBkgCompare

        nBkgCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 110)])
        nSigCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

        i += 1
        if (i % 100) == 0:
            print(f"After {i} iterations with selection = {Sel}:")
            print(f"    Selection:         nBkg = {nBkgCompare}, nSig = {nSigCompare}")
        if i > 300 and nBkgCompare == nBkgBefore:
            break

    Sel = Sel - 0.01 #get prediction right before we are below ATLAS

    nBkg = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 110)])
    nSig = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    print(f"    Final selection data:    nBkg = {nBkg}, nSig = {nSig}")
    print(f"    Final selection: {Sel}\n")
    print(f"    Yielding increase in signal of {np.round(((nSig-nSigATLAS)/nSigATLAS)*100,2)} %")

    # Sel = round(Sel,decimals)

    if Sel == selStart:
        # Check if signal selection had no effect
        print("    Initial selection too high... Exiting.")
        quit()

    return Sel, nBkg, nBkgATLAS


def getSameSS(data, maskFunction, selStart):
    Sel = selStart
    i = 0

    nBkgATLAS = len(data[(data["bkgATLAS"] == 1) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
    nSigATLAS = len(data[(data["isATLAS"] == 1) & (data["invM"] >  91.2-10) & (data["invM"] <  91.2+10)])

    print(f"    ATLAS selection:  nBkg = {nBkgATLAS}, nSig = {nSigATLAS}")

    nBkgCompare = len(data[(logit(data["predLGBM"]) > Sel) & ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
    nSigCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    print(f"   Initial selection data at selection {selStart}:    nBkg = {nBkgCompare}, nSig = {nSigCompare}")


    while nBkgCompare > nBkgATLAS:
        # Increase selection
        Sel = Sel + 0.05
        nBkgBefore = nBkgCompare

        nBkgCompare = len(data[(logit(data["predLGBM"]) > Sel) & ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
        nSigCompare = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

        i += 1
        if (i % 100) == 0:
            print(f"After {i} iterations with selection = {Sel}:")
            print(f"    Selection:         nBkg = {nBkgCompare}, nSig = {nSigCompare}")
        if i > 300 and nBkgCompare == nBkgBefore:
            break

    Sel = Sel - 0.05 #get prediction right before we are below ATLAS

    nBkg = len(data[(logit(data["predLGBM"]) > Sel) & ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40))) ])
    nSig = len(data[(logit(data["predLGBM"]) > Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    print(f"    Final selection data:    nBkg = {nBkg}, nSig = {nSig}")
    print(f"    Final selection: {Sel}\n")
    print(f"    Yielding increase in signal of {np.round(((nSig-nSigATLAS)/nSigATLAS)*100,2)} %")
    print(f"    Yielding significance for LGBM: {np.round(nSig/np.sqrt(nSig+nBkg),4)}, ATLAS: {np.round(nSigATLAS/np.sqrt(nSigATLAS+nBkgATLAS),4)}")
    significanceLGBM = nSig/np.sqrt(nSig+nBkg)
    significanceATLAS = nSigATLAS/np.sqrt(nSigATLAS+nBkgATLAS)
    # Sel = round(Sel,decimals)

    if Sel == selStart:
        # Check if signal selection had no effect
        print("    Initial selection too high... Exiting.")
        quit()

    return Sel, nBkg, nBkgATLAS, nSig, nSigATLAS


def GetFit(data, cutval_min, cutval_max):
    bkgs = []
    print("Getting the value for the fit")
    print("\n")
    ### First up we get the BL bkg
    No_cut = (logit(data["predLGBM"]) > -50)*1
    BL_sig, BL_bkg, f_bkg, chi2 = PeakFit_likelihood(No_cut, data["invM"], "No cut", args.outdir, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);

    ATLAS_cut = (data["isATLAS"])*1
    ATLAS_sig, ATLAS_bkg, f_bkgATLAS, chi2 = PeakFit_likelihood(ATLAS_cut, data["invM"], "ATLAS cut", args.outdir, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);


    data = data[(data["muo1_charge"]*data["muo2_charge"]) == -1]
    cuts = np.linspace(cutval_min, cutval_max, num = 50)
    for cutval in cuts:
        Likelihood_cut = (logit(data["predLGBM"]) > cutval)*1
        sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], "Fpr cut", args.outdir, plots = True, constant_mean = True,
                                               constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                               bkg_exp = True, bkg_cheb = False);
        bkg_ratio = bkg/BL_bkg
        # bkgs.append(bkg_ratio)
        bkgs.append(f_bkg)
        #if (bkg_ratio < ATLAS_bkg/BL_bkg + 0.0001) & (bkg_ratio > ATLAS_bkg/BL_bkg - 0.0001):
        if (((f_bkg < f_bkgATLAS + 0.0001) & (f_bkg > f_bkgATLAS - 0.0001)) | (f_bkg < f_bkgATLAS)):
                    #print(f"Found correct bkg efficiency for a ATLAS cut at {bkg_ratio} compared to true {ATLAS_bkg/BL_bkg}")
                    print(f"Found correct bkg efficiency for a ATLAS cut at {f_bkg} compared to true {f_bkgATLAS}")
                    print(f"My cut is {cutval}")
                    print(f"Breaking....")
                    print("\n")
                    break
    if cutval == cutval_max:
        print(f"The cutval corresponds to the maximum cutval, so no match was found...")
        print(f"The background ratio was {f_bkg}, compared to ATLAS {f_bkgATLAS}")
        print("The background ratios was:")
        print(bkgs)
        print("\n")
    return cutval, sig, bkg, ATLAS_sig, ATLAS_bkg

def GetFitBkg(data, cutval_min, cutval_max):
    bkgs = []
    print("Getting the value for the fit")
    print("\n")
    ### First up we get the BL bkg
    No_cut = (logit(data["predLGBM"]) > -50)*1
    BL_sig, BL_bkg, f_bkg, chi2 = PeakFit_likelihood(No_cut, data["invM"], "No cut", args.outdir, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);

    ATLAS_cut = (data["isATLAS"])*1
    ATLAS_sig, ATLAS_bkg, f_bkgATLAS, chi2 = PeakFit_likelihood(ATLAS_cut, data["invM"], "ATLAS cut", args.outdir, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);


    data = data[(data["muo1_charge"]*data["muo2_charge"]) == -1]
    cuts = np.linspace(cutval_min, cutval_max, num = 50)
    for cutval in cuts:
        Likelihood_cut = (logit(data["predLGBM"]) > cutval)*1
        sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], "Fpr cut bkg", args.outdir, plots = True, constant_mean = True,
                                               constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                               bkg_exp = True, bkg_cheb = False);
        bkg_ratio = bkg/BL_bkg
        # bkgs.append(bkg_ratio)
        bkgs.append(bkg)
        #if (bkg_ratio < ATLAS_bkg/BL_bkg + 0.0001) & (bkg_ratio > ATLAS_bkg/BL_bkg - 0.0001):
        if (bkg < ATLAS_bkg + 10) & (bkg > ATLAS_bkg - 10):
                    #print(f"Found correct bkg efficiency for a ATLAS cut at {bkg_ratio} compared to true {ATLAS_bkg/BL_bkg}")
                    print(f"Found correct bkg for cut, with {bkg} compared to true {ATLAS_bkg}")
                    print(f"My cut is {cutval}")
                    print(f"Breaking....")
                    print("\n")
                    break
    if cutval == cutval_max:
        print(f"The cutval corresponds to the maximum cutval, so no match was found...")
        print(f"The background was {bkg}, compared to ATLAS {ATLAS_bkg}")
        print("The background ratios was:")
        print(bkgs)
        print("\n")
    return cutval, sig, bkg, ATLAS_sig, ATLAS_bkg

def GetMLCutFit(data, maskFunction, cutval_min, cutval_max):
    bkgs = []
    print("Getting the value for the fit")
    print("\n")
    ATLAS_cut = (data["isATLAS"])*1
    ATLAS_sig, ATLAS_bkg, f_bkgATLAS, chi2 = PeakFit_likelihood(ATLAS_cut, data["invM"], "ATLAS cut", args.outdir, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);


    data = data[(data["muo1_charge"]*data["muo2_charge"]) == -1]
    cuts = np.linspace(cutval_min, cutval_max, num = 50)
    for cutval in cuts:
        Likelihood_cut = maskFunction(data, cutval)*1
        sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], "ML cut", args.outdir, plots = True, constant_mean = True,
                                               constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                               bkg_exp = True, bkg_cheb = False);
        bkgs.append(f_bkg)
        #if (bkg_ratio < ATLAS_bkg/BL_bkg + 0.0001) & (bkg_ratio > ATLAS_bkg/BL_bkg - 0.0001):
        if (f_bkg < f_bkgATLAS + 0.005) & (f_bkg > f_bkgATLAS - 0.005):
                    print(f"Found correct bkg efficiency for a ATLAS cut at {f_bkg} compared to true {f_bkgATLAS}")
                    print(f"My cut is {cutval}")
                    print(f"Breaking....")
                    print("\n")
                    break
    if cutval == cutval_max:
        print(f"The cutval corresponds to the maximum cutval, so no match was found...")
        print(f"The background ratio was {f_bkg}, compared to ATLAS {f_bkgATLAS}")
        print("The background ratios was:")
        print(bkgs)
        print("\n")
    return cutval


def getMLcut(data, maskFunction, selStart, decimals=4):
    # Number of background pairs in ATLAS selection
    # nBkgATLAS = np.sum(data[(data[truth_var]==0)]['isATLAS'])
    # nSigATLAS = np.sum(data[(data[truth_var]==1)]['isATLAS'])
    nBkgATLAS = len(data[(data["isATLAS"] == 1) & (data["invM"] > 110)])
    nSigATLAS = len(data[(data["isATLAS"] == 1) & (data["invM"] >  91.2-10) & (data["invM"] <  91.2+10)])

    print(f"    ATLAS selection: nBkg = {nBkgATLAS}, nSig = {nSigATLAS}")

    # Initiate signal selection
    Sel = selStart
    i = 0

    nBkgCompare = len(data[maskFunction(data, Sel) & (data["invM"] > 110)])
    nSigCompare = len(data[maskFunction(data, Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    # nBkgCompare = np.sum( ( (data[truth_var]==0) & maskFunction(data, Sel) ) )
    # nSigCompare = np.sum( ( (data[truth_var]==1) & maskFunction(data, Sel) ) )
    print(f"    Selection (valid):         nBkg = {nBkgCompare}, nSig = {nSigCompare}")

    # Find signal selection
    while nBkgCompare > nBkgATLAS:
        # Increase selection
        Sel = Sel + 10**(-decimals)
        nBkgBefore = nBkgCompare

        nBkgCompare = len(data[maskFunction(data, Sel) & (data["invM"] > 110)])
        nSigCompare = len(data[maskFunction(data, Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

        # nBkgCompare = np.sum( (  (data[truth_var]==0) & maskFunction(data, Sel) ) )
        # nSigCompare = np.sum( (  (data[truth_var]==1) & maskFunction(data, Sel) ) )

        i += 1
        if (i % 100) == 0:
            print(f"After {i} iterations with selection = {Sel}:")
            print(f"    Selection (valid):         nBkg = {nBkgCompare}, nSig = {nSigCompare}")
        if i > 400 and nBkgCompare == nBkgBefore:
            break

    Sel = Sel - 10**(-decimals) #get prediction right before we are below ATLAS

    # nBkg = np.sum( ( (data[truth_var]==0) & maskFunction(data, Sel) ) )
    # nSig = np.sum( ( (data[truth_var]==1) & maskFunction(data, Sel) ) )
    nBkg = len(data[maskFunction(data, Sel) & (data["invM"] > 110)])
    nSig = len(data[maskFunction(data, Sel) & (data["invM"] > 91.2-10) & (data["invM"] < 91.2+10)])

    print(f"    Final selection (valid):    nBkg = {nBkg}, nSig = {nSig}")
    #fpr = {fprSelCompare}, tpr = {tprSel[1]}
    print(f"    Final selection: {Sel}\n")
    Sel = round(Sel,decimals)

    return Sel

def GetATLASBkg(data):
    return ( ( data['muo1_charge']*data['muo2_charge'] > 0 ) & #same sign
             ( (data['muo1_pt']) > 10 ) &
             ( (data['muo2_pt']) > 10 ) &
             ( (np.abs( data['muo1_eta']) < 2.7) ) &
             ( (np.abs( data['muo2_eta']) < 2.7) ) &
             ( data['muo1_LHMedium'] * data['muo2_LHMedium'] ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo1_delta_z0_sin_theta']) < 0.5 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_delta_z0_sin_theta']) < 0.5 ) &
             ( data['muo1_ptvarcone20'] < ptvarcone20Cut ) &
             ( data['muo2_ptvarcone20'] < ptvarcone20Cut ) &
             ### photon cuts
             ( abs(data['pho_et']) > 10 ) &
             ( (np.abs( data['pho_eta'] ) < 2.37) ) &
             ( (np.abs( data['pho_eta'] ) < 1.37) | ((np.abs( data['pho_eta'] ) > 1.52) & (np.abs( data['pho_eta'] ) < 2.37))) &
             ( data['pho_isPhotonEMTight'] )
             )
def GetATLASCut(data):
    return ( ( np.sign(data['muo1_charge'])*np.sign(data['muo2_charge']) < 0 ) & #opposite sign
             ( (data['muo1_pt']) > 10 ) &
             ( (data['muo2_pt']) > 10 ) &
             ( (np.abs( data['muo1_eta']) < 2.7) ) &
             ( (np.abs( data['muo2_eta']) < 2.7) ) &
             ( data['muo1_LHMedium'] * data['muo2_LHMedium'] ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo1_delta_z0_sin_theta']) < 0.5 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_delta_z0_sin_theta']) < 0.5 ) &
             ( data['muo1_ptvarcone20'] < ptvarcone20Cut ) &
             ( data['muo2_ptvarcone20'] < ptvarcone20Cut ) &
             ### photon cuts
             ( abs(data['pho_et']) > 10 ) &
             ( (np.abs( data['pho_eta'] ) < 2.37) ) &
             ( (np.abs( data['pho_eta'] ) < 1.37) | ((np.abs( data['pho_eta'] ) > 1.52) & (np.abs( data['pho_eta'] ) < 2.37))) &
             ( data['pho_isPhotonEMTight'] )
             )

def GetZCut(data, Sel):
    return ( ( data['Z_score'] > Sel ) &
             ( abs(data['pho_et']) > 10 ) &
             # ( (np.abs( data['pho_eta']) < 2.7)) &
             ( (np.abs( data['pho_eta']) < 2.37)) &
             ( (np.abs( data['pho_eta'] ) < 1.37) | ((np.abs( data['pho_eta'] ) > 1.52) & (np.abs( data['pho_eta'] ) < 2.37))) &
             ( data['pho_isPhotonEMTight'] )
             )

def GetPhoCut(data, Sel):
    return ( ( np.sign(data['muo1_charge'])*np.sign(data['muo2_charge']) < 0 ) & #opposite sign
             ( (data['muo1_pt']) > 10 ) &
             ( (data['muo2_pt']) > 10 ) &
             ( (np.abs( data['muo1_eta']) < 2.7) ) &
             ( (np.abs( data['muo2_eta']) < 2.7) ) &
             ( data['muo1_LHMedium'] * data['muo2_LHMedium'] ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo1_delta_z0_sin_theta']) < 0.5 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_delta_z0_sin_theta']) < 0.5 ) &
             ( data['muo1_ptvarcone20'] < ptvarcone20Cut ) &
             ( data['muo2_ptvarcone20'] < ptvarcone20Cut ) &
             ### photon cuts
             ( data['pho_PID_score'] > Sel ) &
             ( data['pho_isPhotonEMTight'] )
             )
# ================================================ #
#                End of functions                  #
# ================================================ #


# Data
data_get = h5ToDf(args.path[0])

# data_get = data_get[data_get["invM_mm"] < 85]

print("Calculating ATLAS ptvarcone20 cut")
# Concatenate ptvarcone 20 of ele1 and ele2 and sort
ptvarcone20 = np.concatenate( (data_get['muo1_ptvarcone20'], data_get['muo2_ptvarcone20'].values) )
ptvarcone20 = np.sort(ptvarcone20)

# Get number of muons to remove
nremove = np.int(ptvarcone20.shape[0] * 0.02)

# Get cut value
ptvarcone20Cut = ptvarcone20[-(nremove+1)]
print(f"    Cut at ptvarcone20:                {ptvarcone20Cut}")
print(f"    Number of data in ptvarcone20 cut: {np.sum(ptvarcone20 > ptvarcone20Cut)} ({np.sum(ptvarcone20 > ptvarcone20Cut)/ptvarcone20.shape[0]*100:.4f}%)\n")



MeV = (np.mean(data_get["muo1_pt"]) > 1000)
if MeV:
    print("Changing the pt to GeV")
    data_get["muo1_pt"] = data_get["muo1_pt"]/1000
    data_get["muo2_pt"] = data_get["muo2_pt"]/1000


data_get["isATLAS"] = GetATLASCut(data_get)
data_get["bkgATLAS"] = GetATLASBkg(data_get)

z1 = data_get['muo1_priTrack_z0']
z2 = data_get['muo2_priTrack_z0']
zSigma1 = data_get['muo1_priTrack_z0Sig']
zSigma2 = data_get['muo2_priTrack_z0Sig']
data_get['muo_z0_WgtAvrg'] = ((z1*zSigma1)+(z2*zSigma2))/(zSigma1+zSigma2)



data = data_get.copy()

# Check shapes
shapeAll = np.shape(data)

log.info(f"---- SHAPE OF DATA ----")
log.info(f"Shape all:       {shapeAll}")

# =========================
#       Variables
# =========================

training_var = ['dZ0',
                'Z_score',
                'pho_PID_score',
                'pho_ISO_score',
                'pho_isConv',
                ]
#============================================================================
# LGBM dataset and parameters
#============================================================================

log.info(f"Predicting")
t_start = time()


X = data[training_var]

# create LGBM dataset
dataset = lgb.Dataset(X)

#============================================================================
# Import the model
#============================================================================

print(f"Importing the model...")
bst = lgb.Booster(model_file = args.model)

#============================================================================
# Predict
#============================================================================

y_pred = bst.predict(X, num_iteration=bst.best_iteration)
data["predLGBM"] = y_pred

sel_train, nBkgLGBM, nBkgATLAS = getSameBkg(data, mask_LGBM, 2)

data["selLGBM"] = 0
data.loc[mask_LGBM(data, sel_train), ["selLGBM"]] = 1

sel_train_peakfit, sig, bkg, sig_ATLAS, bkg_ATLAS = GetFit(data, 2, 7)
sel_train_samesign, nSSLGBM, nSSATLAS, nOSLGBM, nOSATLAS = getSameSS(data, mask_LGBM, 2.5)

# sel_fit_bkg, sigFit, bkgFit, ATLAS_sig, ATLAS_bkg = GetFitBkg(data, 2, 6)
sel_train_peakfitBkg, sig_bkgFit, bkg_bkgFit, _, _ = GetFitBkg(data, 2.5, 3.5)

# data["selLGBM_fit"] = 0
# data.loc[mask_LGBM(data, sel_fit_bkg), ["selLGBM_fit"]] = 1

data["selLGBM_peakfit"] = 0
data.loc[mask_LGBM(data, sel_train_peakfit), ["selLGBM_peakfit"]] = 1
data["selLGBM_peakfitBkg"] = 0
data.loc[mask_LGBM(data, sel_train_peakfitBkg), ["selLGBM_peakfitBkg"]] = 1


data["selLGBM_SS"] = 0
data.loc[mask_LGBM(data, sel_train_samesign), ["selLGBM_SS"]] = 1

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(logit(data["predLGBM"]), bins = 100, range = (-10,15), histtype="step", label = "All data")
# ax.axvline(sel_fit_bkg, c='C0', label = f"Cut, bkg fit = {np.round(sel_fit_bkg,2)}")
ax.axvline(sel_train, c='C3', label = f"Cut, bkg > 110 GeV = {np.round(sel_train,2)}")
# ax.axvline(sel_train_peakfit, c='C1', label = f"Cut, fit = {np.round(sel_train_peakfit,2)}")
ax.axvline(sel_train_peakfitBkg, c='C1', label = f"Cut, peakfit same bkg = {np.round(sel_train_peakfitBkg,2)}")

ax.axvline(sel_train_samesign, c='C2', label = f"Cut, same-sign = {np.round(sel_train_samesign,2)}")
ax.set(xlabel = "LGBM prediction", ylabel = "Frequency", yscale = "log", title = "LGBM score")
ax.legend()
fig.tight_layout()
fig.savefig(args.outdir + "LGBMwCut.png", dpi = 400)

#============================================================================
# Predicting only Z - using ATLAS photon cuts
#============================================================================

#
# MLSel = getMLcut(data, GetZCut, 11, decimals = 2)
MLSel = GetMLCutFit(data, GetZCut, 2, 6)
MLSelPho = GetMLCutFit(data, GetPhoCut, -5, 3)

data["selLGBM_MLSel"] = 0
data.loc[GetZCut(data, MLSel), ["selLGBM_MLSel"]] = 1

data["selLGBM_MLSel_Pho"] = 0
data.loc[GetPhoCut(data, MLSelPho), ["selLGBM_MLSel_Pho"]] = 1



Likelihood_cut = GetZCut(data, 2.94)*1
sigZ, bkgZ, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], "ML cut, as Z model", args.outdir, plots = True, constant_mean = True,
                                       constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                       bkg_exp = True, bkg_cheb = False);

print(data["pho_ISO_score"])

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(data["Z_score"], bins = 100, range = (-10,15), histtype="step", label = "All data")
ax.axvline(MLSel, c='C0', label = f"Fit, Z selection {np.round(MLSel,2)}")
ax.axvline(2.94, linestyle ="dashed", label = r"$Z\mu\mu$ Data selection, fit =" + f"{2.94}")
ax.set(xlabel = "LGBM prediction", ylabel = "Frequency", yscale = "log", title = "LGBM score (Z model only)")
ax.legend()
fig.tight_layout()
fig.savefig(args.outdir + "LGBMscoreZonly2.png", dpi = 400)

#============================================================================
#                           Plotting invMass
#============================================================================

dataSig = data[(data["muo1_charge"]*data["muo2_charge"] == -1)]


dataATLAS = dataSig[(dataSig["isATLAS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM = dataSig[(dataSig["selLGBM"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
# dataLGBM_fitBkg = dataSig[(dataSig["selLGBM_fit"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_peakfit = dataSig[(dataSig["selLGBM_peakfit"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_samesign = dataSig[(dataSig["selLGBM_SS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_MLSel = dataSig[(dataSig["selLGBM_MLSel"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_MLSel_Pho = dataSig[(dataSig["selLGBM_MLSel_Pho"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_peakfitBkg = dataSig[(dataSig["selLGBM_peakfitBkg"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]

diffLGBM_ATLAS = ((len(dataLGBM)- len(dataATLAS))/len(dataATLAS))*100
# diffLGBM_ATLAS_fitBkg = ((len(dataLGBM_fitBkg)- len(dataATLAS))/len(dataATLAS))*100
diffLGBM_ATLAS_peakfit = ((len(dataLGBM_peakfit)- len(dataATLAS))/len(dataATLAS))*100
diffLGBM_ATLAS_samesign = ((len(dataLGBM_samesign)- len(dataATLAS))/len(dataATLAS))*100
diffLGBM_ATLAS_MLSel = ((len(dataLGBM_MLSel)- len(dataATLAS))/len(dataATLAS))*100
diffLGBM_ATLAS_MLSel_Pho = ((len(dataLGBM_MLSel_Pho)- len(dataATLAS))/len(dataATLAS))*100
diffLGBM_ATLAS_peakfitBkg = ((len(dataLGBM_peakfitBkg) - len(dataATLAS))/len(dataATLAS))*100

dataATLAS = dataSig[(dataSig["isATLAS"] == 1)]
dataLGBM = dataSig[(dataSig["selLGBM"] == 1)]
# dataLGBM_fitBkg = dataSig[(dataSig["selLGBM_fit"] == 1)]
dataLGBM_peakfit = dataSig[(dataSig["selLGBM_peakfit"] == 1)]
dataLGBM_samesign = dataSig[(dataSig["selLGBM_SS"] == 1)]
dataLGBM_MLSel = dataSig[(dataSig["selLGBM_MLSel"] == 1)]
dataLGBM_MLSel_Pho = dataSig[(dataSig["selLGBM_MLSel_Pho"] == 1)]
dataLGBM_peakfitBkg = dataSig[(dataSig["selLGBM_peakfitBkg"] == 1)]


fig, ax = plt.subplots(1,1,figsize=(7,5))
a = ax.hist(dataATLAS["invM"], bins = 100, range = (50,160), histtype = "step", label = f"ATLAS");
# ax.hist(dataLGBM["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Bkg cut, fit: LGBM(Z, MC) (+{np.round(diffLGBM_ATLAS_fitBkg,2)} %)\nnBkg ATLAS = {np.round(ATLAS_bkg)}, nBkg LGBM = {np.round(bkg)}");
# ax.hist(dataLGBM_peakfit["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Fit: LGBM(Z, MC) (+{np.round(diffLGBM_ATLAS_peakfit,2)} %)\nFit to get same bkg efficiency");
ax.hist(dataLGBM_peakfitBkg["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Fit, same bkg (+{np.round(diffLGBM_ATLAS_peakfitBkg,2)} %)")#\nnBkgATLAS = {np.round(bkg_ATLAS)}, nBkgLGBM = {np.round(bkg_bkgFit)}");
ax.hist(dataLGBM_samesign["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Same-sign cut ({np.round(diffLGBM_ATLAS_samesign,2)} %)")#\nnSS ATLAS = {nSSATLAS}, nSS LGBM = {nSSLGBM}");
ax.hist(dataLGBM["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Bkg cut > 110 GeV (+{np.round(diffLGBM_ATLAS,2)} %)")#\nnBkg ATLAS = {np.round(nBkgATLAS)}, nBkg LGBM = {np.round(nBkgLGBM)}");

# ax.hist(dataLGBM_MLSel["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Z selection only (+{np.round(diffLGBM_ATLAS_MLSel,2)} %)");
# ax.hist(dataLGBM_MLSel_Pho["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Pho selection only (+{np.round(diffLGBM_ATLAS_MLSel_Pho,2)} %)");
ax.grid(False)
bw = a[1][1] - a[1][0]
ax.set(xlabel = r"M$_{\mu\mu\gamma}$", ylabel = f"Events/{bw:4.2f}")
ax.set_title(r"$Z\rightarrow \mu\mu\gamma$ Data: Invariant mass", loc = 'left')
ax.legend(loc=1)
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(args.outdir + "invMplot.png", dpi=400)


fig, ax = plt.subplots(1,1,figsize=(7,5))
a = ax.hist(dataATLAS["invM"], bins = 100, range = (50,160), histtype = "step", label = f"ATLAS");
# ax.hist(dataLGBM["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Bkg cut, fit (+{np.round(diffLGBM_ATLAS_fitBkg,2)} %)")#\nnBkg ATLAS = {np.round(ATLAS_bkg)}, nBkg LGBM = {np.round(bkg)}");
# ax.hist(dataLGBM_peakfit["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Fit (+{np.round(diffLGBM_ATLAS_peakfit,2)} %)")#\nFit, same bkg efficiency");
ax.hist(dataLGBM_peakfitBkg["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Fit, same bkg (+{np.round(diffLGBM_ATLAS_peakfitBkg,2)} %)")#\nnBkgATLAS = {np.round(bkg_ATLAS)}, nBkgLGBM = {np.round(bkg_bkgFit)}");
ax.hist(dataLGBM_samesign["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Same-sign cut ({np.round(diffLGBM_ATLAS_samesign,2)} %)")#\nnSS ATLAS = {nSSATLAS}, nSS LGBM = {nSSLGBM}");
ax.hist(dataLGBM["invM"], bins = 100, range = (50,160), histtype = "step", label = f"Bkg cut > 110 GeV (+{np.round(diffLGBM_ATLAS,2)} %)")#\nnBkg ATLAS = {np.round(nBkgATLAS)}, nBkg LGBM = {np.round(nBkgLGBM)}");

# ax.hist(dataLGBM_MLSel["invM"], bins = 100, range = (50,200), histtype = "step", label = f"Z selection only: LGBM(Z, MC) (+{np.round(diffLGBM_ATLAS_MLSel,2)} %)");
# ax.hist(dataLGBM_MLSel_Pho["invM"], bins = 100, range = (50,200), histtype = "step", label = f"Pho selection only: LGBM(Z, MC) (+{np.round(diffLGBM_ATLAS_MLSel_Pho,2)} %)");
ax.grid(False)
bw = a[1][1] - a[1][0]
ax.set(xlabel = r"M$_{\mu\mu\gamma}$", ylabel = f"Events/{bw:4.2f}")
ax.set_title(r"$Z\rightarrow \mu\mu\gamma$ Data: Invariant mass", loc = 'left')
ax.legend(loc=1)
# ax.set_yscale('log')
fig.tight_layout()
fig.savefig(args.outdir + "invMplot_NotLog.png", dpi=400)

#============================================================================
# Saving data
#============================================================================


# data = data.fillna(0)
# data.info()
# data = data.astype(np.float16)
# data.info()
data.to_hdf(args.outdir + "pred_data.h5", key='df', mode='w')



#============================================================================
# Plotting cuts
#============================================================================

print(np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 1), np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 99))
print(np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 10), np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 90))

fig, ax = plt.subplots(2,2,figsize=(15,10))
ax = ax.flatten()
ax[0].hist(dataATLAS["dZ0"], bins = 100, range = (np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 15), np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 90)), histtype="step", label = "ATLAS");
ax[0].hist(dataLGBM["dZ0"], bins = 100, range = (np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 15), np.percentile(dataATLAS[~np.isnan(dataATLAS["dZ0"])]["dZ0"], 90)), histtype="step", label = "LGBM");
ax[0].set(xlabel = "dZ0");
ax[0].legend()

ax[1].hist(dataATLAS["Z_score"], bins = 100, histtype="step", range = (-20,20), label = "ATLAS");
ax[1].hist(dataLGBM["Z_score"], bins = 100, histtype="step", range = (-20,20), label = "LGBM");
ax[1].set(xlabel = "Z_score");
ax[1].legend()

ax[2].hist(dataATLAS["pho_PID_score"], bins = 100, histtype="step", range = (-5,5), label = "ATLAS");
ax[2].hist(dataLGBM["pho_PID_score"], bins = 100, histtype="step", range = (-5,5), label = "LGBM");
ax[2].set(xlabel = "pho_PID_score");
ax[2].legend()

ax[3].hist(dataATLAS["pho_ISO_score"], bins = 100, histtype="step", range = (-5,5), label = "ATLAS");
ax[3].hist(dataLGBM["pho_ISO_score"], bins = 100, histtype="step", range = (-5,5), label = "LGBM");
ax[3].set(xlabel = "pho_ISO_score");
ax[3].legend()

fig.tight_layout()
fig.savefig(args.outdir + "ATLAS_LGBM_Vars.png", dpi = 400)


dataLGBM_peakfit = dataSig[(dataSig["selLGBM_peakfit"] == 1)]


dataLGBM_peakfit["deltaPhi1"] = 0
dataLGBM_peakfit["deltaPhi2"] = 0
for index, row in dataLGBM_peakfit.iterrows():
    if np.abs(row["pho_phi"] - row["muo1_phi"] - 2*np.pi) < np.abs(row["pho_phi"] - row["muo1_phi"]):
        row["deltaPhi1"] = np.abs(row["pho_phi"] - row["muo1_phi"] - 2*np.pi)
    else:
        row["deltaPhi1"] = np.abs(row["pho_phi"] - row["muo1_phi"])

    if np.abs(row["pho_phi"] - row["muo2_phi"] - 2*np.pi) < np.abs(row["pho_phi"] - row["muo2_phi"]):
        row["deltaPhi2"] = np.abs(row["pho_phi"] - row["muo2_phi"] - 2*np.pi)
    else:
        row["deltaPhi2"] = np.abs(row["pho_phi"] - row["muo2_phi"])

# deltaPhi1 = (dataLGBM_peakfit["pho_phi"] - dataLGBM_peakfit["muo1_phi"] - 2*np.pi)
# deltaPhi2 = (dataLGBM_peakfit["pho_phi"] - dataLGBM_peakfit["muo2_phi"] - 2*np.pi)
deltaEta1 = (dataLGBM_peakfit["pho_eta"] - dataLGBM_peakfit["muo1_eta"])
deltaEta2 = (dataLGBM_peakfit["pho_eta"] - dataLGBM_peakfit["muo2_eta"])
dataLGBM_peakfit["DeltaR1"] = np.sqrt(dataLGBM_peakfit["deltaPhi1"]**2 + deltaEta1**2)
dataLGBM_peakfit["DeltaR2"] = np.sqrt(dataLGBM_peakfit["deltaPhi2"]**2 + deltaEta2**2)


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(dataLGBM_peakfit["DeltaR1"], histtype="step", density=True, bins = 50, label = "DeltaR for muo1");
ax[0].axvline(0.3, color = 'k', linestyle = "dashed", label = "0.3")
ax[0].set(xlabel = r"$\Delta$R", ylabel = "Frequency")
ax[0].legend()
ax[0].grid(False)
ax[1].hist(dataLGBM_peakfit["DeltaR2"], histtype="step", density=True, bins = 50, label = "DeltaR for muo2");
ax[1].axvline(0.3, color = 'k', linestyle = "dashed", label = "0.3")
ax[1].set(xlabel = r"$\Delta$R", ylabel = "Frequency")
ax[1].legend()
ax[1].grid(False)

fig.savefig(args.outdir + "DeltaR" + ".png")


#============================================================================
# Plots
#============================================================================



fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(sig_ATLAS, bkg_ATLAS, s = 30, c = "C0", label = "ATLAS ref.")
ax.axhline(bkg_ATLAS, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(y = bkg_ATLAS+200, x = sig_ATLAS+3000, s = f"same n_background", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

f_bkg = bkg_ATLAS/(sig_ATLAS) #+bkg_ATLAS
slope_degrees = np.degrees(np.arctan(f_bkg))+25
x = np.linspace(sig_ATLAS-7000, sig_ATLAS*1.3+500+10000, 100)
def slope(x):
    return f_bkg*x

ax.plot(x, slope(x), color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(y = slope(sig_ATLAS+3000)+70, x = sig_ATLAS+3000, s = f"same f_background", alpha = 0.3, rotation=slope_degrees, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(sig_ATLAS, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = sig_ATLAS+100, y = bkg_ATLAS-50, s = f"100%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(sig_ATLAS*1.1, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = sig_ATLAS*1.1+100, y = bkg_ATLAS-50, s = f"110%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(sig_ATLAS*1.2, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = sig_ATLAS*1.2+100, y = bkg_ATLAS-50, s = f"120%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(sig_ATLAS*1.3, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = sig_ATLAS*1.3+100, y = bkg_ATLAS-50, s = f"130%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)


ax.scatter(sig, bkg, s = 30, c = "C1", label = "LGBM, fit")
sig_old, bkg_old = sig, bkg

cuts = [0.3, 0.7, 1, 1.2]

Likelihood_cut = (logit(data["predLGBM"]) > sel_train_peakfit-cuts[0])*1
sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], f"Fpr cut new", args.outdir, plots = True, constant_mean = True,
                                       constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                       bkg_exp = True, bkg_cheb = False);
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C1", label =  "Looser fit")

Likelihood_cut = (logit(data["predLGBM"]) > sel_train_peakfit-cuts[1])*1
sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], f"Fpr cut new", args.outdir, plots = True, constant_mean = True,
                                       constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                       bkg_exp = True, bkg_cheb = False);
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C1")
Likelihood_cut = (logit(data["predLGBM"]) > sel_train_peakfit-cuts[2])*1
sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], f"Fpr cut new", args.outdir, plots = True, constant_mean = True,
                                       constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                       bkg_exp = True, bkg_cheb = False);
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C1")
Likelihood_cut = (logit(data["predLGBM"]) > sel_train_peakfit-cuts[3])*1
sig, bkg, f_bkg, chi2 = PeakFit_likelihood(Likelihood_cut, data["invM"], f"Fpr cut new", args.outdir, plots = True, constant_mean = True,
                                       constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                       bkg_exp = True, bkg_cheb = False);
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C1")

ax.set(xlabel = "n signal", ylabel = "n background", xlim = (sig_ATLAS-100, sig+500), ylim = (bkg_ATLAS-1000, bkg+200))
# ax.set(xlabel = "n signal", ylabel = "n same-sign", xlim = (sig_ATLAS-5000, sig_ATLAS*1.3+500+5000), ylim = (bkg_ATLAS-100, bkg+300))
ax.set_title(r"$Z\rightarrow \mu\mu\gamma$ Data: $f_{bkg}$ increase", loc = 'left')

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels, loc=1)

# ax.legend()
ax.grid(False)
# ax.set_yscale('log')
fig.tight_layout()
fig.savefig(args.outdir + "nSig_nBkg.png", dpi=400)

print(sig_ATLAS, bkg_ATLAS)
print(sig_old, bkg_old)


# min = np.percentile(logit(data["predLGBM"]),1)
# dataNotInf = data[np.isfinite(logit(data["predLGBM"]))]
# max = np.percentile(logit(dataNotInf["predLGBM"]),99)

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(logit(data["predLGBM"]), range = (-10,15), color='C0', bins = 100, histtype="step", label = "All data")
# ax.axvline(sel_train, c='b', label = f"Cut, bkg > 110 GeV = {np.round(sel_train,2)}")
ax.axvline(sel_train_peakfit, color='C1', label = f"Cut, peakfit = {np.round(sel_train_peakfit,2)}")
ax.axvline(sel_train_peakfit-cuts[0], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[1], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[2], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[3], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
# ax.axvline(sel_train_peakfit, c='C2', label = f"Cut, peakfit = {np.round(sel_train_peakfit,2)}")
ax.set(xlabel = "LGBM prediction", ylabel = "Frequency", title = "LGBM score", yscale = 'log')
ax.legend()
ax.grid(False)
fig.tight_layout()
fig.savefig(args.outdir + "LGBMcut_fbkg.png", dpi = 400)



#############################
######### Same-sign #########
#############################

dataATLAS = dataSig[(dataSig["isATLAS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_SS = dataSig[(dataSig["selLGBM_SS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]


fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(len(dataATLAS), nSSATLAS, s = 30, c = "C0", label = "ATLAS ref.")
ax.axhline(nSSATLAS, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(y = nSSATLAS+30, x = len(dataATLAS)+7500, s = f"same n_SS", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.scatter(len(dataLGBM_SS), nSSLGBM, s = 30, c = "C2", label = "LGBM same-sign")

ax.axvline(len(dataATLAS), color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = len(dataATLAS)+1000, y = nSSATLAS-50, s = f"100%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(len(dataATLAS)*1.1, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = len(dataATLAS)*1.1+1000, y = nSSATLAS-50, s = f"110%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(len(dataATLAS)*1.2, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = len(dataATLAS)*1.2+1000, y = nSSATLAS-50, s = f"120%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

ax.axvline(len(dataATLAS)*1.3, color = 'k', linestyle = "dashed", alpha = 0.1)
ax.text(x = len(dataATLAS)*1.3+1000, y = nSSATLAS-50, s = f"130%", alpha = 0.3, color='black', ha = 'center', va = 'top', size = 7)

cuts_SS = [0.5, 1, 1.5, 2]

bkg = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[0]) ) & ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
sig = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[0])) & (data["invM"] > 70) & (data["invM"] < 110)])
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C2", label = "LGBM n same-sign looser")
bkg = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[1]) )& ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
sig = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[1]) )& (data["invM"] > 70) & (data["invM"] < 110)])
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C2")
bkg = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[2])) & ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
sig = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[2])) & (data["invM"] > 70) & (data["invM"] < 110)])
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C2")
bkg = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[3]) )& ((data["muo1_charge"]*data["muo2_charge"]) > 0) & (((data["invM"] <  91.2-15) & (data["invM"] >  91.2-40))  | ((data["invM"] >  91.2+15) & (data["invM"] <  91.2+40)))])
sig = len(data[(logit(data["predLGBM"]) > (sel_train_samesign-cuts_SS[3]) )& (data["invM"] > 70) & (data["invM"] < 110)])
ax.scatter(sig, bkg, s = 30, marker = '*', c = "C2")


ax.set(xlabel = "n signal", ylabel = "n same-sign", xlim = (len(dataATLAS)-5000, len(dataATLAS)*1.3+500+5000), ylim = (nSSATLAS-100, bkg+300))
ax.set_title(r"$Z\rightarrow \mu\mu\gamma$ Data: $n_{SS}$ increase", loc = 'left')

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels, loc = 1)

# ax.legend()
ax.grid(False)
# ax.set_yscale('log')
fig.tight_layout()
fig.savefig(args.outdir + "nSig_nBkg_nSS.png", dpi=400)


# min = np.percentile(logit(data["predLGBM"]),1)
# dataNotInf = data[np.isfinite(logit(data["predLGBM"]))]
# max = np.percentile(logit(dataNotInf["predLGBM"]),99)

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(logit(data["predLGBM"]), range = (-10,15), color='C0', bins = 100, histtype="step", label = "All data")
# ax.axvline(sel_train, c='b', label = f"Cut, bkg > 110 GeV = {np.round(sel_train,2)}")
ax.axvline(sel_train_samesign, color='C2', label = f"Cut, samesign = {np.round(sel_train_samesign,2)}")
ax.axvline(sel_train_samesign-cuts_SS[0], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[1], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[2], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[3], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
# ax.axvline(sel_train_peakfit, c='C2', label = f"Cut, peakfit = {np.round(sel_train_peakfit,2)}")
ax.set(xlabel = "LGBM prediction", ylabel = "Frequency", title = "LGBM score", yscale = 'log')
ax.legend(loc=3)
ax.grid(False)
fig.tight_layout()
fig.savefig(args.outdir + "LGBMcut_nSS.png", dpi = 400)

# min = np.percentile(logit(data["predLGBM"]),1)
# dataNotInf = data[np.isfinite(logit(data["predLGBM"]))]
# max = np.percentile(logit(dataNotInf["predLGBM"]),99)

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(logit(data["predLGBM"]), range = (-10,15), color='C0', bins = 100, histtype="step", label = "LGBM score")
# ax.axvline(sel_train, c='b', label = f"Cut, bkg > 110 GeV = {np.round(sel_train,2)}")
ax.axvline(sel_train_samesign, color='C2', label = f"Cut, samesign = {np.round(sel_train_samesign,2)}")
ax.axvline(sel_train_samesign-cuts_SS[0], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[1], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[2], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
ax.axvline(sel_train_samesign-cuts_SS[3], color='C2', linewidth = 1, linestyle = "dashed", alpha = 0.8)
# ax.axvline(sel_train_peakfit, c='C2', label = f"Cut, peakfit = {np.round(sel_train_peakfit,2)}")
ax.axvline(sel_train_peakfit, color='C1', label = f"Cut, peakfit = {np.round(sel_train_peakfit,2)}")
ax.axvline(sel_train_peakfit-cuts[0], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[1], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[2], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)
ax.axvline(sel_train_peakfit-cuts[3], color='C1', linewidth = 1, linestyle = "dashed", alpha = 0.2)

ax.set(xlabel = "LGBM prediction", ylabel = "Frequency", yscale = 'log')#, title = "LGBM score")
ax.legend(loc=3)
ax.grid(False)
fig.tight_layout()
fig.savefig(args.outdir + "LGBMcut_together.png", dpi = 400)



print("number of same-sign and opposite sign events")
print(f"ATLAS: nSS (bkg) = {nSSATLAS}, nOS (signal) =  {nOSATLAS}")
print(f"LGBM: nSS (bkg) = {nSSLGBM}, nOS (signal) =  {nOSLGBM}")



print("\n")
print("\n")
dataATLAS = dataSig[(dataSig["isATLAS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM = dataSig[(dataSig["selLGBM"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_samesign = dataSig[(dataSig["selLGBM_SS"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_peakfit = dataSig[(dataSig["selLGBM_peakfit"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]
dataLGBM_peakfitBkg = dataSig[(dataSig["selLGBM_peakfitBkg"] == 1) & (dataSig["invM"] > 70) & (dataSig["invM"] < 110)]


print("Get numbers for SS/OS:")
print(f"ATLAS nSS (bkg): {nSSATLAS}, nOS (sig): {len(dataATLAS)}")
change_SS = -((len(dataATLAS)-len(dataLGBM_samesign))/len(dataATLAS))*100
print(f"MC nSS (bkg): {nSSLGBM}, nOS (sig): {len(dataLGBM_samesign)}, change {np.round(change_SS,2)} %")
print("\n")

print("Get numbers for bkg > 110 GeV:")
print(f"ATLAS bkg: {nBkgATLAS}, sig: {len(dataATLAS)}")
change_BKG = -((len(dataATLAS)-len(dataLGBM))/len(dataATLAS))*100
print(f"MC bkg: {nBkgLGBM},  sig: {len(dataLGBM)}, change {np.round(change_BKG,2)} %")
print("\n")

print("Get numbers for fit (same f_bkg):")
print(f"ATLAS bkg: {bkg_ATLAS}, sig: {sig_ATLAS}")
change_fit = -((len(dataATLAS)-len(dataLGBM_peakfit))/len(dataATLAS))*100
print(f"MC bkg: {bkg_old}, sig: {sig_old}, change {np.round(change_fit,2)} %")
print("\n")

print("Get numbers for fit (same bkg):")
print(f"ATLAS bkg: {bkg_ATLAS}, sig: {sig_ATLAS}")
change_fit = -((len(dataATLAS)-len(dataLGBM_peakfitBkg))/len(dataATLAS))*100
print(f"MC bkg: {bkg_bkgFit}, sig: {sig_bkgFit}, change {np.round(change_fit,2)} %")
print("\n")





log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
