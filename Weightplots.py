#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for plotting of the weights
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


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/ReweightPlots/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--invMmin', action='store', default=50.0, type=float,
                    help='Minimum value of invariant mass. (Default = 50.0)')
parser.add_argument('--invMmax', action='store', default=150.0, type=float,
                    help='Maximum value of invariant mass. (Default = 150.0)')
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
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["eeg", "mmg", 'ee', 'mm', "muo", "ele", "pho","gg"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')



args = parser.parse_args()

log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "ele") + (args.PartType != "muo") + (args.PartType != "eeg") + (args.PartType != "mmg") + (args.PartType != "pho")+(args.PartType != "ee")+(args.PartType != "mm")+(args.PartType !="gg"))!=7:
    log.error("Unknown lepton, use either ele, muo, eeg, mmg or pho")
    quit()


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
# Plotting and printing weights for training set
#============================================================================
# Create masks
data_all = pd.concat(data_list, ignore_index= True)
trainMask = (data_all["dataset"] == 0)
validMask = (data_all["dataset"] == 1)
masks = [trainMask, validMask]
maskNames = ["train", "valid"]
maskLabel = ["Training set", "Validation set"]

weightTypes = ["regWeight", "revWeight"]
weightTypeNames = ["Regular", "Reverse"]

weightLinestyle = ["dotted", "dashed", "dashdot", "solid"]
reweightNames = ["nEst10", "nEst40", "nEst100", "nEst200"]
#============================================================================
# Plotting histogram of weighted features
#============================================================================

#log.info(f"Plotting data for {maskNames[iMask]}")

# Columns to be plotted 
if (args.PartType == "eeg") or (args.PartType == "ee"):
    colTypes = ['correctedScaledAverageMu', "eta", "et", 'invM' ]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$", r"$M_{ee}$" ]
elif (args.PartType == "mmg") or (args.PartType == "mm"):
    colTypes = ['correctedScaledAverageMu', "eta", "pt", 'invM' ]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$P_T$", r"$M_{\mu\mu}$" ]
elif args.PartType == "ele":
    #colTypes = ['correctedScaledAverageMu', 'ele_eta', 'ele_et']
    #colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$"]
    colTypes = ['correctedScaledAverageMu', 'ele_eta', 'ele_et','correctedScaledAverageMu', 'ele_eta', 'ele_et']
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$","$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$"]
elif args.PartType == "pho":
    colTypes = ['correctedScaledAverageMu', 'pho_eta', 'pho_et','correctedScaledAverageMu', 'pho_eta', 'pho_et']
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$","$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$"]
elif args.PartType == "muo":
    colTypes = ['correctedScaledAverageMu', "muo_eta", "muo_pt",'correctedScaledAverageMu', "muo_eta", "muo_pt"]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$P_T$","$\\langle \\mu \\rangle$", "$\\eta$", "$P_T$"]
elif args.PartType == "gg": 
    colTypes = ['correctedScaledAverageMu', "eta", "et", 'invM' ]
    colNames = ["$\\langle \\mu \\rangle$", "$\\eta$", "$E_T$", "$M_{\gamma\gamma}$" ]


# Set plotting parameters
if (args.PartType == "eeg") or (args.PartType == "mmg") or (args.PartType == "ee") or (args.PartType == "mm")or (args.PartType=="gg"):
    if args.PartType =="gg":
        xRange = [(0,100), (-7,7), (0,1000), (80, 150)]
    elif (args.PartType == "eeg") or (args.PartType == "ee"):
        xRange = [(0,100), (-7,7), (0,1000), (args.invMmin, args.invMmax)]
    else:
        xRange = [(0,100), (-7,7), (0,10**4), (args.invMmin, args.invMmax)]
    yRangeTrain = [(0,6.5*10**4), (10**0/2,10**4), (10**0/2,5*10**5), (0,8*10**4)]
    if args.PartType =="gg":
        yRangeValid = [(0,4*10**4), (10**0/2,10**3), (10**0/2,3*10**3), (0,4*10**4)]    
    elif args.PartType == "eeg":
        yRangeValid = [(0,4*10**4), (10**0/2,2*10**4), (10**0/2,10**5), (0,4*10**4)]
    elif args.PartType == "mmg":
        yRangeValid = [(0,3*10**4), (10**0/2,2*10**4), (10**0/2,2.5*10**2), (0,3*10**4)]
    else:
        yRangeValid = [(0,10**5), (10**0/2,2*10**4), (10**0/2,5*10**4), (0,10**5)]
    binwidth = [5,0.05,10,args.binWidth]
    binUnits = ["",""," GeV", " GeV"]
    nBins = [int((xRange[0][1]-xRange[0][0])/binwidth[0]),int((xRange[1][1]-xRange[1][0])/binwidth[1]), int((xRange[2][1]-xRange[2][0])/binwidth[2]),  int((args.invMmax-args.invMmin)/binwidth[3])]
    logScale = [False, True, True, False]

    weightColor = ['C1', 'C2', 'C4', 'C5']

    placement = [(0,0),(1,0),(0,1),(1,1)]
else: 
    xRange = [(0,100), (-7,7), (0,1000),(0,100), (-7,7), (0,1000)]
    yRangeTrain = [(10**0,6*10**5), (10**0/2,5*10**4), (10**0/2,6*10**5),(10**0,6*10**5), (10**0/2,5*10**4), (10**0/2,6*10**5)]
    if args.PartType =="muo":
        yRangeValid = [(10**0,3*10**5), (10**0/2,3*10**4), (10**0/2,5*10**5),(10**0,3*10**5), (10**0/2,3*10**4), (10**0/2,5*10**5)]
    else:
        yRangeValid = [(10**0,3*10**5), (10**0/2,2*10**4), (10**0/2,4*10**5),(10**0,3*10**5), (10**0/2,2*10**4), (10**0/2,4*10**5)]
    binwidth = [5,0.05,10,5,0.05,10]
    binUnits = ["",""," GeV","",""," GeV"]
    nBins = [int((xRange[0][1]-xRange[0][0])/binwidth[0]),int((xRange[1][1]-xRange[1][0])/binwidth[1]), int((xRange[2][1]-xRange[2][0])/binwidth[2]),int((xRange[0][1]-xRange[0][0])/binwidth[0]),int((xRange[1][1]-xRange[1][0])/binwidth[1]), int((xRange[2][1]-xRange[2][0])/binwidth[2])]
    logScale = [True, True, True,True, True, True]

    weightColor = ['C1', 'C2', 'C4','C1', 'C2', 'C4']

    placement = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]


# Plot weighted eta, et and <mu> for regular and reverse reweighing
if (args.PartType == "ee") or (args.PartType == "mm") or (args.PartType =="gg"):
    for iType, weightType in enumerate(weightTypes):
        for iMask, mask in enumerate(masks):
            print(f"Plotting {weightTypeNames[iType]} weighted distributions for {maskNames[iMask]}")

            # Initiate figure
            fig, ax = plt.subplots(2,2,figsize=(8,5))

            for i, colType in enumerate(colTypes):
                sigData = data_all[colType][mask & (data_all["type"]==1)]
                bkgData = data_all[colType][mask & (data_all["type"]==0)]
                # Plot data in separate figure
                ax[placement[i]].hist(sigData, label="Signal",     color='C0', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
                ax[placement[i]].hist(bkgData, label="Background", color='C3', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)

                for iWeight, weightName in enumerate(reweightNames[:-1]):
                    print(f"        Adding weighted data: {weightType+'_'+weightName}")
                    if iType==0: # Regular weights
                        bkgWeights = data_all[weightType+'_'+weightName][mask & (data_all["type"]==0)]
                        bkgLabel = f"Weights {weightName}"
                        ax[placement[i]].hist(bkgData, weights=bkgWeights, label=bkgLabel, color=weightColor[iWeight], linestyle=weightLinestyle[iWeight], range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
                    elif iType==1: # Reverse weights
                        sigWeights = data_all[weightType+'_'+weightName][mask & (data_all["type"]==1)]
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
            if args.PartType == "ee":
                plt.text(0.01, 0.99, f"Zee - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
            elif args.PartType =="mm":
                plt.text(0.01, 0.99, r"Z$\mu\mu$"+f" - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
            elif args.PartType == "ele":
                plt.text(0.01, 0.99, f"Electrons - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
            elif args.PartType == "muo":
                plt.text(0.01, 0.99, f"Muons - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
            elif args.PartType =="pho":
                plt.text(0.01, 0.99, f"Photons - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
            elif args.PartType == "gg":
                plt.text(0.01, 0.99, r"H$\gamma\gamma$"+f" - {weightTypeNames[iType]} reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)

            # Save and close separate figure
            fig.savefig(args.outdir+fname+f'_weightedHistogram_{weightType}_{maskNames[iMask]}.png')
            plt.close(fig)
    #for iType, weightType in enumerate(weightTypes):
elif (args.PartType == "ele") or (args.PartType == "muo") or (args.PartType == "pho"):
    for iMask, mask in enumerate(masks):
        print(f"Plotting weighted distributions for {maskNames[iMask]}")

        # Initiate figure
        fig, ax = plt.subplots(2,3,figsize=(12,8))

        for i, colType in enumerate(colTypes):
            sigData = data_all[colType][mask & (data_all["type"]==1)]
            bkgData = data_all[colType][mask & (data_all["type"]==0)]
            # Plot data in separate figure
            ax[placement[i]].hist(sigData, label="Signal",     color='C0', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
            ax[placement[i]].hist(bkgData, label="Background", color='C3', range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)

            for iWeight, weightName in enumerate(reweightNames[:-1]):
                print(f"        Adding weighted data: {'_'+weightName}")
                if placement[i][0]==0: # Regular weights
                    bkgWeights = data_all[weightTypes[0]+'_'+weightName][mask & (data_all["type"]==0)]
                    bkgLabel = f"Weights {weightName}"
                    ax[placement[i]].hist(bkgData, weights=bkgWeights, label=bkgLabel, color=weightColor[iWeight], linestyle=weightLinestyle[iWeight], range=xRange[i], bins=nBins[i], histtype='step', alpha=0.9)
                elif placement[i][0]==1: # Reverse weights
                    sigWeights = data_all[weightTypes[1]+'_'+weightName][mask & (data_all["type"]==1)]
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
        ax[0,2].legend(loc="upper right" ) #, bbox_to_anchor=(1.04, 1), borderaxespad=0) #, framealpha=1, fontsize=9, edgecolor='k')
        fig.tight_layout(rect=[0,0,1,0.98], h_pad=0.3, w_pad=0.3)
        if args.PartType == "ele":
            plt.text(0.01, 0.99, f"Electrons - Regular (Upper) and Reverse (Lower) reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        elif args.PartType == "muo":
            plt.text(0.01, 0.99, f"Muons - Regular (Upper) and Reverse (Lower) reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        elif args.PartType =="pho":
            plt.text(0.01, 0.99, f"Photons - Regular (Upper) and Reverse (Lower)reweighted {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        # Save and close separate figure

        fig.savefig(args.outdir+fname+f'_weightedHistogram_{maskNames[iMask]}.png')
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
                    processData = data_all['invM'][mask & (data_all["type"]==subfigi) & (data_all["process"]==process)]
                    processesData.append(processData)
                    if weightName!=None:
                        processWeights = data_all[weightTypes[subfigi]+'_'+weightName][mask & (data_all["type"]==subfigi) & (data_all["process"]==process)]
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
                    ax[-1,subfigj].set_xlabel(r"$M_{\muo\muo\gamma}$")
            ax[1,0].set_ylabel(f"Signal \nFrequency / {args.binWidth} GeV")
            ax[0,0].set_ylabel(f"Background \nFrequency / {args.binWidth} GeV")

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",ncol=4)

        # Save
        fig.tight_layout(rect=[0,0.1,1,0.98], h_pad=0.3, w_pad=0.3)
        plt.text(0.01, 0.99, f"Zeeg {maskLabel[iMask]}", ha='left', va='top', transform=fig.transFigure, fontsize=12)
        fig.savefig(args.outdir+fname+f'_SigBkgByProcess_combined_{maskNames[iMask]}.png')
        plt.close(fig)

print("Program done!")