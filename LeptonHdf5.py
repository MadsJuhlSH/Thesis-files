#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This script is supposed to find the chosen lepton in the data.
print("Program running...")
import warnings
warnings.filterwarnings('ignore', 'ROOT .+ is currently active but you ')
warnings.filterwarnings('ignore', 'numpy .+ is currently installed but you ')

import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt

from utils import mkdir
from itertools import combinations
from skhep.math import vectors
from sklearn.model_selection import train_test_split
import multiprocessing

from time import time
from datetime import timedelta

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for training.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/SignalFiles/Lep/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--fwd', action='store', type=int, default=0,
                    help='Pair using forward container? default=0 (false)')
parser.add_argument('--selection', action = 'store', default = 2, type = int,
                    help = 'Selects selection, 0 = ePid, 1 = eIso, 2 = both')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["ele", "muo"],
                    help = 'The choice of particle l in Z->llgam, either ele or muo')


args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error("No HDF5 file was specified.")
    quit()

if args.max_processes > 20:
    log.error("The requested number of processes ({}) is excessive (>20). Exiting.".format(args.max_processes))
    quit()

log.info("Selected particletype is {}".format(args.PartType))
if ((args.PartType != "ele") + (args.PartType != "muo"))!=1:
    log.error("Unknown lepton, use either ele or muo")
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

def signalSelection(hf, event, ele, selectiontype, histNtype):
    """
    Selects Electrons from electron candidates based on flowchart and type of selection.

    Arguments:
        hf: File to get variables from
        event: Event number
        ele: Electron candidates.
        Selectiontype: ePid, eIso or both (0, 1, 2)

    Returns:
        The type of the electron:
        0 = background
        1 = signal
        2 = Trash
    """
    
    #log.info("pdgId = {}, ele = {}, event = {}".format(hf['ele_truthPdgId_atlas'][event], ele, event))
    #log.info("What is the shape? {}, the element {}".format(hf['ele_truthPdgId_atlas'][event].shape,[ele]))
    if args.PartType == "ele":
        if selectiontype == 0:          #Pid    
            if (np.abs(hf['ele_truthPdgId_atlas'][event][ele]) == 11):
                histNtype[0,1] += 1
                return 1 # Signal
            else:
                histNtype[0,0] += 1
                return 0 # background
        elif selectiontype == 1:           #Iso
            if (hf['ele_truthType'][event][ele]==2):
                histNtype[0,1] += 1
                return 1 # Signal
            elif (hf['ele_truthType'][event][ele]==3) or (hf['ele_truthType'][event][ele]==0) :      # 3 is nonisoelectron 0 is unknown.
                histNtype[0,0] += 1
                return 0 # background
            else: 
                histNtype[0,2] += 1
                return 2 #Trash
    elif args.PartType == "muo":
        if selectiontype == 0:      #Pid
            if (np.abs(hf['muo_truthPdgId'][event][ele]) == 13):
                histNtype[0,1] += 1
                return 1 # Signal
            else:
                histNtype[0,0] += 1
                return 0 # background
        elif selectiontype == 1:        # Iso
            if (hf['muo_truthType'][event][ele]==6):
                histNtype[0,1]+=1
                return 1 # Signal
            elif (hf['muo_truthType'][event][ele]==7) or (hf['muo_truthType'][event][ele]==0):
                histNtype[0,0] += 1
                return 0 # background
            else: 
                histNtype[0,2] += 1
                return 2 #Trash

def addElectronVariables(hf, event, data_temp, ele):
    """ Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data

    Arguments:
        hf: File to get variables from.
        event: Event Number.
        data_temp: Numpy array to add variables to.
        ele: Electron index.

    Returns: 
        Nothing. Data is set in existing array.
    """
    data_temp[ 0, column_names.index( f'ele_truthParticle' ) ] = hf[ 'ele_egamTruthParticle' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_truthPdgId' ) ] = hf[ 'ele_truthPdgId_atlas' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_truthType' ) ] = hf[ 'ele_truthType' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_truthOrigin' ) ] = hf[ 'ele_truthOrigin' ][ event ][ ele ]    
    data_temp[ 0, column_names.index( f'ele_LHLoose' ) ] = hf[ 'ele_LHLoose' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_LHMedium' ) ] = hf[ 'ele_LHMedium' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_LHTight' ) ] = hf[ 'ele_LHTight' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_trigger' ) ] = hf[ 'ele_trigger' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_m' ) ] = hf[ 'ele_m' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_e' ) ] = hf[ 'ele_e' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_et' ) ] = hf[ 'ele_et' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_eta' ) ] = hf[ 'ele_eta' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_phi' ) ] = hf[ 'ele_phi' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_charge' ) ] = hf[ 'ele_charge' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele_ECIDSResult' ) ] = hf[ 'ele_ECIDSResult' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_d0' ) ] = hf[ 'ele_d0' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_d0Sig' ) ] = hf[ 'ele_d0Sig' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_z0' ) ] = hf[ 'ele_z0' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_z0Sig' ) ] = hf[ 'ele_z0Sig' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_f3' ) ] = hf[ 'ele_f3' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_f1' ) ] = hf[ 'ele_f1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_Reta' ) ] = hf[ 'ele_Reta' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_Rphi' ) ] = hf[ 'ele_Rphi' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_Rhad' ) ] = hf[ 'ele_Rhad' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_Rhad1' ) ] = hf[ 'ele_Rhad1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_weta2' ) ] = hf[ 'ele_weta2' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_Eratio' ) ] = hf[ 'ele_Eratio' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_TRTPID' ) ] = hf[ 'ele_TRTPID' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_dPOverP' ) ] = hf[ 'ele_dPOverP' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_deltaEta1' ) ] = hf[ 'ele_deltaEta1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_deltaPhiRescaled2' ) ] = hf[ 'ele_deltaPhiRescaled2' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptvarcone20' ) ] = hf[ 'ele_ptvarcone20' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_topoetcone20' ) ] = hf[ 'ele_topoetcone20' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_topoetcone40' ) ] = hf[ 'ele_topoetcone40' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptcone20_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptcone20_TightTTVALooseCone_pt500' ) ] = hf[ 'ele_ptcone20_TightTTVALooseCone_pt500' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptvarcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone20_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele_ptvarcone20_TightTTVA_pt1000' ) ] = hf[ 'ele_ptvarcone20_TightTTVA_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptvarcone30_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone30_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptvarcone30_TightTTVALooseCone_pt500' ) ] = hf[ 'ele_ptvarcone30_TightTTVALooseCone_pt500' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele_ptvarcone30_TightTTVA_pt1000' ) ] = hf[ 'ele_ptvarcone30_TightTTVA_pt1000' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele_ptvarcone30_TightTTVA_pt500' ) ] = hf[ 'ele_ptvarcone30_TightTTVA_pt500' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_ptvarcone40_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone40_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_topoetcone20ptCorrection' ) ] = hf[ 'ele_topoetcone20ptCorrection' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_expectInnermostPixelLayerHit')] = hf[ 'ele_expectInnermostPixelLayerHit' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_expectNextToInnermostPixelLayerHit')] = hf[ 'ele_expectNextToInnermostPixelLayerHit'][ event ][ ele ]
    try:
        data_temp[ 0, column_names.index( f'ele_nTracks')] = hf[ 'ele_nTracks'][ event ][ ele ]
    except: 
        data_temp[ 0, column_names.index( f'ele_nTracks')] = -999
    data_temp[ 0, column_names.index( f'ele_numberOfInnermostPixelHits')] = hf[ 'ele_numberOfInnermostPixelHits'][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_numberOfPixelHits')] = hf[ 'ele_numberOfPixelHits'][ event ][ ele]
    data_temp[ 0, column_names.index( f'ele_numberOfSCTHits')] = hf[ 'ele_numberOfSCTHits'][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele_numberOfTRTHits')] = hf[ 'ele_numberOfTRTHits'][ event ][ ele ]
    try:
        data_temp[ 0, column_names.index( f'ele_core57cellsEnergyCorrection')] = hf[ 'ele_core57cellsEnergyCorrection'][ event ][ ele ]
    except: 
        data_temp[ 0, column_names.index( f'ele_core57cellsEnergyCorrection')] = -999

def addMuonVariables(hf, event, data_temp, lep):
    """ Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data

    Arguments:
        hf: File to get variables from.
        event: Event Number.
        data_temp: Numpy array to add variables to.
        ele: Electron index.

    Returns: 
        Nothing. Data is set in existing array.
    """
    data_temp[ 0, column_names.index( f'muo_truthPdgId' ) ] = hf[ 'muo_truthPdgId' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_truthType' ) ] = hf[ 'muo_truthType' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_truthOrigin' ) ] = hf[ 'muo_truthOrigin' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_trigger' ) ] = hf[ 'muo_trigger' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_pt' ) ] = hf[ 'muo_pt' ][ event ][ lep ]/1000
    data_temp[ 0, column_names.index( f'muo_eta' ) ] = hf[ 'muo_eta' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_phi' ) ] = hf[ 'muo_phi' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_charge' ) ] = hf[ 'muo_charge' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_delta_z0') ] = hf[ 'muo_delta_z0' ][event][ lep ]
    data_temp[ 0, column_names.index( f'muo_delta_z0_sin_theta') ] = hf[ 'muo_delta_z0_sin_theta' ][event][ lep ]
    data_temp[ 0, column_names.index( f'muo_muonType' ) ] = hf[ 'muo_muonType' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptvarcone20' ) ] = hf[ 'muo_ptvarcone20' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptvarcone30' ) ] = hf[ 'muo_ptvarcone30' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptvarcone40' ) ] = hf[ 'muo_ptvarcone40' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptcone20' ) ] = hf[ 'muo_ptcone20' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptcone30' ) ] = hf[ 'muo_ptcone30' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_ptcone40' ) ] = hf[ 'muo_ptcone40' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_numberOfPrecisionLayers' ) ] = hf[ 'muo_numberOfPrecisionLayers' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_numberOfPrecisionHoleLayers' ) ] = hf[ 'muo_numberOfPrecisionHoleLayers' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_quality' ) ] = hf[ 'muo_quality' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_innerSmallHits' ) ] = hf[ 'muo_innerSmallHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_innerLargeHits' ) ] = hf[ 'muo_innerLargeHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_middleSmallHits' ) ] = hf[ 'muo_middleSmallHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_middleLargeHits' ) ] = hf[ 'muo_middleLargeHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_outerSmallHits' ) ] = hf[ 'muo_outerSmallHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_outerLargeHits' ) ] = hf[ 'muo_outerLargeHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_CaloLRLikelihood' ) ] = hf[ 'muo_CaloLRLikelihood' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_d0' ) ] = hf[ 'muo_priTrack_d0' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_z0' ) ] = hf[ 'muo_priTrack_z0' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_d0Sig' ) ] = hf[ 'muo_priTrack_d0Sig' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_z0Sig' ) ] = hf[ 'muo_priTrack_z0Sig' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_chiSquared' ) ] = hf[ 'muo_priTrack_chiSquared' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberDoF' ) ] = hf[ 'muo_priTrack_numberDoF' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfPixelHits' ) ] = hf[ 'muo_priTrack_numberOfPixelHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfSCTHits' ) ] = hf[ 'muo_priTrack_numberOfSCTHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfTRTHits' ) ] = hf[ 'muo_priTrack_numberOfTRTHits' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_neflowisol20' ) ] = hf[ 'muo_neflowisol20' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_MuonSpectrometerPt' ) ] = hf[ 'muo_MuonSpectrometerPt' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_etconecoreConeEnergyCorrection' ) ] = hf[ 'muo_etconecoreConeEnergyCorrection' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_InnerDetectorPt' ) ] = hf[ 'muo_InnerDetectorPt' ][ event ][ lep ]
    #data_temp[ 0, column_names.index( f'muo_author' ) ] = hf[ 'muo_author' ][ event ][ lep ]
    #data_temp[ 0, column_names.index( f'muo_allAuthors' ) ] = hf[ 'muo_allAuthors' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_scatteringCurvatureSignificance' ) ] = hf[ 'muo_scatteringCurvatureSignificance' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_scatteringNeighbourSignificance' ) ] = hf[ 'muo_scatteringNeighbourSignificance' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_momentumBalanceSignificance' ) ] = hf[ 'muo_momentumBalanceSignificance' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_EnergyLoss' ) ] = hf[ 'muo_EnergyLoss' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_energyLossType' ) ] = hf[ 'muo_energyLossType' ][ event ][ lep ]
    data_temp[ 0, column_names.index( f'muo_vertex_z' ) ] = hf[ 'vertex_z' ][ event ][ lep ]
    


def multiExtract(arguments):
    """"
    Extracts the variables of the hdf5 file produced with root2hdf5
    Method varies depending on ePid or eIso
    ePid: Determined by pdgId
    eIso: Determined by Particletype

    Arguments:
        process: process number
        counter: file counter
        path: path to file
        start: event index in file to start at
        stop: event index in file to stop at

    Returns: 
        Data of electron candidate in numpy array
        Counters for histogram over signal selection and cut flow
    """
    # Unpack arguments 
    process, counter, path, start, stop, selectiontype = arguments

    # Counters for histograms implement later
    histNtype = np.zeros((1,3))
    histTrigPass = np.zeros((1,2))
    # Read ROOT data from file
    log.info("[{}] Importing data from {}".format(process,path))
    hf = h5py.File(path,"r")

    # Numpy array to return
    data = np.empty((0,len(column_names)),float)

    # Total number of events in batch 
    n_events = stop-start

    #Run over all events in the start stop range
    for i, event in enumerate(np.arange(start, stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}] {} of {} events examined". format(process,i,n_events))

        # Number of Leptons in event
        if args.PartType == "ele":
            nLep = np.shape(hf['ele_truthType'][event])[0]
        elif args.PartType == "muo":
            try:
                nLep = np.shape(hf['muo_truthType'][event])[0]
            except:
                #log.info("It broke down at nLep at event {}". format(event))
                nLep = 0
        if (nLep>0):
            histTrigPass[0,1] += 1
            for lep in range(nLep):
                data_temp = np.zeros((1,len(column_names)))
                # Get type 
                try:
                    selection = signalSelection(hf, event, lep, selectiontype, histNtype)
                except:
                    #log.info("It broke down at signalSelection at event #{}". format(event))
                    selection = 2
                if selection == 2:
                    continue
        
                # Add event variables to array
                data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                data_temp[ 0, column_names.index( 'type' ) ] = selection
                # Add electron variables to array
                if args.PartType == "ele":
                    addElectronVariables(hf, event, data_temp, lep)
                elif args.PartType == "muo":
                    addMuonVariables(hf, event, data_temp, lep)

                # Append data to array that is returned
                try:
                    data = np.append(data, data_temp, axis = 0)
                except:
                    #log.info("It broke down when appending data at event # {}". format(event))
                    continue
        else: 
            histTrigPass[0,0] += 1
    return data, histNtype, histTrigPass

def saveToFile(fname, data, column_names, column_dtype, selection):
    """
    Simply saves data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.
        selection: ePid or eIso (0 and 1 respectively)
    Returns:
        Nothing.
    """
    if selection == 0:
        seltype = "lPid_"
    else:
        seltype = "lIso_"
    log.info("Saving to {}".format(args.outdir + seltype + fname))
    with h5py.File(args.outdir + "Lepton_" + seltype + fname, 'w') as hf:
        for var in column_names:
            hf.create_dataset( f'{var}',
                              data=data[:,column_names.index(f'{var}')],
                              dtype=column_dtype[f'{var}'],
                              chunks=True,
                              maxshape= (None,),
                              compression='lzf')

def appendToFile(fname, data, column_names, column_dtype, selection):
    """
    Simply appends data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.
        selection: ePid or eIso (0 and 1 respectively)

    Returns:
        Nothing.
    """
    if selection == 0:
        seltype = "lPid_"
    else:
        seltype = "lIso_"
    log.info("Appending to {}".format(args.outdir+ seltype + fname))
    with h5py.File(args.outdir + "Lepton_" + seltype + fname, 'a') as hf:
        for var in column_names:

            array = data[:,column_names.index(f'{var}')]
            hf[f'{var}'].resize((hf[f'{var}'].shape[0] + array.shape[0]), axis = 0)
            hf[f'{var}'][-array.shape[0]:] = array.astype(column_dtype[f'{var}'])

def plotHistogram(histVal, fname, names, title, xlabel):
    """
    Simply plots histogram and saves it.

    Arguments:
        histVal: Values on y axis.
        fname: filename (directory is taken from script's args).
        names: Names on x axis.
        title: Title of histogram.
        xlabel: xlabel of histogram.

    Returns:
        Nothing.
    """
    log.info("Plot and save histogram: {}".format(args.outdir + fname))
    # Length, position and values of data
    n = len(histVal)
    x = np.arange(n)
    val = histVal

    # Plot histogram
    fig, ax = plt.subplots()
    ax.bar(x, height=val)           # Create bar plot
    plt.xticks(x, names)            # Rename x ticks
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    # Add values to histogram
    shift = np.max(val)*0.01
    for i in range(n):
        ax.text(x[i], val[i]+shift, f"{int(val[i])}", horizontalalignment='center')

    # Save histogram
    plt.tight_layout()
    fig.savefig(args.outdir + fname)

def MassHistogram(HistVal, DatType, fname, title, xlabel, min, max, nbins):
    """
    Simply plots histogram and saves it.

    Arguments:
        histVal: Values on y axis.
        fname: filename (directory is taken from script's args).
        title: Title of histogram.
        xlabel: xlabel of histogram.
        min: Minimum mass value
        max: maximum mass value
        nbins: The number of bins for the histogram
        

    Returns:
        Nothing.
    """
    log.info("Plot and save histogram of masses/Transverse energy: {}".format(args.outdir + fname))
    # Length, position and values of data

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    
    # Add Values to histogram 
    ax.hist(HistVal[DatType==1],bins=np.linspace(min,max,nbins),histtype="step", label = "Signal")
    ax.hist(HistVal[DatType==0],bins=np.linspace(min,max,nbins),histtype="step", label = "Background")

    ax.legend()

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(args.outdir+fname)

#============================================================================
# Define column names and dtypes (Used when saving) Opdater til single electron
#============================================================================
if args.PartType == "ele":
    column_dtype = {"NvtxReco" : int,
                    "correctedScaledAverageMu" : float,
                    "type" : int,
                    "ele_truthParticle" : int,
                    "ele_truthPdgId" : int,
                    "ele_truthType" : int,
                    "ele_truthOrigin" : int,
                    "ele_pdf_score" : float,
                    #"ele_ECIDSResult" : float,
                    "ele_charge" : float,
                    "ele_d0Sig" : float,
                    "ele_d0" : float,
                    "ele_z0" : float,
                    "ele_z0Sig" : float,
                    "ele_eta" : float,
                    "ele_et" : float,
                    "ele_m" : float,
                    "ele_e" : float,
                    "ele_phi" : float,
                    "ele_f3": float,
                    "ele_f1": float,
                    "ele_Reta": float,
                    "ele_Rphi": float,
                    "ele_Rhad": float,
                    "ele_Rhad1": float,
                    "ele_weta2": float,
                    "ele_TRTPID": int,
                    "ele_dPOverP": float,
                    "ele_Eratio": float,
                    "ele_LHLoose" : int,
                    "ele_LHMedium" : int,
                    "ele_LHTight" : int,
                    "ele_trigger" : int,
                    "ele_deltaEta1": float,
                    "ele_deltaPhiRescaled2": float,
                    "ele_ptvarcone20" : float,
                    "ele_topoetcone20" : float,
                    "ele_topoetcone40" : float,
                    "ele_ptcone20_TightTTVALooseCone_pt1000" : float,
                    "ele_ptcone20_TightTTVALooseCone_pt500" : float,
                    "ele_ptvarcone20_TightTTVALooseCone_pt1000" : float,
                    #"ele_ptvarcone20_TightTTVA_pt1000" : float,
                    "ele_ptvarcone30_TightTTVALooseCone_pt1000" : float,
                    "ele_ptvarcone30_TightTTVALooseCone_pt500" : float,
                    #"ele_ptvarcone30_TightTTVA_pt1000" : float,
                    #"ele_ptvarcone30_TightTTVA_pt500" : float,
                    "ele_ptvarcone40_TightTTVALooseCone_pt1000" : float,
                    "ele_topoetcone20ptCorrection" : float,
                    "ele_expectInnermostPixelLayerHit" : int,
                    "ele_expectNextToInnermostPixelLayerHit" : int,
                    "ele_nTracks" : int,
                    "ele_numberOfInnermostPixelHits" : int,
                    "ele_numberOfPixelHits" : int,
                    "ele_numberOfSCTHits" : int,
                    "ele_numberOfTRTHits" : int,
                    "ele_core57cellsEnergyCorrection" : float}
elif args.PartType == "muo":  
    column_dtype = {"NvtxReco" : int,
                    "correctedScaledAverageMu" : float,
                    "type" : int,
                    "muo_truthPdgId" : int,
                    "muo_truthType" : int,
                    "muo_truthOrigin" : int,
                    #"ele_pdf_score" : float,
                    #"ele_ECIDSResult" : float,
                    "muo_charge" : float,
                    "muo_LHLoose" : int,
                    "muo_LHMedium" : int,
                    "muo_LHTight" : int,
                    "muo_trigger" : int,
                    'muo_pt' : float,
                    "muo_phi" : float,
                    'muo_eta' : float,
                    'muo_delta_z0' : float,
                    'muo_delta_z0_sin_theta' : float,
                    'muo_muonType' : int,
                    'muo_ptvarcone20': float,
                    'muo_ptvarcone30': float,
                    'muo_ptvarcone40': float,
                    'muo_ptcone20' : float,
                    'muo_ptcone30' : float,
                    'muo_ptcone40' : float,
                    'muo_numberOfPrecisionLayers' : int,
                    'muo_numberOfPrecisionHoleLayers' : int,
                    'muo_quality' : float,
                    'muo_innerSmallHits' : int,
                    'muo_innerLargeHits' : int,
                    'muo_vertex_z' : float,
                    'muo_middleSmallHits' : int,
                    'muo_middleLargeHits' : int,
                    'muo_outerSmallHits' : int,
                    'muo_outerLargeHits' : int,
                    'muo_CaloLRLikelihood' : float,
                    'muo_priTrack_d0' : float,
                    'muo_priTrack_z0' : float,
                    'muo_priTrack_d0Sig' : float,
                    'muo_priTrack_z0Sig' : float, 
                    'muo_priTrack_chiSquared' : float,
                    'muo_priTrack_numberDoF': int,
                    'muo_priTrack_numberOfPixelHits' : int,
                    'muo_priTrack_numberOfSCTHits' : int,
                    'muo_priTrack_numberOfTRTHits' : int,
                    'muo_neflowisol20' : float, 
                    'muo_MuonSpectrometerPt': float,
                    'muo_etconecoreConeEnergyCorrection': float,
                    'muo_InnerDetectorPt': float,
                    #'muo_author' : str,
                    #'muo_allAuthors': str,
                    'muo_scatteringCurvatureSignificance' : float,
                    'muo_scatteringNeighbourSignificance' : float,
                    'muo_momentumBalanceSignificance' : float,
                    'muo_EnergyLoss' : float,
                    'muo_energyLossType' : int}                




column_names = list(column_dtype.keys())



#============================================================================
# Main
#============================================================================
# Total counters for signal selection diagram
#histNtype_total = np.zeros((1,5))
#histTrigPass_total = np.zeros((1,2))
if args.PartType == "ele":
    ePid_et, eIso_et = [], []
elif args.PartType == "muo":
    mPid_pt, mIso_pt = [], []
lPid_SBT, lIso_SBT = [], []
histNtype_total = np.zeros((1,3))
histTrigPass_total = np.zeros((1,2))
# create file name and check if the file already exists
filename = '{:s}.h5'.format(args.tag)
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file already exists - please remove yourself. Output: {args.outdir + filename}")
    quit()

# Make a pool of processes (This must come after the functions needed to run over since it apparently imports _main_here)
pool = multiprocessing.Pool(processes=args.max_processes)
log.info("============================================================================")
log.info("Starting lepton Pid")
log.info("============================================================================")
if args.selection == 0 or args.selection == 2:
    for path in args.paths:
        # Count which files made it to
        counter += 1

        # Read hdf5 data to get number of events
        hf_read = h5py.File(path, "r")

        print(hf_read.keys())
        
        N = hf_read["NvtxReco"].shape[0]

        print("N= ", N)

        # Split indices into equally-sized batches
        index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
        index_ranges = zip(index_edges[:-1], index_edges[1:])
        lPidresults = pool.map(multiExtract, [(i, counter, path, start, stop, 0) for i, (start, stop) in enumerate(index_ranges)])
        lPidresults_np = np.array(lPidresults)

        # Concatenate resulting data from the multiple converters
        lPiddata = np.concatenate(lPidresults_np[:,0])
        log.info("{} PidData shape: {}".format(args.PartType, lPiddata.shape))
        if args.PartType == "ele":
            ePid_et = np.append(ePid_et,lPiddata[:,column_names.index('ele_et')])
        elif args.PartType == "muo":
            mPid_pt = np.append(mPid_pt,lPiddata[:,column_names.index('muo_pt')])
        lPid_SBT = np.append(lPid_SBT,lPiddata[:,column_names.index("type")])
        
         # Concatenate data and add to total
        histNtype = np.concatenate(lPidresults_np[:,1], axis = 0)
        histNtype_total = histNtype_total + np.sum(histNtype, axis = 0)

        # Concatenate data and add tot total
        histTrigPass = np.concatenate(lPidresults_np[:,2], axis = 0)
        histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis = 0)
        # Save output to a file
        if counter == 0:
            saveToFile(filename, lPiddata, column_names, column_dtype, 0)
        else: 
            appendToFile(filename, lPiddata, column_names, column_dtype, 0)
log.info("============================================================================")
log.info("Starting lepton Iso")
log.info("============================================================================")
counter = -1
if args.selection == 1 or args.selection == 2:
    for path in args.paths:
        # Count which files made it to
        counter += 1

        # Read hdf5 data to get number of events
        hf_read = h5py.File(path, "r")

        print(hf_read.keys())
        
        N = hf_read["NvtxReco"].shape[0]

        print("N= ", N)

        # Split indices into equally-sized batches
        index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
        index_ranges = zip(index_edges[:-1], index_edges[1:])
        lIsoresults = pool.map(multiExtract, [(i, counter, path, start, stop, 1) for i, (start, stop) in enumerate(index_ranges)])
        lIsoresults_np = np.array(lIsoresults)
        # Concatenate resulting data from the multiple converters
        lIsodata = np.concatenate(lIsoresults_np[:,0])
        log.info("lepton IsoData shape: {}".format(lIsodata.shape))
        if args.PartType == "ele":
            eIso_et = np.append(eIso_et,lIsodata[:,column_names.index("ele_et")])
        elif args.PartType == "muo":
            mIso_pt = np.append(mIso_pt,lIsodata[:,column_names.index("muo_pt")])
        lIso_SBT = np.append(lIso_SBT,lIsodata[:,column_names.index("type")])
        # Concatenate data and add to total
        histNtype = np.concatenate(lIsoresults_np[:,1], axis = 0)
        histNtype_total = histNtype_total + np.sum(histNtype, axis = 0)

        # Concatenate data and add tot total
        histTrigPass = np.concatenate(lIsoresults_np[:,2], axis = 0)
        histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis = 0)
        # Save output to a file
        if counter == 0:
            saveToFile(filename, lIsodata, column_names, column_dtype, 1)
        else: 
            appendToFile(filename, lIsodata, column_names, column_dtype, 1)

plotHistogram(histVal=histNtype_total[0],
              fname=args.tag+"_sigSelDiag.png",
              names=[ "Background","Signal", "Trash"],
              title = f"Signal selection ({args.tag})",
              xlabel = "Selection types")
plotHistogram(histVal=histTrigPass_total[0],
              fname=args.tag+"_trigPassDiag.png",
              names=["No trigger in event", "At least one trigger in event"],
              title = f"Events that passes trigger ({args.tag})",
              xlabel = "")
if args.PartType == "ele":
    if (args.selection == 0) or (args.selection == 2):
        MassHistogram(HistVal = ePid_et, 
                DatType = lPid_SBT,
                fname = args.tag+"_ePidet.png", 
                title = "Transverse energy of the electron", 
                xlabel = "Transverse energy", 
                min = 1, 
                max = 150, 
                nbins = 151-1)
    if (args.selection == 1) or (args.selection == 2):
        MassHistogram(HistVal = eIso_et, 
                DatType = lIso_SBT,
                fname = args.tag+"_eIsoet.png", 
                title = "Transverse energy of the electron", 
                xlabel = "Transverse energy", 
                min = 1, 
                max = 150, 
                nbins = 151-1)
elif args.PartType == "muo":
    if (args.selection == 0) or (args.selection == 2):
        MassHistogram(HistVal = mPid_pt, 
                DatType = lPid_SBT,
                fname = args.tag+"_mPidpt.png", 
                title = "Transverse energy of the muon", 
                xlabel = "Transverse energy", 
                min = 1, 
                max = 150, 
                nbins = 151-1)
    if (args.selection == 1) or (args.selection == 2):
        MassHistogram(HistVal = mIso_pt, 
                DatType = lIso_SBT,
                fname = args.tag+"_mIsopt.png", 
                title = "Transverse energy of the muon", 
                xlabel = "Transverse energy", 
                min = 1, 
                max = 150, 
                nbins = 151-1)
