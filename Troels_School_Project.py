#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Author: Mads Juhl Storr-Hansen
This code is used for the signal selection of H->gam(*)gam
This will later be extended to a reweigh function
"""
print("Program running")

import warnings
warnings.filterwarnings('ignore', 'ROOT .+ is currently active but you ')
warnings.filterwarnings('ignore', 'numpy .+ is currently installed but you ')

import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt

from utils import mkdir, h5ToDf
from itertools import combinations
from skhep.math import vectors
import multiprocessing
import csv
import lightgbm as lgb
import pandas as pd

from time import time
from datetime import timedelta

from returnLGBMScore import pIsoScore, pPidScore
# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for training.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/SignalFiles/Troels_project/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--nJobs', action='store', type=int, required=False, default=5,
                    help='Number of jobs (Default 5)')
#parser.add_argument('--phoOrigin', action='store', default=14, type=int,
#                    help='Photon origin used to detirmine signal. (Default = 3 [single photon])')
#parser.add_argument('--IsMC', action = 'store', type = int, required= False, default  = 1,
#                    help = 'Is the file MC or Data (1,0), default is MC' )

args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error("No HDF5 file was specified.")
    quit()

if args.max_processes > 20:
    log.error("The requested number of processes ({}) is excessive (>20). Exiting.".format(args.max_processes))
    quit()


#Makes an output directory, and gives an error if it exists.
args.outdir = args.outdir+f"{args.tag}/"#_{nameSigBkg}/"
if os.path.exists(args.outdir):
    log.error(f"Output already exists - please remove yourself. Output: {args.outdir}")
    quit()
else:
    log.info(f"Creating output folder: {args.outdir}")
    mkdir(args.outdir)

# File number counter (incremented first in loop)
counter = -1
pIsoModel = "output/Training/models/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23/phoIsomodel_LGBM_phoIsophoIso_phiReg10Iso230921_regWeight_nEst10_2021-09-23.txt"
pPidModel = "output/Training/models/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29/phoPidmodel_LGBM_phoPidphoPid_phoReg40Pid290921_regWeight_nEst40_2021-09-29.txt"

#============================================================================
# Functions
#============================================================================


def getTagsAndProbes(hf, event, process):
    """
    Get tag and probe indices in process
    Input: 
        hf: File to get variables from
        event: Event number
    Output:
        eTag: Array of tag leptons.
        eProbe: Array of probe leptons.
        pProbe: Array of probe photons. 
    """
    

    #Get photon probes
    
    pho = []
    for ipho in range(len(hf["pho_et"][event])):
        if hf["pho_et"][event][ipho]>10:
            pho.append(ipho)
    return pho
    
def addPhotonVariables(hf, event, data_temp, phoNr, pho):
    """
    Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        data_temp: Numpy array to add variables to.
        pho: Photon index.

    Returns:
        Nothing. Data is set in existing array.
    """

    #data_temp[ 0, column_names.index( f'pho{phoNr}_truthPdgId_egam') ] = hf[ 'pho_truthPdgId_egam' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_truthPdgId_atlas') ] = hf[ 'pho_truthPdgId_atlas' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_egamTruthParticle') ] = hf[ 'pho_egamTruthParticle' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_truthType') ] = hf[ 'pho_truthType' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_truthOrigin') ] = hf[ 'pho_truthOrigin' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_isPhotonEMLoose') ] = hf[ 'pho_isPhotonEMLoose' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_isPhotonEMTight') ] = hf[ 'pho_isPhotonEMTight' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f"pho{phoNr}_et_truth" ) ] = np.sqrt(hf["pho_truth_px_atlas"][ event ][ pho ]**2+hf["pho_truth_py_atlas"][ event ][ pho ]**2)
    data_temp[ 0, column_names.index( f"pho{phoNr}_eta_truth" ) ] = 0.5*np.log((hf["pho_truth_E_atlas"][ event ][ pho ]+hf["pho_truth_pz_atlas"][ event ][ pho ])/(hf["pho_truth_E_atlas"][ event ][ pho ]-hf["pho_truth_pz_atlas"][ event ][ pho ]))
    data_temp[ 0, column_names.index( f"pho{phoNr}_phi_truth" ) ] = np.arcsin(hf["pho_truth_py_atlas"][ event ][ pho ]/np.sqrt(hf["pho_truth_px_atlas"][ event ][ pho ]**2+hf["pho_truth_py_atlas"][ event ][ pho ]**2))
    data_temp[ 0, column_names.index( f'pho{phoNr}_e') ] = hf[ 'pho_e' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_eta') ] = hf[ 'pho_eta' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_phi') ] = hf[ 'pho_phi' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_et') ] = hf[ 'pho_et' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_pt') ] = hf[ 'pho_pt' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_Rhad1') ] = hf[ 'pho_Rhad1' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_Rhad') ] = hf[ 'pho_Rhad' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_weta2') ] = hf[ 'pho_weta2' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_Rphi') ] = hf[ 'pho_Rphi' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_Reta') ] = hf[ 'pho_Reta' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_Eratio') ] = hf[ 'pho_Eratio' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_f1') ] = hf[ 'pho_f1' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_wtots1') ] = hf[ 'pho_wtots1' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_DeltaE') ] = hf[ 'pho_DeltaE' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_weta1') ] = hf[ 'pho_weta1' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_fracs1') ] = hf[ 'pho_fracs1' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_ConversionRadius') ] = hf[ 'pho_ConversionRadius' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_VertexConvEtOverPt') ] = hf[ 'pho_VertexConvEtOverPt' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_VertexConvPtRatio') ] = hf[ 'pho_VertexConvPtRatio' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_topoetcone20') ] = hf[ 'pho_topoetcone20' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( f'pho{phoNr}_topoetcone30') ] = hf[ 'pho_topoetcone30' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_topoetcone40') ] = hf[ 'pho_topoetcone40' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_ptvarcone20') ] = hf[ 'pho_ptvarcone20' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_ptvarcone30') ] = hf[ 'pho_ptvarcone30' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_ptvarcone40') ] = hf[ 'pho_ptvarcone40' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_z0') ] = hf[ 'pho_z0' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_z0Sig') ] = hf[ 'pho_z0Sig' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_maxEcell_time') ] = hf[ 'pho_maxEcell_time' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_maxEcell_energy') ] = hf[ 'pho_maxEcell_energy' ][ event ][ pho ]
    data_temp[ 0, column_names.index( f'pho{phoNr}_r33over37allcalo') ] = hf[ 'pho_r33over37allcalo' ][ event ][ pho ]
    #data_temp[ 0, column_names.index( 'pho{phoNr}_GradientIso') ] = hf[ 'pho_GradientIso' ][ event ][ pho ]

def signalSelection(hf, event, tag, probe, histNtype):
    '''
    Selects a type for the given lepton pair based on a flowchart.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        lTag: Index of tag photon.
        lProbe: Index of probe photon.

    Returns:
        The type of the pair:
        0 = background
        1 = signal
    '''
    # Do the leptons originate from a Z boson?
    # Do the leptons have opposite charge?
    isHgg_0 = hf['pho_truthOrigin'][event][tag]==14
    isHgg_1 = hf['pho_truthOrigin'][event][probe] == 14
    gam0 = np.abs(hf['pho_truthPdgId_atlas'][event][tag])==22
    gam1 = np.abs(hf['pho_truthPdgId_atlas'][event][probe])==22
    iso0 = hf['pho_truthType'][event][tag] == 14 #Is it an isolated photon
    iso1 = hf['pho_truthType'][event][probe]==14

    # test2
    """if (isHgg_0*isHgg_1):
        if (gam0*gam1)*(iso0*iso1):
            histNtype[0,1] += 1  
            return 1 # Signal
        elif (gam0*gam1):
            histNtype[0,0] +=1
            return 0 #Background
        else:
            histNtype[0,2] += 1  
            return 2 # trash
    else: #It seems like there are no cases with only one or 
        if (isHgg_0+isHgg_1)<2:
            histNtype[0,0] += 1  
            return 0 # Background"""
    #Everything is  signal - no background. test
    if (isHgg_0*isHgg_1):
        if (gam0*gam1):#*(iso0*iso1):
            histNtype[0,1] += 1  
            return 1 # Signal
        else:
            histNtype[0,2] += 1  
            return 2 # trash
    else: 
        if (isHgg_0+isHgg_1)<2:
            histNtype[0,0] += 1  
            return 0 # Background
        

def combinedVariables(hf, event, phoTag, phoProbe):
    """
    Calculate variables of the lepton pair and photon.
    It should be noted that electrons' mass and energy
    is given in GeV while they are given as MeV for 
    muons. Therefore the multiplication and divisions by 1000.
    Arguments:
        hf: File to get variables from.
        event: Event number.
        lTag: Index of tag lepton.
        lProbe: Index of probe lepton.
        phoProbe: Index of probe photon.

    Returns:
        invM: Invariant mass of combined four-vector.
        et: Transverse energy of combined four-vector.
        eta: Eta of combined four-vector.
        phi: Phi of combined four-vector.
    """
    # Calculate invM etc. of eegamma using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variable
    
    p1 = hf['pho_et'][event][phoTag]
    eta1 = hf['pho_eta'][event][phoTag]
    phi1 = hf['pho_phi'][event][phoTag]
    p2 = hf[f'pho_et'][event][phoProbe]
    eta2 = hf[f'pho_eta'][event][phoProbe]
    phi2 = hf[f'pho_phi'][event][phoProbe]
    
    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()
    vecFour1.setptetaphim(p1,eta1,phi1,0)
    vecFour2.setptetaphim(p2,eta2,phi2,0)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2
    invM = vecFour.mass
    et = vecFour.et
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, et, eta, phi


def multiExtract(arguments):
    """
    Extracts the variable from the Hdf5 file with root2hdf5
    Using tag and probe:
        tag = lepton that has triggered
        probe = any other lepton of same generation in container
    selects signal type based on pdgid, truthorigin and truthparticle
    selection: 0=bkg, 1=sig,2=trash.

    Arguments:
        Process: process number
        counter: file counter
        path: path to file
        start: event index in file to start at
        stop: event index in file to stop at

    Returns:
        Data of lepton pairs in numpy array
        Counters for histogram over signal selection and cut flow
    """
    # Unpack arguments
    process, counter, path, start, stop, pIsoGBM, pPidGBM = arguments

    # Counters for histograms
    histNtype = np.zeros((1,3))
    histTrigPass = np.zeros((1,2))
    # Read ROOT data from file
    log.info("[{}]  Importing data from {}".format(process,path))
    hf = h5py.File(path,"r")

    # Numpy array to return
    data = np.empty((0,len(column_names)), float)

    # Total number of events in batch
    n_events = stop-start
    NHPhotons = []
    # Run over all events in the start stop range
    for i, event  in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}] {} of {} events examined".format(process, i, n_events))
        
        # Get tags and probes
        try:
            pho = getTagsAndProbes(hf,event, process)
            npho = len(pho)
        except: 
            npho = 0
        PHO = []
        #Matches index with photon et
        for i in range(npho):
            PHO.append([pho[i],hf["pho_et"][event][pho[i]]])
        #Sorts index so highest photon et first, needed since the first two photons are signal
        sortpho = sorted(PHO, key=lambda l:l[1],reverse=True)
        sortPho = [phonr[0] for phonr in sortpho]
        nHiggphotons = sum(hf['pho_truthOrigin'][event][:]==14)
        NHPhotons.append(nHiggphotons)
        # If there is at least one tag, 2 photons -> get values
        if npho>1:
            histTrigPass[0,1] += 1
            for iPhoprim, Phoprim in enumerate(sortPho):

                phosecond = sortPho[iPhoprim+1:].copy()
                #print(f"Phoprim: {sortPho}, phoSecond: {phosecond}")
                for iPhosecond, Phosecond in enumerate(phosecond):
                    selection = signalSelection(hf, event, Phoprim, Phosecond, histNtype)
                    if selection == 2:
                        continue

                    data_temp = np.zeros((1,len(column_names)))
                    
                    invM, et, eta, phi = combinedVariables(hf,event, Phoprim,Phosecond)
                    data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                    data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                    data_temp[ 0, column_names.index( "eventNr")] = event
                    data_temp[ 0, column_names.index( 'invM' ) ] = invM
                    data_temp[ 0, column_names.index( 'et' ) ] = et
                    data_temp[ 0, column_names.index( 'eta' ) ] = eta
                    data_temp[ 0, column_names.index( 'phi' ) ] = phi
                    data_temp[ 0, column_names.index( 'type' ) ] = selection
                    # Add electron variables to array
                    addPhotonVariables(hf, event, data_temp, 1, Phoprim)
                    addPhotonVariables(hf, event, data_temp, 2, Phosecond)

                    df = pd.DataFrame(data_temp, columns = column_names)
                    data_temp[ 0, column_names.index( 'pho1_pIso_score')] = pIsoScore(pIsoGBM, df,1, n_jobs = args.nJobs)
                    data_temp[ 0, column_names.index( 'pho1_pPid_score')] = pPidScore(pPidGBM, df,1, n_jobs = args.nJobs)
                    data_temp[ 0, column_names.index( 'pho2_pIso_score')] = pIsoScore(pIsoGBM, df,2, n_jobs = args.nJobs)
                    data_temp[ 0, column_names.index( 'pho2_pPid_score')] = pPidScore(pPidGBM, df,2, n_jobs = args.nJobs)

                    # Append data to array that is returned
                    data = np.append(data, data_temp, axis=0)       
        else: 
            # Event did not pas trigger cut - add to histogram
            histTrigPass[0,0] += 1
    return data, histNtype, histTrigPass, NHPhotons

def saveToFile(fname, data, column_names, column_dtype):
    """
    Simply saves data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    """
    log.info("Saving to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'w') as hf:
        for var in column_names:
            hf.create_dataset( f'{var}',
                              data=data[:,column_names.index(f'{var}')],
                              dtype=column_dtype[f'{var}'],
                              chunks=True,
                              maxshape= (None,),
                              compression='lzf')
    header_names = ', '.join([str(elem) for elem in column_names])
    np.savetxt(args.outdir+args.tag+".csv",data,delimiter=',',header=header_names)

def appendToFile(fname, data, column_names, column_dtype):
    """
    Simply appends data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    """
    log.info("Appending to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'a') as hf:
        for var in column_names:

            array = data[:,column_names.index(f'{var}')]
            hf[f'{var}'].resize((hf[f'{var}'].shape[0] + array.shape[0]), axis = 0)
            hf[f'{var}'][-array.shape[0]:] = array.astype(column_dtype[f'{var}'])
    with open(args.outdir+args.tag+".csv","w") as f:
        writer=csv.writer(f)
        writer.writerows(data)
        


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
    log.info("Plot and save histogram of masses: {}".format(args.outdir + fname))
   
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    
    # Add Values to histogram 
    ax.hist(HistVal[DatType==1],bins=np.linspace(min,max,nbins),histtype="step", label = "Signal")
    ax.hist(HistVal[DatType == 0],bins=np.linspace(min,max,nbins),histtype="step", label = "Background")

    ax.legend()

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(args.outdir+fname)
    
    

    
def MassVsMassHistogram(LLMass, LLGamMass, fname, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    nbins = np.linspace(10,150,141)
    ax.hist2d(LLGamMass,LLMass,bins=[nbins,nbins],cmin=1)
    fig.savefig(args.outdir+fname)
#============================================================================
# Define column names and dtypes (Used when saving)
#============================================================================
column_dtype = {"NvtxReco" : int,
                "correctedScaledAverageMu" : float,
                "eventNr": int,
                "invM" : float,
                "et" : float,
                "eta" : float,
                "phi" : float,
                "type" : int,
                "pho1_pIso_score": float,
                "pho1_pPid_score": float,
                "pho1_truthPdgId_egam" : int,
                "pho1_truthPdgId_atlas" : int,
                #"pho1_egamTruthParticle" : int,
                "pho1_truthType" : int,
                "pho1_truthOrigin" : int,
                "pho1_isPhotonEMLoose" : int,
                "pho1_isPhotonEMTight" : int,
                "pho1_et_truth" : float,
                "pho1_eta_truth" : float,
                "pho1_phi_truth" : float,
                "pho1_e" : float,
                "pho1_eta" : float,
                "pho1_phi" : float,
                "pho1_et" : float,
                "pho1_pt" : float,
                "pho1_Rhad1" : float,
                "pho1_Rhad" : float,
                "pho1_weta2" : float,
                "pho1_Rphi" : float,
                "pho1_Reta" : float,
                "pho1_Eratio" : float,
                "pho1_f1" : float,
                "pho1_wtots1" : float,
                "pho1_DeltaE" : float,
                "pho1_weta1" : float,
                "pho1_fracs1" : float,
                "pho1_ConversionType" : float,
                "pho1_ConversionRadius" : float,
                "pho1_VertexConvEtOverPt" : float,
                "pho1_VertexConvPtRatio" : float,
                "pho1_topoetcone20" : float,
                "pho1_topoetcone30" : float,
                "pho1_topoetcone40" : float,
                "pho1_ptvarcone20" : float,
                "pho1_ptvarcone30" : float,
                "pho1_ptvarcone40" : float,
                "pho1_z0" : float,
                "pho1_z0Sig" : float,
                #'pho1_maxEcell_time': float,
                #'pho1_maxEcell_energy': float,
                'pho1_core57cellsEnergyCorrection': float,
                'pho1_r33over37allcalo': float,
                "pho2_pIso_score": float,
                "pho2_pPid_score": float,
                "pho2_truthPdgId_egam" : int,
                "pho2_truthPdgId_atlas" : int,
                #"pho2_egamTruthParticle" : int,
                "pho2_truthType" : int,
                "pho2_truthOrigin" : int,
                "pho2_isPhotonEMLoose" : int,
                "pho2_isPhotonEMTight" : int,
                "pho2_et_truth" : float,
                "pho2_eta_truth" : float,
                "pho2_phi_truth" : float,
                "pho2_e" : float,
                "pho2_eta" : float,
                "pho2_phi" : float,
                "pho2_et" : float,
                "pho2_pt" : float,
                "pho2_Rhad1" : float,
                "pho2_Rhad" : float,
                "pho2_weta2" : float,
                "pho2_Rphi" : float,
                "pho2_Reta" : float,
                "pho2_Eratio" : float,
                "pho2_f1" : float,
                "pho2_wtots1" : float,
                "pho2_DeltaE" : float,
                "pho2_weta1" : float,
                "pho2_fracs1" : float,
                "pho2_ConversionType" : float,
                "pho2_ConversionRadius" : float,
                "pho2_VertexConvEtOverPt" : float,
                "pho2_VertexConvPtRatio" : float,
                "pho2_topoetcone20" : float,
                "pho2_topoetcone30" : float,
                "pho2_topoetcone40" : float,
                "pho2_ptvarcone20" : float,
                "pho2_ptvarcone30" : float,
                "pho2_ptvarcone40" : float,
                "pho2_z0" : float,
                "pho2_z0Sig" : float,
                #'pho2_maxEcell_time': float,
                #'pho2_maxEcell_energy': float,
                'pho2_core57cellsEnergyCorrection': float,
                'pho2_r33over37allcalo': float
                #'pho2_GradientIso': float,
                #'pho2_GradientIso': float, }
}

column_names = list(column_dtype.keys())


#============================================================================
# Main Signal selection
#============================================================================
# Total counters for signal selection diagram
histNtype_total = np.zeros((1,3))
histTrigPass_total = np.zeros((1,2))
All_masses, All_SBT= [], []
All_nhphoton = []


# Create file name and check if the file already exists
filename = f"{args.tag}.h5"
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file already exists - please remove yourself")
    quit()

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports _main_here)
pool = multiprocessing.Pool(processes = args.max_processes)

log.info(f"Loading ML_pIso, ML_pPid and photon pair model.")

log.info(f"        Photon models:   {pIsoModel}")
log.info(f"                         {pPidModel}")

pIsoGBM = lgb.Booster(model_file = pIsoModel)
pPidGBM = lgb.Booster(model_file = pPidModel)
for path in args.paths:
    # Count which file we made it to
    counter +=1

    # Read hdf5 data to get number of events
    hf_read = h5py.File(path,"r")

    print(hf_read.keys())

    # Number of reconstructed verteces
    N = hf_read['NvtxReco'].shape[0]

    print("N= ", N)

    # Split indices into equally-sized batches
    index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
    index_ranges = zip(index_edges[:-1], index_edges[1:])
    
    results = pool.map(multiExtract, [(i, counter, path, start, stop, pIsoGBM, pPidGBM) for i, (start, stop) in enumerate(index_ranges)])
    results_np = np.array(results)

    # Concatenate resulting data from the multiple converters
    data = np.concatenate(results_np[:,0])
    All_masses = np.append(All_masses,data[:,column_names.index("invM")])
    All_SBT = np.append(All_SBT,data[:,column_names.index("type")])
    All_nhphoton = np.concatenate(results_np[:,3])
    # Concatenate data and add to total
    histNtype = np.concatenate(results_np[:,1], axis = 0)
    histNtype_total = histNtype_total + np.sum(histNtype, axis = 0)

    # Concatenate data and add tot total
    histTrigPass = np.concatenate(results_np[:,2], axis = 0)
    histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis = 0)

    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))
    
    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)
    
# Create and save figure of signal selection
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

MassHistogram(HistVal = All_masses, 
              DatType = All_SBT,
              fname = args.tag+"_InvMass.png", 
              title = f"Invariant mass of the photon pair", 
              xlabel = "Invariant mass", 
              min = 60, 
              max = 160, 
              nbins = 161-60)

fig, ax = plt.subplots()
ax.hist(All_nhphoton,bins=np.linspace(0,20,21))
ax.set_xlabel("Number of Higgs photons candidates")
ax.set_ylabel("Frequency")
ax.set_title("Number of Photon candidates in each event")
fig.savefig(args.outdir+"numHiggphoton.png")
sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")

