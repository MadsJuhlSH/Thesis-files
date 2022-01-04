#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select photons from dataset, Bliver der faktisk udvalgt signal og baggrund?
"""
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
import multiprocessing

from time import time
from datetime import timedelta

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for PID and ISO.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/SignalFiles/pho/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='Hdf5 file(s) to be converted.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--phoOrigin', action='store', default=3, type=int,
                    help='Photon origin used to detirmine signal. (Default = 3 [single photon])')
parser.add_argument('--selection', action = 'store', default = 2, type = int,
                    help = 'Selects selection, 0 = pPid, 1 = pIso, 2 = both')

args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error("No HDF5 file was specified.")
    quit()

if args.max_processes > 20:
    log.error("The requested number of processes ({}) is excessive (>20). Exiting.".format(args.max_processes))
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

#============================================================================
# Functions
#============================================================================
def signalSelection(hf, event, pho, histNtype, Selectiontype):
    """
    Selects photons from photon candidates
    
    Arguments:
        hf: File to get variables from
        event: Event number
        ele: Photon candidate
        Selectiontype: Pid, Iso or both (0, 1, 2)
        

    Returns:
        The type of the photon:
        0 = background
        1 = signal
        2 = trash
    """
    origInput = (hf['pho_truthOrigin'][event][pho] == args.phoOrigin)
    pdgId22 = (np.abs(hf['pho_truthPdgId_egam'][event][pho]) == 22)
    if Selectiontype == 0:   #Pid
        if (hf['pho_truthType'][event][pho] == 14 or hf['pho_truthType'][event][pho] == 15):   #Is the photon an isolated photon?
            histNtype[0,1] += 1
            return 1 # Signal
        elif (hf['pho_truthType'][event][pho] == 16 or hf['pho_truthType'][event][pho] == 2 or hf['pho_truthType'][event][pho] == 3 or hf['pho_truthType'][event][pho] == 0 or hf['pho_truthType'][event][pho] == 35 or hf['pho_truthType'][event][pho] == 36 or hf['pho_truthType'][event][pho] == 38): #Is the photon a bkg photon?
            histNtype[0,0] += 1
            return 0 # bkg
        else:
            histNtype[0,2] += 1
            return 2 # Trash
    elif Selectiontype == 1: #Iso
        if (hf['pho_truthType'][event][pho] == 14 or hf['pho_truthType'][event][pho] == 2):   # Is the photon an isolated photon?
            histNtype[0,1] += 1
            return 1 # signal
        # Prøv også 0, 1, 3, 4, 
        elif (hf['pho_truthType'][event][pho] == 16 or hf['pho_truthType'][event][pho] == 13 or hf['pho_truthType'][event][pho] == 15 or hf['pho_truthType'][event][pho] == 3 or hf['pho_truthType'][event][pho] == 17 or hf['pho_truthType'][event][pho] == 35 or hf['pho_truthType'][event][pho] == 36 or hf['pho_truthType'][event][pho] == 38):# or (hf['pho_truthType'][event][pho] == 16): # Is the photon unknown, noniso or bkg
            histNtype[0,0] += 1
            return 0
        else: 
            histNtype[0,2] += 1
            return 2

def addPhotonVariables(hf, event, data_temp, pho):
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
    data_temp[ 0, column_names.index( 'pho_truthPdgId_egam') ] = hf[ 'pho_truthPdgId_egam' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_truthPdgId_atlas') ] = hf[ 'pho_truthPdgId_atlas' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_egamTruthParticle') ] = hf[ 'pho_egamTruthParticle' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_truthType') ] = hf[ 'pho_truthType' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_truthOrigin') ] = hf[ 'pho_truthOrigin' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_isPhotonEMLoose') ] = hf[ 'pho_isPhotonEMLoose' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_isPhotonEMTight') ] = hf[ 'pho_isPhotonEMTight' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_e') ] = hf[ 'pho_e' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_eta') ] = hf[ 'pho_eta' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_pt') ] = hf[ 'pho_pt' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_phi') ] = hf[ 'pho_phi' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_et') ] = hf[ 'pho_et' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rhad1') ] = hf[ 'pho_Rhad1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rhad') ] = hf[ 'pho_Rhad' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_weta2') ] = hf[ 'pho_weta2' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rphi') ] = hf[ 'pho_Rphi' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Reta') ] = hf[ 'pho_Reta' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Eratio') ] = hf[ 'pho_Eratio' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_f1') ] = hf[ 'pho_f1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_wtots1') ] = hf[ 'pho_wtots1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_DeltaE') ] = hf[ 'pho_DeltaE' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_weta1') ] = hf[ 'pho_weta1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_fracs1') ] = hf[ 'pho_fracs1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ConversionType') ] = hf[ 'pho_ConversionType' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ConversionRadius') ] = hf[ 'pho_ConversionRadius' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_VertexConvEtOverPt') ] = hf[ 'pho_VertexConvEtOverPt' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_VertexConvPtRatio') ] = hf[ 'pho_VertexConvPtRatio' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_topoetcone20') ] = hf[ 'pho_topoetcone20' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_topoetcone30') ] = hf[ 'pho_topoetcone30' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_topoetcone40') ] = hf[ 'pho_topoetcone40' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ptvarcone20') ] = hf[ 'pho_ptvarcone20' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_ptvarcone30') ] = hf[ 'pho_ptvarcone30' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_ptvarcone40') ] = hf[ 'pho_ptvarcone40' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_z0') ] = hf[ 'pho_z0' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_z0Sig') ] = hf[ 'pho_z0Sig' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_maxEcell_time') ] = hf[ 'pho_maxEcell_time' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_maxEcell_energy') ] = hf[ 'pho_maxEcell_energy' ][ event][ pho ]
    try:
        data_temp[ 0, column_names.index( 'pho_core57cellsEnergyCorrection') ] = hf[ 'pho_core57cellsEnergyCorrection' ][ event][ pho ]
    except:
        data_temp[ 0, column_names.index( 'pho_core57cellsEnergyCorrection') ] = -999
    data_temp[ 0, column_names.index( 'pho_r33over37allcalo') ] = hf[ 'pho_r33over37allcalo' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_GradientIso') ] = hf[ 'pho_GradientIso' ][ event][ pho ]
    data_temp[ 0, column_names.index('pho_truth_E_atlas') ] = hf['pho_truth_E_atlas'][event][pho]
    data_temp[ 0, column_names.index('pho_truth_px_atlas') ] = hf['pho_truth_px_atlas'][event][pho]
    data_temp[ 0, column_names.index('pho_truth_py_atlas') ] = hf['pho_truth_py_atlas'][event][pho]
    data_temp[ 0, column_names.index('pho_truth_pz_atlas') ] = hf['pho_truth_pz_atlas'][event][pho]

def multiExtract(arguments):
    """
    Extracts files and determines Sig/bkg events.
    Arguments:
        tree: the root tree
        start: event index in file to start at
        stop: event index in file to stop at

    Returns: 
        Data of Photon in array
    """
    # Unpack arguments
    process, counter, path, start, stop, selectiontype, pathtype = arguments
    
    # Counters for histograms
    histNtype = np.zeros((1,3))
    histTrigPass = np.zeros((1,2))
    # Read ROOT data from file
    log.info("[{}] Importing data from {}".format(process,path))
    hf = h5py.File(path, "r")
    
    # Numpy array to return
    data = np.empty((0,len(column_names)), float)

    # Total number of events in batch
    n_events = stop-start

    for i, event in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}]  {} of {} events examined".format(process,i,n_events))

        # Number of photons in event
        try:
            nPho = np.shape(hf[ 'pho_truthType' ][ event ])[0]
        except:
            nPho = 0
        if nPho > 0:
            histTrigPass[0,1] += 1
            for pho in range(nPho):
                if hf["pho_et"][event][pho]<4.5:
                    continue
                data_temp = np.zeros((1,len(column_names)))
                
                try:
                    selection = signalSelection(hf, event, pho, histNtype, selectiontype)
                except:
                    selection = 2
                if selection == 2:
                    continue
                # Add event variables to array 
                data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                data_temp[ 0, column_names.index( 'correctedScaledActualMu' ) ] = hf[ 'correctedScaledActualMu' ][ event ]
                data_temp[ 0, column_names.index( 'type' ) ] = selection
                data_temp[ 0, column_names.index( 'pathtype' ) ] = pathtype
                #log.info("Signal type: {}, Origin: {}, truthtype: {}.".format(selection,hf['pho_truthOrigin'][event][pho], hf['pho_truthType'][event][pho]))
                addPhotonVariables(hf, event, data_temp, pho)
                data = np.append(data, data_temp, axis = 0)
        else:
            histTrigPass[0,0] +=1

    return data, histNtype, histTrigPass

def saveToFile(fname, data, column_names, column_dtype, selection):
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
    if selection == 0:
        seltype = "phoPid_"
    else:
        seltype = "phoIso_"
    log.info("Saving to {}".format(args.outdir + seltype + fname))
    with h5py.File(args.outdir +seltype + fname, 'w') as hf:
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

    Returns:
        Nothing.
    """
    if selection == 0:
        seltype = "phoPid_"
    else:
        seltype = "phoIso_"
    log.info("Appending to {}".format(args.outdir + seltype + fname))
    with h5py.File(args.outdir + seltype + fname, 'a') as hf:
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

def MassHistogram(HistVal, DatType, PathType, fname, title, xlabel, min, max, nbins):
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
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    
  
    TYPES = np.unique(PathType)
    # Add Values to histogram 
    typename =["VBF Hllg", "gg Hllg", r"gg $\gamma\gamma$", r"VBF $\gamma\gamma$", r"gg $\gamma*\gamma$", r"VBF $\gamma*\gamma$",r"Zee$\gamma$", r"Z$\mu\mu\gamma$"]
    for i in TYPES:
        ax.hist(HistVal[DatType==1][TYPES[i]],bins=np.linspace(min,max,nbins),histtype="step", label = f"{typename[TYPES[i]]} Signal")
        ax.hist(HistVal[DatType==0][TYPES[i]],bins=np.linspace(min,max,nbins),histtype="step", label = f"{typename[TYPES[i]]} Background")

    ax.legend()

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(args.outdir+"_PathTypes"+fname)


#============================================================================
# Define column names and dtypes
#============================================================================

column_dtype = {
'correctedScaledAverageMu': float,
'correctedScaledActualMu': float,
'NvtxReco': float,
'eventWeight':float,
'type' : int,
'pathtype': int,
####
####
"pho_truthPdgId_egam" : int,
"pho_truthPdgId_atlas" : int,
# "pho_egamTruthParticle" : int,
"pho_truthType" : int,
"pho_truthOrigin" : int,
"pho_isPhotonEMLoose" : int,
"pho_isPhotonEMTight" : int,
"pho_e" : float,
"pho_eta" : float,
"pho_phi" : float,
"pho_et" : float,
"pho_pt" : float,
"pho_Rhad1" : float,
"pho_Rhad" : float,
"pho_weta2" : float,
"pho_Rphi" : float,
"pho_Reta" : float,
"pho_Eratio" : float,
"pho_f1" : float,
"pho_wtots1" : float,
"pho_DeltaE" : float,
"pho_weta1" : float,
"pho_fracs1" : float,
"pho_ConversionType" : float,
"pho_ConversionRadius" : float,
"pho_VertexConvEtOverPt" : float,
"pho_VertexConvPtRatio" : float,
'pho_truth_px_atlas' : float, 
'pho_truth_py_atlas' : float, 
'pho_truth_pz_atlas' : float, 
"pho_topoetcone20" : float,
#"pho_topoetcone30" : float,
"pho_topoetcone40" : float,
"pho_ptvarcone20" : float,
#"pho_ptvarcone30" : float,
#"pho_ptvarcone40" : float,
"pho_z0" : float,
"pho_z0Sig" : float,
#'pho_maxEcell_time': float,
#'pho_maxEcell_energy': float,
'pho_core57cellsEnergyCorrection': float,
'pho_r33over37allcalo': float,
#'pho_GradientIso': float
'pho_truth_E_atlas': float
}

column_names = list(column_dtype.keys())



#============================================================================
# Main
#============================================================================
# Total counters for signal selection diagram
histNtype_total = np.zeros((1,3))
histTrigPass_total = np.zeros((1,2))
All_masses, All_SBT, All_PathType = [], [], []

histNtype_total2 = np.zeros((1,3))
histTrigPass_total2 = np.zeros((1,2))
All_masses2, All_SBT2, All_PathType2 = [], [], []


# create file name and check if the file already exists
filename = '{:s}.h5'.format(args.tag)
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file already exists - please remove yourself. Output: {args.outdir + filename}")
    quit()

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports __main__ here)
pool = multiprocessing.Pool(processes=args.max_processes)
log.info("============================================================================")
log.info("Starting photon Pid")
log.info("============================================================================")
if args.selection == 0 or args.selection == 2:
    for path in args.paths:
        # Count which file we have made it to
        counter += 1

        # Read hdf5 data to get number of events
        hf_read = h5py.File(path, "r")

        print(hf_read.keys())

        N = hf_read['NvtxReco'].shape[0]
        # Save what file the data is from, to see the different distributions.
        print("N = ", N)
        pathName = os.path.basename(path).split('.',1)[0][0:4]
        if pathName == "VBFH":
            pathtype = 0
        elif pathName == "ggHl":
            pathtype = 1
        elif pathName == "GGgg":
            pathtype = 2
        elif pathName == "VBgg":
            pathtype = 3
        elif pathName == "GGgs":
            pathtype = 4
        elif pathName == "VBgs":
            pathtype = 5
        elif pathName == "Zeeg":
            pathtype = 6
        elif pathName == "Zmmg":
            pathtype = 7
        elif pathName == "jetb":
            pathtype = 8
        else: 
            log.info("No pathtype was selected")
        log.info("Path type is {}".format(pathtype))
        # Split indices into equally-sized batches
        index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
        index_ranges = zip(index_edges[:-1], index_edges[1:])

        results = pool.map(multiExtract, [(i, counter, path, start, stop, 0, pathtype) for i, (start, stop) in enumerate(index_ranges)])
        results_np = np.array(results)

        # Concatenate resulting data from the multiple converters
        data = np.concatenate(results_np[:,0])
        All_masses = np.append(All_masses,data[:,column_names.index("pho_et")])
        All_SBT = np.append(All_SBT,data[:,column_names.index("type")])
        All_PathType = np.append(All_PathType, data[:,column_names.index("pathtype")])
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
            saveToFile(filename, data, column_names, column_dtype, 0)
        else:
            appendToFile(filename, data, column_names, column_dtype, 0)

log.info("============================================================================")
log.info("Starting photon Iso")
log.info("============================================================================")
counter = -1
if args.selection == 1 or args.selection == 2:
    for path in args.paths:
        # Count which file we have made it to
        counter += 1

        # Read hdf5 data to get number of events
        hf_read = h5py.File(path, "r")

        print(hf_read.keys())

        N = hf_read['NvtxReco'].shape[0]
        # Save what file the data is from, to see the different distributions.
        print("N = ", N)
        pathName = os.path.basename(path).split('.',1)[0][0:4]
        if pathName == "VBFH":
            pathtype = 0
        elif pathName == "ggHl":
            pathtype = 1
        elif pathName == "GGgg":
            pathtype = 2
        elif pathName == "VBgg":
            pathtype = 3
        elif pathName == "GGgs":
            pathtype = 4
        elif pathName == "VBgs":
            pathtype = 5
        elif pathName == "Zeeg":
            pathtype = 6
        elif pathName == "Zmmg":
            pathtype = 7
        elif pathName == "jetb":
            pathtype = 8
        else: 
            log.info("No pathtype was selected")
        log.info("Path type is {}".format(pathtype))
        # Split indices into equally-sized batches
        index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
        index_ranges = zip(index_edges[:-1], index_edges[1:])

        results = pool.map(multiExtract, [(i, counter, path, start, stop, 1, pathtype) for i, (start, stop) in enumerate(index_ranges)])
        results_np = np.array(results)

        # Concatenate resulting data from the multiple converters
        data = np.concatenate(results_np[:,0])
        All_masses2 = np.append(All_masses,data[:,column_names.index("pho_et")])
        All_SBT2 = np.append(All_SBT,data[:,column_names.index("type")])
        All2_PathType = np.append(All_PathType, data[:,column_names.index("pathtype")])
        # Concatenate data and add to total
        histNtype = np.concatenate(results_np[:,1], axis = 0)
        histNtype_total2 = histNtype_total + np.sum(histNtype, axis = 0)

        # Concatenate data and add tot total
        histTrigPass = np.concatenate(results_np[:,2], axis = 0)
        histTrigPass_total2 = histTrigPass_total + np.sum(histTrigPass, axis = 0)


        # Print the total event count in the file
        log.info("Data shape: {}".format(data.shape))

        # Save output to a file
        if counter == 0:
            saveToFile(filename, data, column_names, column_dtype, 1)
        else:
            appendToFile(filename, data, column_names, column_dtype, 1)

if (args.selection == 0) or(args.selection == 2):
    plotHistogram(histVal=histNtype_total[0],
                fname=args.tag+"_phoPid_sigSelDiag.png",
                names=[ "Background","Signal", "Trash"],
                title = f"Signal selection ({args.tag})",
                xlabel = "Selection types")
    plotHistogram(histVal=histTrigPass_total[0],
                fname=args.tag+"_phoPid_trigPassDiag.png",
                names=["No trigger in event", "At least one trigger in event"],
                title = f"Events that passes trigger ({args.tag})",
                xlabel = "")

    MassHistogram(HistVal = All_masses, 
                DatType = All_SBT,
                PathType = All_PathType,
                fname = args.tag+"_phoPid_et.png", 
                title = f"Transverse energy of the photon", 
                xlabel = "Transverse energy", 
                min = 20, 
                max = 150, 
                nbins = 151-20)
elif (args.selection==1) or(args.selection==2):                
    plotHistogram(histVal=histNtype_total2[0],
                fname=args.tag+"_phoIso_sigSelDiag.png",
                names=[ "Background","Signal", "Trash"],
                title = f"Signal selection ({args.tag})",
                xlabel = "Selection types")
    plotHistogram(histVal=histTrigPass_total2[0],
                fname=args.tag+"_phoIso_trigPassDiag.png",
                names=["No trigger in event", "At least one trigger in event"],
                title = f"Events that passes trigger ({args.tag})",
                xlabel = "")

    MassHistogram(HistVal = All_masses2, 
                DatType = All_SBT2,
                PathType = All_PathType2,
                fname = args.tag+"_phoIso_et.png", 
                title = f"Transverse energy of the photon", 
                xlabel = "Transverse energy", 
                min = 1, 
                max = 150, 
                nbins = 151-1)


sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
