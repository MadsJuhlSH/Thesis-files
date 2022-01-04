#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Author: Mads Juhl Storr-Hansen
This code is used to select Z->llg events in data - no signal or data will be taken into account
I currently only have muon data, so this script will only take that into account.
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
parser.add_argument('--outdir', action='store', default="output/SignalFiles/data_Llgam/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
#parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=["muoFULL", "muo"],
#                    help = 'The choice of particle l in Z->llgam, either ele or muo')


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

#============================================================================
# Functions
#============================================================================

def combinedVariables(hf, event, lTag, lProbe, phoProbe):
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
    
    lepton = "muo"#args.PartType
    if lepton == "ele":
        p1 = hf['ele_et'][event][lTag]
        eta1 = hf['ele_eta'][event][lTag]
        phi1 = hf['ele_phi'][event][lTag]
        p2 = hf[f'ele_et'][event][lProbe]
        eta2 = hf[f'ele_eta'][event][lProbe]
        phi2 = hf[f'ele_phi'][event][lProbe]
        p3 = hf[f'pho_et'][event][phoProbe]
    elif lepton == "muo":
        p1 = hf['muo_pt'][event][lTag]
        eta1 = hf['muo_eta'][event][lTag]
        phi1 = hf['muo_phi'][event][lTag]
        p2 = hf[f'muo_pt'][event][lProbe]
        eta2 = hf[f'muo_eta'][event][lProbe]
        phi2 = hf[f'muo_phi'][event][lProbe]
        p3 = hf[f'pho_et'][event][phoProbe]*1000
    
    eta3 = hf[f'pho_eta'][event][phoProbe]
    phi3 = hf[f'pho_phi'][event][phoProbe]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()
    vecFour3 = vectors.LorentzVector()
    if lepton == "ele":
        vecFour1.setptetaphim(p1,eta1,phi1,0)
        vecFour2.setptetaphim(p2,eta2,phi2,0)
        vecFour3.setptetaphim(p3,eta3,phi3,0)
    elif lepton == "muo":
        vecFour1.setptetaphim(p1,eta1,phi1,105)
        vecFour2.setptetaphim(p2,eta2,phi2,105)
        vecFour3.setptetaphim(p3,eta3,phi3,0)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2+vecFour3
    vecFourll = vecFour1+vecFour2
    if lepton == "ele":
        invM = vecFour.mass
        invMll = vecFourll.mass
    elif lepton == "muo":
        invM = vecFour.mass/1000
        invMll = vecFourll.mass/1000
    et = vecFour.et
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, et, eta, phi, invMll



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
    
    lTag = []
    lProbe = []
    pProbe = []
    """
    # Just using as many muon candidates as possible

    for muo in range(len(hf['muo_pt'][event])):
        if (hf[ "muo_pt" ][ event ][ muo ] > 10000):
            lTag.append(muo)
    """
    """
    #Here using same requirements as ATLAS
    for muo in range(len(hf['muo_pt'][event])):
        if (hf[ "muo_trigger" ][ event ][ muo ]) * (hf[ "muo_LHMedium" ][ event ][ muo ]) * (hf[ "muo_pt" ][ event ][ muo ] > 10000):
            lTag.append(muo)
        elif (hf[ "muo_LHMedium" ][ event ][ muo ]) * (hf[ "muo_pt" ][ event ][ muo ] > 10000):
            lProbe.append(muo)
    """
    #Trying to get more muons
    for muo in range(len(hf["muo_pt"][event])):
        if (hf[ "muo_trigger" ][ event ][ muo ]):# * (hf[ "muo_pt" ][ event ][ muo ] > 5000):
            lTag.append(muo)
        else:#elif (hf[ "muo_pt" ][ event ][ muo ] > 5000):
            lProbe.append(muo)
    #Get photon probes
    pProbe = np.arange(0,len(hf['pho_et'][event]),1)

    return lTag, lProbe, pProbe


    

def addMuonVariables(hf, event, data_temp, muoNr, muo):
    """
    Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        data_temp: Numpy array to add variables to.
        muoNr: Used for naming variables. (1=tag, 2=probe)
        muo: Muon index.

    Returns:
        Nothing. Data is set in existing array.
    """
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_trigger' ) ] = hf[ 'muo_trigger' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_pt' ) ] = hf[ 'muo_pt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_eta' ) ] = hf[ 'muo_eta' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_phi' ) ] = hf[ 'muo_phi' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_charge' ) ] = hf[ 'muo_charge' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_delta_z0') ] = hf[ 'muo_delta_z0' ][event][muo]
    data_temp[ 0, column_names.index( f'muo{muoNr}_delta_z0_sin_theta') ] = hf[ 'muo_delta_z0_sin_theta' ][event][muo]
    data_temp[ 0, column_names.index( f'muo{muoNr}_muonType' ) ] = hf[ 'muo_muonType' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone20' ) ] = hf[ 'muo_ptvarcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30' ) ] = hf[ 'muo_ptvarcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone40' ) ] = hf[ 'muo_ptvarcone40' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone20' ) ] = hf[ 'muo_ptcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone30' ) ] = hf[ 'muo_ptcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone40' ) ] = hf[ 'muo_ptcone40' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionLayers' ) ] = hf[ 'muo_numberOfPrecisionLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionHoleLayers' ) ] = hf[ 'muo_numberOfPrecisionHoleLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_quality' ) ] = hf[ 'muo_quality' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_innerSmallHits' ) ] = hf[ 'muo_innerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_innerLargeHits' ) ] = hf[ 'muo_innerLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_middleSmallHits' ) ] = hf[ 'muo_middleSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_middleLargeHits' ) ] = hf[ 'muo_middleLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_outerSmallHits' ) ] = hf[ 'muo_outerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_outerLargeHits' ) ] = hf[ 'muo_outerLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_CaloLRLikelihood' ) ] = hf[ 'muo_CaloLRLikelihood' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0' ) ] = hf[ 'muo_priTrack_d0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0' ) ] = hf[ 'muo_priTrack_z0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0Sig' ) ] = hf[ 'muo_priTrack_d0Sig' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0Sig' ) ] = hf[ 'muo_priTrack_z0Sig' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_chiSquared' ) ] = hf[ 'muo_priTrack_chiSquared' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberDoF' ) ] = hf[ 'muo_priTrack_numberDoF' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfPixelHits' ) ] = hf[ 'muo_priTrack_numberOfPixelHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfSCTHits' ) ] = hf[ 'muo_priTrack_numberOfSCTHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfTRTHits' ) ] = hf[ 'muo_priTrack_numberOfTRTHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisol20' ) ] = hf[ 'muo_neflowisol20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_MuonSpectrometerPt' ) ] = hf[ 'muo_MuonSpectrometerPt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_etconecoreConeEnergyCorrection' ) ] = hf[ 'muo_etconecoreConeEnergyCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_InnerDetectorPt' ) ] = hf[ 'muo_InnerDetectorPt' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_author' ) ] = hf[ 'muo_author' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_allAuthors' ) ] = hf[ 'muo_allAuthors' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_scatteringCurvatureSignificance' ) ] = hf[ 'muo_scatteringCurvatureSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_scatteringNeighbourSignificance' ) ] = hf[ 'muo_scatteringNeighbourSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_momentumBalanceSignificance' ) ] = hf[ 'muo_momentumBalanceSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_EnergyLoss' ) ] = hf[ 'muo_EnergyLoss' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_energyLossType' ) ] = hf[ 'muo_energyLossType' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_vertex_z' ) ] = hf[ 'vertex_z' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_delta_z0' ) ] = hf[ 'muo_delta_z0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_delta_z0_sin_theta' ) ] = hf[ 'muo_delta_z0_sin_theta' ][ event ][ muo ]

    #data_temp[ 0, column_names.index( f'muo{muoNr}_m' ) ] = hf[ 'muo_m' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_e' ) ] = hf[ 'muo_e' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_et' ) ] = hf[ 'muo_et' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ECIDSResult' ) ] = hf[ 'muo_ECIDSResult' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_d0' ) ] = hf[ 'muo_d0' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_d0Sig' ) ] = hf[ 'muo_d0Sig' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_z0' ) ] = hf[ 'muo_z0' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_z0Sig' ) ] = hf[ 'muo_z0Sig' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_f3' ) ] = hf[ 'muo_f3' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_f1' ) ] = hf[ 'muo_f1' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_Reta' ) ] = hf[ 'muo_Reta' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_Rphi' ) ] = hf[ 'muo_Rphi' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_Rhad' ) ] = hf[ 'muo_Rhad' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_Rhad1' ) ] = hf[ 'muo_Rhad1' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_weta2' ) ] = hf[ 'muo_weta2' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_Eratio' ) ] = hf[ 'muo_Eratio' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_TRTPID' ) ] = hf[ 'muo_TRTPID' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_dPOverP' ) ] = hf[ 'muo_dPOverP' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_deltaEta1' ) ] = hf[ 'muo_deltaEta1' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_deltaPhiRescaled2' ) ] = hf[ 'muo_deltaPhiRescaled2' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_topoetcone20' ) ] = hf[ 'muo_topoetcone20' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_topoetcone40' ) ] = hf[ 'muo_topoetcone40' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'muo_ptcone20_TightTTVALooseCone_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone20_TightTTVALooseCone_pt500' ) ] = hf[ 'muo_ptcone20_TightTTVALooseCone_pt500' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'muo_ptvarcone20_TightTTVALooseCone_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone20_TightTTVA_pt1000' ) ] = hf[ 'muo_ptvarcone20_TightTTVA_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30_TightTTVALooseCone_pt1000' ) ] = hf[ 'muo_ptvarcone30_TightTTVALooseCone_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30_TightTTVALooseCone_pt500' ) ] = hf[ 'muo_ptvarcone30_TightTTVALooseCone_pt500' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30_TightTTVA_pt1000' ) ] = hf[ 'muo_ptvarcone30_TightTTVA_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30_TightTTVA_pt500' ) ] = hf[ 'muo_ptvarcone30_TightTTVA_pt500' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone40_TightTTVALooseCone_pt1000' ) ] = hf[ 'muo_ptvarcone40_TightTTVALooseCone_pt1000' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_topoetcone20ptCorrection' ) ] = hf[ 'muo_topoetcone20ptCorrection' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_expectInnermostPixelLayerHit')] = hf[ 'muo_expectInnermostPixelLayerHit' ][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_expectNextToInnermostPixelLayerHit')] = hf[ 'muo_expectNextToInnermostPixelLayerHit'][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_nTracks')] = hf[ 'muo_nTracks'][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfInnermostPixelHits')] = hf[ 'muo_numberOfInnermostPixelHits'][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPixelHits')] = hf[ 'muo_numberOfPixelHits'][ event ][ muo]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfSCTHits')] = hf[ 'muo_numberOfSCTHits'][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfTRTHits')] = hf[ 'muo_numberOfTRTHits'][ event ][ muo ]
    #data_temp[ 0, column_names.index( f'muo{muoNr}_core57cellsEnergyCorrection')] = hf[ 'muo_core57cellsEnergyCorrection'][ event ][ muo ]
    """
    if muoNr==2:
        data_temp[ 0, column_names.index( 'muo2_topoetcone30' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_topoetconecoreConeEnergyCorrection' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_secondLambdaCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_lateralCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_longitudinalCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_fracMaxCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_secondRCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_centerLambdaCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_isFwd' ) ] = 0
    """

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
    
    data_temp[ 0, column_names.index( 'pho_isPhotonEMLoose') ] = hf[ 'pho_isPhotonEMLoose' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_isPhotonEMTight') ] = hf[ 'pho_isPhotonEMTight' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_e') ] = hf[ 'pho_e' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_eta') ] = hf[ 'pho_eta' ][ event][ pho ]
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
    #data_temp[ 0, column_names.index( 'pho_topoetcone30') ] = hf[ 'pho_topoetcone30' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_topoetcone40') ] = hf[ 'pho_topoetcone40' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ptvarcone20') ] = hf[ 'pho_ptvarcone20' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_ptvarcone30') ] = hf[ 'pho_ptvarcone30' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_ptvarcone40') ] = hf[ 'pho_ptvarcone40' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_z0') ] = hf[ 'pho_z0' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_z0Sig') ] = hf[ 'pho_z0Sig' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_maxEcell_time') ] = hf[ 'pho_maxEcell_time' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_maxEcell_energy') ] = hf[ 'pho_maxEcell_energy' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_core57cellsEnergyCorrection') ] = hf[ 'pho_core57cellsEnergyCorrection' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_r33over37allcalo') ] = hf[ 'pho_r33over37allcalo' ][ event][ pho ]
    #data_temp[ 0, column_names.index( 'pho_GradientIso') ] = hf[ 'pho_GradientIso' ][ event][ pho ]



def multiExtract(arguments):
    """
    Extracts the variable from the Hdf5 file with root2hdf5
    Using tag and probe:
        tag = lepton that has triggered
        probe = any other lepton of same generation in container
    selects signal type based on pdgid, truthorigin and truthparticle
    selection: 0=bkg, 1=sig,2=trash.
    bkgsel: 0=sig, 1=bad lepton, 2=bad photon, 3= both leptons and photon are bad

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
    process, counter, path, start, stop = arguments

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

    # Run over all events in the start stop range
    for i, event  in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}] {} of {} events examined".format(process, i, n_events))
        
        # Get tags and probes
        try:
            lTag, lProbe, pProbe = getTagsAndProbes(hf,event, process)
            nTag = len(lTag)
        except: 
            nTag, lProbe, pProbe = 0,[],[]

        #print(f"nTag: {nTag}, nlProbe: {len(lProbe)}, npProbe: {len(pProbe)}.")
        # If there is at least one tag, 2 leptons of same kind and 1 photon -> get values
        if ((nTag>0) & (len(lProbe)+nTag >1) & (len(pProbe)>0) ):
            # Event passed trigger cut - add to histogram
            histTrigPass[0,1] +=  1

            for iLTag, LTag in enumerate(lTag):
                # Get Probes
                lProbes = lProbe.copy()
                phoProbes = pProbe.copy()
                # Append subsecuent lepton tags to lepton probes (applies if there is more than one tag lepton)
                for lep in (lTag[(iLTag+1):]):
                    lProbes.append(lep)
               
               #Havde tidigere en med kravende 5,5,5, prøver nu -,-,2.5
               
                for iLProbe, LProbe in enumerate(lProbes):
                    for iPhoProbe, phoProbe in enumerate(phoProbes):
                        # Create array for saving data
                        if hf["pho_et"][event][phoProbe] <5:
                            continue
                        data_temp = np.zeros((1,len(column_names)))
                        
                        
                        # Calculate varaibles of the muon pair and the photon
                        invM, et, eta, phi, invMll = combinedVariables(hf, event, LTag, LProbe, phoProbe)
                        data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                        data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                        data_temp[ 0, column_names.index( 'invM' ) ] = invM
                        data_temp[ 0, column_names.index( 'invMll' ) ] = invMll
                        data_temp[ 0, column_names.index( 'pt' ) ] = et
                        data_temp[ 0, column_names.index( 'eta' ) ] = eta
                        data_temp[ 0, column_names.index( 'phi' ) ] = phi
                        

                        # Add muon variables to array
                        addMuonVariables(hf, event, data_temp, 1, LTag)
                        addMuonVariables(hf, event, data_temp, 2, LProbe)
                        addPhotonVariables(hf, event, data_temp, phoProbe)

                        # Append data to array that is returned
                        data = np.append(data, data_temp, axis=0)    
        else: 
            # Event did not pas trigger cut - add to histogram
            histTrigPass[0,0] += 1
    return data, histNtype, histTrigPass

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

def MassHistogram(HistVal, fname, title, xlabel, min, max, nbins):
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
    ax.hist(HistVal,bins=np.linspace(min,max,nbins),histtype="step", label = "Signal")
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
                "invM" : float,
                "invMll": float,
                "pt" : float,
                "eta" : float,
                "phi" : float,
                'muo1_LHLoose' : int ,
                'muo1_LHMedium' : int,
                'muo1_LHTight' : int,
                'muo1_trigger' : int,
                'muo1_pt' : float,
                'muo1_eta' : float,
                'muo1_phi' : float,
                'muo1_charge' : int,
                'muo1_delta_z0' : float,
                'muo1_delta_z0_sin_theta' : float,
                'muo1_muonType' : int,
                'muo1_ptvarcone20' : float,
                'muo1_ptvarcone30' : float,
                'muo1_ptvarcone40' : float,
                'muo1_ptcone20' : float,
                'muo1_ptcone30' : float,
                'muo1_ptcone40' : float,
                'muo1_numberOfPrecisionLayers' : int,
                'muo1_numberOfPrecisionHoleLayers' : int,
                'muo1_quality' : float,
                'muo1_innerSmallHits' : int,
                'muo1_innerLargeHits' : int,
                'muo1_middleSmallHits' : int,
                'muo1_middleLargeHits' : int,
                'muo1_outerSmallHits' : int,
                'muo1_outerLargeHits' : int,
                'muo1_CaloLRLikelihood' : float,
                'muo1_priTrack_d0' : float,
                'muo1_priTrack_z0' : float,
                'muo1_priTrack_d0Sig' : float,
                'muo1_priTrack_z0Sig' : float,
                'muo1_priTrack_chiSquared' : float,
                'muo1_priTrack_numberDoF' : int,
                'muo1_priTrack_numberOfPixelHits' : int,
                'muo1_priTrack_numberOfSCTHits' : int,
                'muo1_priTrack_numberOfTRTHits' : int,
                'muo1_neflowisol20' : float,
                'muo1_MuonSpectrometerPt' :float ,
                'muo1_etconecoreConeEnergyCorrection' : float,
                'muo1_InnerDetectorPt' : float,
                #'muo1_author' : str,
                #'muo1_allAuthors' : str,
                'muo1_scatteringCurvatureSignificance' : float,
                'muo1_scatteringNeighbourSignificance' : float,
                'muo1_momentumBalanceSignificance' : float,
                'muo1_EnergyLoss' : float,
                'muo1_energyLossType' : int,
                'muo1_vertex_z' : float,
                'muo1_delta_z0' : float,
                'muo1_delta_z0_sin_theta' : float,
                'muo2_LHLoose' : int ,
                'muo2_LHMedium' : int,
                'muo2_LHTight' : int,
                'muo2_trigger' : int,
                'muo2_pt' : float,
                'muo2_eta' : float,
                'muo2_phi' : float,
                'muo2_charge' : int,
                'muo2_delta_z0' : float,
                'muo2_delta_z0_sin_theta' : float,
                'muo2_muonType' : int,
                'muo2_ptvarcone20' : float,
                'muo2_ptvarcone30' : float,
                'muo2_ptvarcone40' : float,
                'muo2_ptcone20' : float,
                'muo2_ptcone30' : float,
                'muo2_ptcone40' : float,
                'muo2_numberOfPrecisionLayers' : int,
                'muo2_numberOfPrecisionHoleLayers' : int,
                'muo2_quality' : float,
                'muo2_innerSmallHits' : int,
                'muo2_innerLargeHits' : int,
                'muo2_middleSmallHits' : int,
                'muo2_middleLargeHits' : int,
                'muo2_outerSmallHits' : int,
                'muo2_outerLargeHits' : int,
                'muo2_CaloLRLikelihood' : float,
                'muo2_priTrack_d0' : float,
                'muo2_priTrack_z0' : float,
                'muo2_priTrack_d0Sig' : float,
                'muo2_priTrack_z0Sig' : float,
                'muo2_priTrack_chiSquared' : float,
                'muo2_priTrack_numberDoF' : int,
                'muo2_priTrack_numberOfPixelHits' : int,
                'muo2_priTrack_numberOfSCTHits' : int,
                'muo2_priTrack_numberOfTRTHits' : int,
                'muo2_neflowisol20' : float,
                'muo2_MuonSpectrometerPt' :float ,
                'muo2_etconecoreConeEnergyCorrection' : float,
                'muo2_InnerDetectorPt' : float,
                #'muo2_author' : str,
                #'muo2_allAuthors' : str,
                'muo2_scatteringCurvatureSignificance' : float,
                'muo2_scatteringNeighbourSignificance' : float,
                'muo2_momentumBalanceSignificance' : float,
                'muo2_EnergyLoss' : float,
                'muo2_energyLossType' : int,
                'muo2_vertex_z' : float,
                'muo2_delta_z0' : float,
                'muo2_delta_z0_sin_theta' : float,
                #"pho_egamTruthParticle" : int,
                "pho_truthType" : int,
                "pho_truthOrigin" : int,
                "pho_isPhotonEMLoose" : int,
                "pho_isPhotonEMTight" : int,
                "pho_e" : float,
                "pho_eta" : float,
                "pho_phi" : float,
                "pho_et" : float,
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
                "pho_topoetcone20" : float,
                "pho_topoetcone30" : float,
                "pho_topoetcone40" : float,
                "pho_ptvarcone20" : float,
                "pho_ptvarcone30" : float,
                "pho_ptvarcone40" : float,
                "pho_z0" : float,
                "pho_z0Sig" : float,
                #'pho_maxEcell_time': float,
                #'pho_maxEcell_energy': float,
                #'pho_core57cellsEnergyCorrection': float,
                'pho_r33over37allcalo': float,
                #'pho_GradientIso': float, }
}

column_names = list(column_dtype.keys())


#============================================================================
# Main Signal selection
#============================================================================
# Total counters for signal selection diagram
histNtype_total = np.zeros((1,3))
histTrigPass_total = np.zeros((1,2))
All_masses, All_mll = [], []



# Create file name and check if the file already exists
filename = f"{args.tag}.h5"
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file already exists - please remove yourself")
    quit()

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports _main_here)
pool = multiprocessing.Pool(processes = args.max_processes)

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
    
    results = pool.map(multiExtract, [(i, counter, path, start, stop) for i, (start, stop) in enumerate(index_ranges)])
    results_np = np.array(results)

    # Concatenate resulting data from the multiple converters
    data = np.concatenate(results_np[:,0])
    All_masses = np.append(All_masses,data[:,column_names.index("invM")])
    All_mll = np.append(All_mll,data[:,column_names.index("invMll")])


    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))
    
    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)
    
# Create and save figure of signal selection

MassHistogram(HistVal = All_masses, 
              fname = args.tag+"_InvMass.png", 
              title = f"Invariant mass of the muon pair and photon", 
              xlabel = "Invariant mass", 
              min = 1, 
              max = 150, 
              nbins = 151-1)
MassVsMassHistogram(LLMass=All_mll,
                    LLGamMass=All_masses,
                    fname= args.tag+"_Mll_Vs_Mllgam.png",
                    title=f"Invariant mass of llgam vs ll pair",
                    xlabel="Mass of LLgam",
                    ylabel= "Mass of LL"
)

sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")

