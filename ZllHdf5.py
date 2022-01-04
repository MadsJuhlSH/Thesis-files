#!/usr/bin/env python 
# -*- coding: utf-8 -*-

'''
Author: Mads Juhl Storr-Hansen
This code is used for the signal selection of Z->ll
This will later be extended to a reweigh function
'''
print('Program running')

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
log.info('Packages imported')

# Start 'timer'
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description='Extract data from HDF5 files into flat HDF5 files for training.')
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default='output/SignalFiles/Ll/', type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--PartType', action = 'store', type = str, required= True, choices=['ele', 'muo'],
                    help = 'The choice of particle l in Z->ll, either ele or muo')
parser.add_argument('--LepOrigin', action = 'store', type = int, default = 13, required =False,
                    help = 'The origin of the leptons, will default to 13 (Z).')

args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error('No HDF5 file was specified.')
    quit()

if args.max_processes > 20:
    log.error('The requested number of processes ({}) is excessive (>20). Exiting.'.format(args.max_processes))
    quit()
log.info('Selected particletype is {}'.format(args.PartType))
if ((args.PartType != 'ele') + (args.PartType != 'muo'))!=1:
    log.error('Unknown lepton, use either ele or muo')
    quit()


#Makes an output directory, and gives an error if it exists.
args.outdir = args.outdir+f'{args.tag}/'#_{nameSigBkg}/'
if os.path.exists(args.outdir):
    log.error(f'Output already exists - please remove yourself. Output: {args.outdir}')
    quit()
else:
    log.info(f'Creating output folder: {args.outdir}')
    mkdir(args.outdir)

# File number counter (incremented first in loop)
counter = -1

#============================================================================
# Functions
#============================================================================

def combinedVariables(hf, event, lTag, lProbe):
    '''
    Calculate variables of the lepton pair.
    It should be noted that electrons' mass and energy
    is given in GeV while they are given as MeV for 
    muons. Therefore the multiplication and divisions by 1000.
    Arguments:
        hf: File to get variables from.
        event: Event number.
        lTag: Index of tag lepton.
        lProbe: Index of probe lepton.

    Returns:
        invM: Invariant mass of combined four-vector.
        et: Transverse energy of combined four-vector.
        eta: Eta of combined four-vector.
        phi: Phi of combined four-vector.
    '''
    # Calculate invM etc. of ll using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variable
    
    lepton = args.PartType
    if lepton == 'ele':
        p1 = hf['ele_et'][event][lTag]
        eta1 = hf['ele_eta'][event][lTag]
        phi1 = hf['ele_phi'][event][lTag]
        p2 = hf[f'ele_et'][event][lProbe]
        eta2 = hf[f'ele_eta'][event][lProbe]
        phi2 = hf[f'ele_phi'][event][lProbe]
    elif lepton == 'muo':
        p1 = hf['muo_pt'][event][lTag]
        eta1 = hf['muo_eta'][event][lTag]
        phi1 = hf['muo_phi'][event][lTag]
        p2 = hf[f'muo_pt'][event][lProbe]
        eta2 = hf[f'muo_eta'][event][lProbe]
        phi2 = hf[f'muo_phi'][event][lProbe]
    
    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()
    if lepton == 'ele':
        vecFour1.setptetaphim(p1,eta1,phi1,0)
        vecFour2.setptetaphim(p2,eta2,phi2,0)
    elif lepton == 'muo':
        vecFour1.setptetaphim(p1,eta1,phi1,105)
        vecFour2.setptetaphim(p2,eta2,phi2,105)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2
    if lepton == 'ele':
        invM = vecFour.mass
    elif lepton == 'muo':
        invM = vecFour.mass/1000
    et = vecFour.et
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, et, eta, phi


def signalSelection(hf, event, tag, probe, histNtype):
    '''
    Selects a type for the given lepton pair based on a flowchart.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        lTag: Index of tag lepton.
        lProbe: Index of probe lepton.

    Returns:
        The type of the pair:
        0 = background
        1 = signal
    '''
    # Do the leptons originate from a Z boson?
    # Do the leptons have opposite charge?
    if args.PartType == 'ele':
        isZll_0 = (hf['ele_truthOrigin'][event][tag] == args.LepOrigin)
        isZll_1 = (hf['ele_truthOrigin'][event][probe] == args.LepOrigin)
        lep0_PdgId = hf['ele_truthPdgId_atlas'][event][tag]
        lep1_PdgId = hf['ele_truthPdgId_atlas'][event][probe]
        lep0 = (np.abs(lep0_PdgId) == 11)
        lep1 = (np.abs(lep1_PdgId) == 11)
    elif args.PartType == 'muo':
        isZll_0 = (hf['muo_truthOrigin'][event][tag] == args.LepOrigin)
        isZll_1 = (hf['muo_truthOrigin'][event][probe] == args.LepOrigin)
        lep0_PdgId = hf['muo_truthPdgId'][event][tag]
        lep1_PdgId = hf['muo_truthPdgId'][event][probe]
        lep0 = (np.abs(lep0_PdgId) == 13)
        lep1 = (np.abs(lep1_PdgId) == 13)

    if (isZll_0*isZll_1): 
        if args.PartType == 'ele':
            EGAMtruthMatched_0 = (hf['ele_egamTruthParticle'][event][tag] == 1)
            EGAMtruthMatched_1 = (hf['ele_egamTruthParticle'][event][probe] == 1)
            bothEle = lep0*lep1
            oppositeSign = (lep0_PdgId*lep1_PdgId < 0)
            bothEGAMtruthMatched = EGAMtruthMatched_0*EGAMtruthMatched_1

            if (bothEle*oppositeSign*bothEGAMtruthMatched):
                histNtype[0,1] += 1
                return 1 # Signal
#            elif (oppositeSign==0):
#                histNtype[0,0] += 1
#                return 0 # Background
            else:
                histNtype[0,2] += 1
                return 2 # Trash
        elif args.PartType == 'muo': 
            bothMuo = lep0*lep1
            oppositeSign = (lep0_PdgId*lep1_PdgId < 0)
            if (bothMuo*oppositeSign):
                histNtype[0,1] += 1  
                return 1    #Signal
            #elif (oppositeSign==0):
            #    histNtype[0,0] += 1
            #    return 0 # Background
            else: 
                histNtype[0,2] += 1
                return 2
    else:
        if (isZll_0+isZll_1<2): # none or one origins from a Z
            histNtype[0,0] += 1
            return 0 # Bkg
            
        """    
        elif (lep0+lep1==2): # Both are the given leptons
            histNtype[0,0] += 1
            return 0 # Bkg2Ele
        elif (lep0+lep1==1): # One is a lepton
            histNtype[0,0] += 1
            return 0 # Bkg1Ele
        else: # No leptons
            histNtype[0,0] += 1
            return 0  # Bkg0Ele"""

    
def getTagsAndProbes(hf, event, process):
    '''
    Get tag and probe indices in process
    Input: 
        hf: File to get variables from
        event: Event number
    Output:
        lTag: Array of tag leptons.
        lProbe: Array of probe leptons.
    '''
    
    lTag = []
    lProbe = []
    if args.PartType == 'ele':
        for ele in range(len(hf['ele_et'][event])):
            if hf['ele_trigger'][event][ele]:
                lTag.append(ele)
            else:
                lProbe.append(ele)
    elif args.PartType == 'muo':
        for muo in range(len(hf['muo_pt'][event])):
            if hf['muo_trigger'][event][muo]:
                lTag.append(muo)
            else: 
                lProbe.append(muo)
    
    return lTag, lProbe
    
def addElectronVariables(hf, event, data_temp, eleNr, ele):
    '''
    Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        data_temp: Numpy array to add variables to.
        eleNr: Used for naming variables. (1=tag, 2=probe)
        ele: Electron index.

    Returns:
        Nothing. Data is set in existing array.
    '''
    data_temp[ 0, column_names.index( f'ele{eleNr}_truthParticle' ) ] = hf[ 'ele_egamTruthParticle' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_truthPdgId' ) ] = hf[ 'ele_truthPdgId_atlas' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_truthType' ) ] = hf[ 'ele_truthType' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_truthOrigin' ) ] = hf[ 'ele_truthOrigin' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_LHLoose' ) ] = hf[ 'ele_LHLoose' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_LHMedium' ) ] = hf[ 'ele_LHMedium' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_LHTight' ) ] = hf[ 'ele_LHTight' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_trigger' ) ] = hf[ 'ele_trigger' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_m' ) ] = hf[ 'ele_m' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_e' ) ] = hf[ 'ele_e' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_et' ) ] = hf[ 'ele_et' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_eta' ) ] = hf[ 'ele_eta' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_phi' ) ] = hf[ 'ele_phi' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_charge' ) ] = hf[ 'ele_charge' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele{eleNr}_ECIDSResult' ) ] = hf[ 'ele_ECIDSResult' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_d0' ) ] = hf[ 'ele_d0' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_d0Sig' ) ] = hf[ 'ele_d0Sig' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_z0' ) ] = hf[ 'ele_z0' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_z0Sig' ) ] = hf[ 'ele_z0Sig' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_f3' ) ] = hf[ 'ele_f3' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_f1' ) ] = hf[ 'ele_f1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_Reta' ) ] = hf[ 'ele_Reta' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_Rphi' ) ] = hf[ 'ele_Rphi' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_Rhad' ) ] = hf[ 'ele_Rhad' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_Rhad1' ) ] = hf[ 'ele_Rhad1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_weta2' ) ] = hf[ 'ele_weta2' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_Eratio' ) ] = hf[ 'ele_Eratio' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_TRTPID' ) ] = hf[ 'ele_TRTPID' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_dPOverP' ) ] = hf[ 'ele_dPOverP' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_deltaEta1' ) ] = hf[ 'ele_deltaEta1' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_deltaPhiRescaled2' ) ] = hf[ 'ele_deltaPhiRescaled2' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone20' ) ] = hf[ 'ele_ptvarcone20' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_topoetcone20' ) ] = hf[ 'ele_topoetcone20' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_topoetcone40' ) ] = hf[ 'ele_topoetcone40' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptcone20_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptcone20_TightTTVALooseCone_pt500' ) ] = hf[ 'ele_ptcone20_TightTTVALooseCone_pt500' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone20_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone20_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone20_TightTTVA_pt1000' ) ] = hf[ 'ele_ptvarcone20_TightTTVA_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone30_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone30_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone30_TightTTVALooseCone_pt500' ) ] = hf[ 'ele_ptvarcone30_TightTTVALooseCone_pt500' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone30_TightTTVA_pt1000' ) ] = hf[ 'ele_ptvarcone30_TightTTVA_pt1000' ][ event ][ ele ]
    #data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone30_TightTTVA_pt500' ) ] = hf[ 'ele_ptvarcone30_TightTTVA_pt500' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_ptvarcone40_TightTTVALooseCone_pt1000' ) ] = hf[ 'ele_ptvarcone40_TightTTVALooseCone_pt1000' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_topoetcone20ptCorrection' ) ] = hf[ 'ele_topoetcone20ptCorrection' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_expectInnermostPixelLayerHit')] = hf[ 'ele_expectInnermostPixelLayerHit' ][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_expectNextToInnermostPixelLayerHit')] = hf[ 'ele_expectNextToInnermostPixelLayerHit'][ event ][ ele ]
    try:
        data_temp[ 0, column_names.index( f'ele{eleNr}_nTracks')] = hf[ 'ele_nTracks'][ event ][ ele ]
    except: 
        data_temp[ 0, column_names.index( f'ele{eleNr}_nTracks')] = -999
    data_temp[ 0, column_names.index( f'ele{eleNr}_numberOfInnermostPixelHits')] = hf[ 'ele_numberOfInnermostPixelHits'][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_numberOfPixelHits')] = hf[ 'ele_numberOfPixelHits'][ event ][ ele]
    data_temp[ 0, column_names.index( f'ele{eleNr}_numberOfSCTHits')] = hf[ 'ele_numberOfSCTHits'][ event ][ ele ]
    data_temp[ 0, column_names.index( f'ele{eleNr}_numberOfTRTHits')] = hf[ 'ele_numberOfTRTHits'][ event ][ ele ]
    try:
        data_temp[ 0, column_names.index( f'ele{eleNr}_core57cellsEnergyCorrection')] = hf[ 'ele_core57cellsEnergyCorrection'][ event ][ ele ]
    except:
        data_temp[ 0, column_names.index( f'ele{eleNr}_core57cellsEnergyCorrection')] = -999
    if eleNr==2:
        data_temp[ 0, column_names.index( 'ele2_topoetcone30' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_topoetconecoreConeEnergyCorrection' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_secondLambdaCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_lateralCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_longitudinalCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_fracMaxCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_secondRCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'ele2_centerLambdaCluster' ) ] = -999

def addMuonVariables(hf, event, data_temp, muoNr, muo):
    '''
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
    '''
    data_temp[ 0, column_names.index( f'muo2_truthPdgId' ) ] = hf[ 'muo_truthPdgId' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_truthType' ) ] = hf[ 'muo_truthType' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_truthOrigin' ) ] = hf[ 'muo_truthOrigin' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_trigger' ) ] = hf[ 'muo_trigger' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_pt' ) ] = hf[ 'muo_pt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_eta' ) ] = hf[ 'muo_eta' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_phi' ) ] = hf[ 'muo_phi' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_charge' ) ] = hf[ 'muo_charge' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_delta_z0') ] = hf[ 'muo_delta_z0' ][event][muo]
    data_temp[ 0, column_names.index( f'muo{muoNr}_muo_delta_z0_sin_theta') ] = hf[ 'muo_delta_z0_sin_theta' ][event][muo]
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
    '''
    if muoNr==2:
        data_temp[ 0, column_names.index( 'muo2_topoetcone30' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_topoetconecoreConeEnergyCorrection' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_secondLambdaCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_lateralCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_longitudinalCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_fracMaxCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_secondRCluster' ) ] = -999
        data_temp[ 0, column_names.index( 'muo2_centerLambdaCluster' ) ] = -999
    '''

def multiExtract(arguments):
    '''
    Extracts the variable from the Hdf5 file with root2hdf5
    Using tag and probe:
        tag = lepton that has triggered
        probe = any other lepton of same generation in container
    selects signal type based on pdgid, truthorigin and truthparticle

    Arguments:
        Process: process number
        counter: file counter
        path: path to file
        start: event index in file to start at
        stop: event index in file to stop at

    Returns:
        Data of lepton pairs in numpy array
        Counters for histogram over signal selection and cut flow
    '''
    # Unpack arguments
    process, counter, path, start, stop = arguments

    # Counters for histograms
    histNtype = np.zeros((1,3))
    histTrigPass = np.zeros((1,2))
    # Read ROOT data from file
    log.info('[{}]  Importing data from {}'.format(process,path))
    hf = h5py.File(path,'r')

    # Numpy array to return
    data = np.empty((0,len(column_names)), float)

    # Total number of events in batch
    n_events = stop-start

    # Run over all events in the start stop range
    for i, event  in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info('[{}] {} of {} events examined'.format(process, i, n_events))
        
        # Get tags and probes
        try:
            lTag, lProbe = getTagsAndProbes(hf,event, process)
            nTag = len(lTag)
            #log.info('Number of tags: {}, number of probes: {}'.format(nTag,len(lProbe)))
        except: 
            log.info('Get Tags and probes gave an error')
            nTag, lProbe = 0,[]

        # If there is at least one tag, 2 leptons of same kind -> get values
        if ((nTag>0) & (len(lProbe)+nTag >1) ):
            # Event passed trigger cut - add to histogram
            histTrigPass[0,1] +=  1

            for iLTag, LTag in enumerate(lTag):
                if args.PartType == "ele":
                    if hf["ele_et"][event][LTag] < 4.5:
                        continue
                elif args.PartType == "muo":
                    if hf["muo_pt"][event][LTag] < 4.5:
                        continue
                # Get Probes
                lProbes = lProbe.copy()
                # Append subsecuent lepton tags to lepton probes (applies if there is more than one tag lepton)
                for lep in (lTag[(iLTag+1):]):
                    lProbes.append(lep)
                
                for iLProbe, LProbe in enumerate(lProbes):
                    if args.PartType == "ele":
                        if hf["ele_et"][event][LProbe] < 4.5:
                            continue
                    elif args.PartType == "muo":
                        if hf["muo_pt"][event][LProbe] < 4.5:
                            continue
                    # Create array for saving data
                    data_temp = np.zeros((1,len(column_names)))
                    
                    selection = signalSelection(hf, event, LTag, LProbe, histNtype)      
                    if selection == 2:
                        continue
                    
                    # Calculate varaibles of the electron pair
                    invM, et, eta, phi = combinedVariables(hf, event, LTag, LProbe)
                    #print(selection, invM)
                    data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                    data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                    data_temp[ 0, column_names.index( 'invM' ) ] = invM
                    if args.PartType == 'ele':
                        data_temp[ 0, column_names.index( 'et' ) ] = et
                    elif args.PartType == 'muo': 
                        data_temp[ 0, column_names.index( 'pt' ) ] = et
                    data_temp[ 0, column_names.index( 'eta' ) ] = eta
                    data_temp[ 0, column_names.index( 'phi' ) ] = phi
                    data_temp[ 0, column_names.index( 'type' ) ] = selection

                    # Add electron variables to array
                    if args.PartType == 'ele':
                        addElectronVariables(hf, event, data_temp, 1, LTag)
                        addElectronVariables(hf, event, data_temp, 2, LProbe)
                    elif args.PartType == 'muo':
                        addMuonVariables(hf, event, data_temp, 1, LTag)
                        addMuonVariables(hf, event, data_temp, 2, LProbe)

                    # Append data to array that is returned
                    data = np.append(data, data_temp, axis=0)    
        else: 
            # Event did not pas trigger cut - add to histogram
            histTrigPass[0,0] += 1
    return data, histNtype, histTrigPass

def saveToFile(fname, data, column_names, column_dtype):
    '''
    Simply saves data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    '''
    log.info('Saving to {}'.format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'w') as hf:
        for var in column_names:
            hf.create_dataset( f'{var}',
                              data=data[:,column_names.index(f'{var}')],
                              dtype=column_dtype[f'{var}'],
                              chunks=True,
                              maxshape= (None,),
                              compression='lzf')

def appendToFile(fname, data, column_names, column_dtype):
    '''
    Simply appends data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    '''
    log.info('Appending to {}'.format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'a') as hf:
        for var in column_names:

            array = data[:,column_names.index(f'{var}')]
            hf[f'{var}'].resize((hf[f'{var}'].shape[0] + array.shape[0]), axis = 0)
            hf[f'{var}'][-array.shape[0]:] = array.astype(column_dtype[f'{var}'])

def plotHistogram(histVal, fname, names, title, xlabel):
    '''
    Simply plots histogram and saves it.

    Arguments:
        histVal: Values on y axis.
        fname: filename (directory is taken from script's args).
        names: Names on x axis.
        title: Title of histogram.
        xlabel: xlabel of histogram.

    Returns:
        Nothing.
    '''
    log.info('Plot and save histogram: {}'.format(args.outdir + fname))
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
    ax.set_ylabel('Frequency')

    # Add values to histogram
    shift = np.max(val)*0.01
    for i in range(n):
        ax.text(x[i], val[i]+shift, f'{int(val[i])}', horizontalalignment='center')

    # Save histogram
    plt.tight_layout()
    fig.savefig(args.outdir + fname)

def MassHistogram(HistVal, DatType, fname, title, xlabel, min, max, nbins):
    '''
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
    '''
    log.info('Plot and save histogram of masses: {}'.format(args.outdir + fname))
   
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    
    # Add Values to histogram 
    ax.hist(HistVal[DatType==1],bins=np.linspace(min,max,nbins),histtype='step', label = 'Signal')
    ax.hist(HistVal[DatType==0],bins=np.linspace(min,max,nbins),histtype='step', label = 'Background')

    ax.legend()

    # Save Histogram 
    plt.tight_layout()
    fig.savefig(args.outdir+fname)
    
    
   
#============================================================================
# Define column names and dtypes (Used when saving)
#============================================================================
if args.PartType == 'ele':
    column_dtype = {'NvtxReco' : int,
                    'correctedScaledAverageMu' : float,
                    'invM' : float,
                    'et' : float,
                    'eta' : float,
                    'phi' : float,
                    'invM_truth' : float,
                    'et_truth' : float,
                    'eta_truth' : float,
                    'phi_truth' : float,
                    'type' : int,
                    'ele1_truthParticle' : int,
                    'ele1_truthPdgId' : int,
                    'ele1_truthType' : int,
                    'ele1_truthOrigin' : int,
                    'ele1_LHLoose' : int,
                    'ele1_LHMedium' : int,
                    'ele1_LHTight' : int,
                    'ele1_trigger' : int,
                    'ele1_m' : float,
                    'ele1_e' : float,
                    'ele1_et' : float,
                    'ele1_eta' : float,
                    'ele1_phi' : float,
                    'ele1_charge' : float,
                    'ele1_ECIDSResult' : float,
                    'ele1_d0' : float,
                    'ele1_d0Sig' : float,
                    'ele1_z0' : float,
                    'ele1_z0Sig' : float,
                    'ele1_f3' : float,
                    'ele1_f1' : float,
                    'ele1_Reta' : float,
                    'ele1_Rphi' : float,
                    'ele1_Rhad' : float,
                    'ele1_Rhad1' : float,
                    'ele1_weta2' : float,
                    'ele1_Eratio' : float,
                    'ele1_TRTPID' : float,
                    'ele1_dPOverP' : float,
                    'ele1_deltaEta1' : float,
                    'ele1_deltaPhiRescaled2' : float,
                    'ele1_ptvarcone20' : float,
                    'ele1_topoetcone20' : float,
                    'ele1_topoetcone40' : float,
                    'ele1_ptcone20_TightTTVALooseCone_pt1000' : float,
                    'ele1_ptcone20_TightTTVALooseCone_pt500' : float,
                    'ele1_ptvarcone20_TightTTVALooseCone_pt1000' : float,
                    #'ele1_ptvarcone20_TightTTVA_pt1000' : float,
                    'ele1_ptvarcone30_TightTTVALooseCone_pt1000' : float,
                    'ele1_ptvarcone30_TightTTVALooseCone_pt500' : float,
                    #'ele1_ptvarcone30_TightTTVA_pt1000' : float,
                    #'ele1_ptvarcone30_TightTTVA_pt500' : float,
                    'ele1_ptvarcone40_TightTTVALooseCone_pt1000' : float,
                    'ele1_topoetcone20ptCorrection' : float,
                    'ele1_expectInnermostPixelLayerHit' : int,
                    'ele1_expectNextToInnermostPixelLayerHit' : int,
                    'ele1_nTracks' : int,
                    'ele1_numberOfInnermostPixelHits' : int,
                    'ele1_numberOfPixelHits' : int,
                    'ele1_numberOfSCTHits' : int,
                    'ele1_numberOfTRTHits' : int,
                    'ele1_core57cellsEnergyCorrection' : float,
                    'ele2_truthParticle' : int,
                    'ele2_truthPdgId' : int,
                    'ele2_truthType' : int,
                    'ele2_truthOrigin' : int,
                    'ele2_LHLoose' : int,
                    'ele2_LHMedium' : int,
                    'ele2_LHTight' : int,
                    'ele2_trigger' : int,
                    'ele2_m' : float,
                    'ele2_e' : float,
                    'ele2_et' : float,
                    'ele2_eta' : float,
                    'ele2_phi' : float,
                    'ele2_charge' : float,
                    'ele2_ECIDSResult' : float,
                    'ele2_d0' : float,
                    'ele2_d0Sig' : float,
                    'ele2_z0' : float,
                    'ele2_z0Sig' : float,
                    'ele2_f3' : float,
                    'ele2_f1' : float,
                    'ele2_Reta' : float,
                    'ele2_Rphi' : float,
                    'ele2_Rhad' : float,
                    'ele2_Rhad1' : float,
                    'ele2_weta2' : float,
                    'ele2_Eratio' : float,
                    'ele2_TRTPID' : float,
                    'ele2_dPOverP' : float,
                    'ele2_deltaEta1' : float,
                    'ele2_deltaPhiRescaled2' : float,
                    'ele2_ptvarcone20' : float,
                    'ele2_topoetcone20' : float,
                    'ele2_topoetcone30' : float,
                    'ele2_topoetcone40' : float,
                    'ele2_ptcone20_TightTTVALooseCone_pt1000' : float,
                    'ele2_ptcone20_TightTTVALooseCone_pt500' : float,
                    'ele2_ptvarcone20_TightTTVALooseCone_pt1000' : float,
                    #'ele2_ptvarcone20_TightTTVA_pt1000' : float,
                    'ele2_ptvarcone30_TightTTVALooseCone_pt1000' : float,
                    'ele2_ptvarcone30_TightTTVALooseCone_pt500' : float,
                    #'ele2_ptvarcone30_TightTTVA_pt1000' : float,
                    #'ele2_ptvarcone30_TightTTVA_pt500' : float,
                    'ele2_ptvarcone40_TightTTVALooseCone_pt1000' : float,
                    'ele2_topoetcone20ptCorrection' : float,
                    'ele2_topoetconecoreConeEnergyCorrection' : float,
                    'ele2_secondLambdaCluster' : float,
                    'ele2_lateralCluster' : float,
                    'ele2_longitudinalCluster' : float,
                    'ele2_fracMaxCluster' : float,
                    'ele2_secondRCluster' : float,
                    'ele2_centerLambdaCluster' : float,
                    'ele2_expectInnermostPixelLayerHit' : int,
                    'ele2_expectNextToInnermostPixelLayerHit' : int,
                    'ele2_nTracks' : int,
                    'ele2_numberOfInnermostPixelHits' : int,
                    'ele2_numberOfPixelHits' : int,
                    'ele2_numberOfSCTHits' : int,
                    'ele2_numberOfTRTHits' : int,
                    'ele2_core57cellsEnergyCorrection' : float}
elif args.PartType == 'muo':
    column_dtype = {'NvtxReco' : int,
                    'correctedScaledAverageMu' : float,
                    'invM' : float,
                    'pt' : float,
                    'eta' : float,
                    'phi' : float,
                    'type' : int,
                    'muo1_truthPdgId' : int , 
                    'muo1_truthType' : int,
                    'muo1_truthOrigin' : int ,
                    'muo1_LHLoose' : int ,
                    'muo1_LHMedium' : int,
                    'muo1_LHTight' : int,
                    'muo1_trigger' : int,
                    'muo1_pt' : float,
                    'muo1_eta' : float,
                    'muo1_phi' : float,
                    'muo1_charge' : int,
                    'muo1_delta_z0' : float,
                    'muo1_muo_delta_z0_sin_theta' : float,
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
                    'muo2_truthPdgId' : int , 
                    'muo2_truthType' : int,
                    'muo2_truthOrigin' : int ,
                    'muo2_LHLoose' : int ,
                    'muo2_LHMedium' : int,
                    'muo2_LHTight' : int,
                    'muo2_trigger' : int,
                    'muo2_pt' : float,
                    'muo2_eta' : float,
                    'muo2_phi' : float,
                    'muo2_charge' : int,
                    'muo2_delta_z0' : float,
                    'muo2_muo_delta_z0_sin_theta' : float,
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
                    'muo2_delta_z0_sin_theta' : float}

column_names = list(column_dtype.keys())


#============================================================================
# Main Signal selection
#============================================================================
# Total counters for signal selection diagram
histNtype_total = np.zeros((1,3))
histTrigPass_total = np.zeros((1,2))
All_masses, All_SBT, = [], []


# Create file name and check if the file already exists
filename = f'{args.tag}.h5'
if os.path.isfile(args.outdir + filename):
    log.error(f'Output file already exists - please remove yourself')
    quit()

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports _main_here)
pool = multiprocessing.Pool(processes = args.max_processes)

for path in args.paths:
    # Count which file we made it to
    counter +=1

    # Read hdf5 data to get number of events
    hf_read = h5py.File(path,'r')

    print(hf_read.keys())

    # Number of reconstructed verteces
    N = hf_read['NvtxReco'].shape[0]

    print('N= ', N)

    # Split indices into equally-sized batches
    index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
    index_ranges = zip(index_edges[:-1], index_edges[1:])
    
    results = pool.map(multiExtract, [(i, counter, path, start, stop) for i, (start, stop) in enumerate(index_ranges)])
    results_np = np.array(results)

    # Concatenate resulting data from the multiple converters
    data = np.concatenate(results_np[:,0])
    All_masses = np.append(All_masses,data[:,column_names.index('invM')])
    All_SBT = np.append(All_SBT,data[:,column_names.index('type')])

    # Concatenate data and add to total
    histNtype = np.concatenate(results_np[:,1], axis = 0)
    histNtype_total = histNtype_total + np.sum(histNtype, axis = 0)

    # Concatenate data and add tot total
    histTrigPass = np.concatenate(results_np[:,2], axis = 0)
    histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis = 0)

    # Print the total event count in the file
    log.info('Data shape: {}'.format(data.shape))
    
    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)
    
# Create and save figure of signal selection
plotHistogram(histVal=histNtype_total[0],
              fname=args.tag+'_sigSelDiag.png',
              names=[ 'Background','Signal', 'Trash'],
              title = f'Signal selection ({args.tag})',
              xlabel = 'Selection types')
plotHistogram(histVal=histTrigPass_total[0],
              fname=args.tag+'_trigPassDiag.png',
              names=['No trigger in event', 'At least one trigger in event'],
              title = f'Events that passes trigger ({args.tag})',
              xlabel = '')

MassHistogram(HistVal = All_masses, 
              DatType = All_SBT,
              fname = args.tag+'_InvMass.png', 
              title = f'Invariant mass of the {args.PartType} pair', 
              xlabel = 'Invariant mass', 
              min = 1, 
              max = 150, 
              nbins = 151-1)

sec = timedelta(seconds=time() - t_start)
log.info(f'Extraction finished. Time spent: {str(sec)}')

