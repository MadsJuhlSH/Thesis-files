#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to return LGBM score from a model

"""
import lightgbm as lgb
from scipy.special import logit


def eIsoScore_et(gbm,data,eleNr,n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    f'ele{eleNr}_et',
                    f'ele{eleNr}_ptvarcone20',
                    f'ele{eleNr}_topoetcone20',
                    f'ele{eleNr}_topoetcone40']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def eIsoScore_eta(gbm,data,eleNr,n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    f'ele{eleNr}_eta',
                    f'ele{eleNr}_ptvarcone20',
                    f'ele{eleNr}_topoetcone20',
                    f'ele{eleNr}_topoetcone40']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def eIsoScore_etaRel(gbm,data,eleNr,n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    f'ele{eleNr}_eta',
                    f'ele{eleNr}_ptvarcone20_rel',
                    f'ele{eleNr}_topoetcone20_rel',
                    f'ele{eleNr}_topoetcone40']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def eIsoScore(gbm, data, eleNr, n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    f'ele{eleNr}_ptvarcone20_rel',
                    f'ele{eleNr}_ptvarcone40_rel',
                    f'ele{eleNr}_topoetcone20_rel',
                    f'ele{eleNr}_topoetcone40_rel',
                    f'ele{eleNr}_topoetcone20ptCorrection',
                    ]
    """
    training_var = ["NvtxReco",
                    "correctedScaledAverageMu",
                    f"ele{eleNr}_ptvarcone20_rel",
                    f"ele{eleNr}_ptvarcone40_rel",
                    f"ele{eleNr}_topoetcone20_rel",
                    f"ele{eleNr}_topoetcone40_rel",
                    f"ele{eleNr}_expectInnermostPixelLayerHit",
                    f"ele{eleNr}_expectNextToInnermostPixelLayerHit",
                    f"ele{eleNr}_core57cellsEnergyCorrection",
                    f"ele{eleNr}_nTracks",
                    f"ele{eleNr}_numberOfInnermostPixelHits",
                    f"ele{eleNr}_numberOfPixelHits",
                    f"ele{eleNr}_numberOfSCTHits",
                    f"ele{eleNr}_numberOfTRTHits",
                    f"ele{eleNr}_topoetcone20ptCorrection"]"""
    score = gbm.predict(data[training_var][:], num_iteration = gbm.best_iteration, n_jobs = n_jobs)
    return logit(score)

def ePidScore(gbm,data,eleNr,n_jobs):
    training_var = [f'ele{eleNr}_d0',
                    f'ele{eleNr}_d0Sig',
                    f'ele{eleNr}_Rhad1',
                    f'ele{eleNr}_Rhad',
                    f'ele{eleNr}_f3',
                    f'ele{eleNr}_weta2',
                    f'ele{eleNr}_Rphi',
                    f'ele{eleNr}_Reta',
                    f'ele{eleNr}_Eratio',
                    f'ele{eleNr}_f1',
                    f'ele{eleNr}_dPOverP',
                    f'ele{eleNr}_deltaEta1',
                    f'ele{eleNr}_deltaPhiRescaled2',
                    f'ele{eleNr}_expectInnermostPixelLayerHit',
                    f'ele{eleNr}_expectNextToInnermostPixelLayerHit',
                    f'ele{eleNr}_core57cellsEnergyCorrection',
                    f'ele{eleNr}_nTracks',
                    f'ele{eleNr}_numberOfInnermostPixelHits',
                    f'ele{eleNr}_numberOfPixelHits',
                    f'ele{eleNr}_numberOfSCTHits',
                    f'ele{eleNr}_numberOfTRTHits',
                    f'ele{eleNr}_TRTPID'
                    ]
    """training_var = [f'ele{eleNr}_d0',
                    f'ele{eleNr}_d0Sig',
                    f'ele{eleNr}_Rhad1',
                    f'ele{eleNr}_Rhad',
                    f'ele{eleNr}_f3',
                    f'ele{eleNr}_weta2',
                    f'ele{eleNr}_Rphi',
                    f'ele{eleNr}_Reta',
                    f'ele{eleNr}_Eratio',
                    f'ele{eleNr}_f1',
                    f'ele{eleNr}_dPOverP',
                    f'ele{eleNr}_deltaEta1',
                    f'ele{eleNr}_deltaPhiRescaled2',
                    f'ele{eleNr}_TRTPID']"""
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def mIsoScore(gbm, data, muoNr, n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    #'muo_etcon20',
                    f'muo{muoNr}_ptvarcone20_rel',
                    f'muo{muoNr}_ptvarcone40_rel',
                    f'muo{muoNr}_etconecoreConeEnergyCorrection'
                    #'muo_topoetconecoreConeEnergyCorrection',
                    ]
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def mPidScore(gbm, data, muoNr, n_jobs):
    training_var = [f'muo{muoNr}_priTrack_d0',
                    f'muo{muoNr}_priTrack_d0Sig',
                    f'muo{muoNr}_numberOfPrecisionHoleLayers', 
                    f'muo{muoNr}_numberOfPrecisionLayers',
                    f'muo{muoNr}_quality',
                    #'muo_ET_TileCore',
                    f'muo{muoNr}_MuonSpectrometerPt',
                    #'muo_deltatheta_1',
                    f'muo{muoNr}_scatteringCurvatureSignificance', 
                    f'muo{muoNr}_scatteringNeighbourSignificance',
                    f'muo{muoNr}_momentumBalanceSignificance',
                    f'muo{muoNr}_EnergyLoss',
                    f'muo{muoNr}_energyLossType']
                    #f'muo{muoNr}_priTrack_numberOfPixelHits',
                    #f'muo{muoNr}_priTrack_numberOfSCTHits',
                    #f'muo{muoNr}_priTrack_numberOfTRTHits']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def fwdIsoScore(gbm,data,eleNr,n_jobs):
    training_var = ['ele2_topoetcone20',
                    'ele2_topoetcone30',
                    'ele2_topoetcone40',
                    'ele2_et',
                    'ele2_topoetconecoreConeEnergyCorrection',
                    'ele2_eta',
                    'NvtxReco']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def fwdPidScore(gbm,data,eleNr,n_jobs):
    training_var = ['ele2_centerLambdaCluster',
                    'ele2_fracMaxCluster',
                    'ele2_lateralCluster',
                    'ele2_longitudinalCluster',
                    'ele2_secondRCluster',
                    'ele2_secondLambdaCluster']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def pIsoScore(gbm, data,phoNr, n_jobs):
    #if there is only 1 photon eg Z->eegamma phoNr should be 0
    if phoNr==0:
        training_var = ['correctedScaledAverageMu',
                        'NvtxReco',
                        f'pho_et',
                        f'pho_topoetcone20',
                        f'pho_topoetcone40',
                        f'pho_ptvarcone20'
                        ]
    else:
        training_var = ['correctedScaledAverageMu',
                    'NvtxReco',
                    f'pho{phoNr}_et',
                    f'pho{phoNr}_topoetcone20',
                    f'pho{phoNr}_topoetcone40',
                    f'pho{phoNr}_ptvarcone20'
                    ]
    score = gbm.predict(data[training_var][:], num_iteration = gbm.best_iteration, n_jobs = n_jobs)
    return logit(score)
def pPidScore(gbm, data,phoNr, n_jobs):
    #if there is only 1 photon eg Z->eegamma phoNr should be 0
    if phoNr==0:
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
        """[f'pho_Rhad',
                        f'pho_Rhad1',
                        f'pho_Reta',
                        f'pho_weta2',
                        f'pho_Rphi',
                        f'pho_wtots1',
                        f'pho_Eratio',
                        'NvtxReco',
                        'correctedScaledAverageMu',
                        f'pho_ConversionRadius',
                        f'pho_ConversionType',
                        f'pho_f1',
                        f'pho_r33over37allcalo']"""
    else:
        training_var =[f'pho{phoNr}_isPhotonEMLoose',
                        f'pho{phoNr}_isPhotonEMTight',
                        f'pho{phoNr}_Rhad1',
                        f'pho{phoNr}_Rhad',
                        f'pho{phoNr}_weta2',
                        f'pho{phoNr}_Rphi',
                        f'pho{phoNr}_Reta',
                        f'pho{phoNr}_Eratio',
                        f'pho{phoNr}_f1',
                        f'pho{phoNr}_wtots1',
                        f'pho{phoNr}_DeltaE',
                        f'pho{phoNr}_weta1',
                        f'pho{phoNr}_fracs1',
                        #f'pho{phoNr}_ConversionType',
                        f'pho{phoNr}_ConversionRadius',
                        f'pho{phoNr}_VertexConvEtOverPt',
                        f'pho{phoNr}_VertexConvPtRatio',
                        f'pho{phoNr}_z0',
                        f'pho{phoNr}_z0Sig',
                        f'pho{phoNr}_core57cellsEnergyCorrection',
                        f'pho{phoNr}_r33over37allcalo']
    """ [f'pho{phoNr}_Rhad',
    f'pho{phoNr}_Rhad1',
    f'pho{phoNr}_Reta',
    f'pho{phoNr}_weta2',
    f'pho{phoNr}_Rphi',
    f'pho{phoNr}_wtots1',
    f'pho{phoNr}_Eratio',
    'NvtxReco',
    'correctedScaledAverageMu',
    f'pho{phoNr}_ConversionRadius',
    f'pho{phoNr}_ConversionType',
    f'pho{phoNr}_f1',
    f'pho{phoNr}_r33over37allcalo']"""
    score = gbm.predict(data[training_var][:], num_iteration = gbm.best_iteration, n_jobs = n_jobs)
    return logit(score)

# De to nedenunder er Helles, Undersøg deres resultater.
"""def phoPidScore(gbm,data,n_jobs):
    training_var = ['correctedScaledAverageMu',
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
                    'pho_wtots1']

    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)

def phoIsoScore(gbm,data,n_jobs):
    training_var = ['correctedScaledAverageMu',
                    'NvtxReco',
                    'pho_et',
                    'pho_ptvarcone20',
                    'pho_ptvarcone30',
                    'pho_ptvarcone40',
                    'pho_topoetcone20',
                    'pho_topoetcone30',
                    'pho_topoetcone40']
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
"""
def ppScore(gbm,data,n_jobs):
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
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def ZeeScore(gbm,data,n_jobs):
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

    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def ZeegScore(gbm,data,n_jobs):
    training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                'ee_score',
                'pho_pIso_score',
                'pho_pPid_score']

    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def ZmmScore(gbm,data,n_jobs):
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

    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def ZmmgScore(gbm,data,n_jobs):
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
    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)      
def HggScore(gbm, data, n_jobs):
    training_var = ['NvtxReco',
                'correctedScaledAverageMu',
                'pho_deltaZ0',
                'pho_deltaZ0sig',
                'pho1_pIso_score',
                'pho1_pPid_score',
                'pho2_pIso_score',
                'pho2_pPid_score'
                ]
    score = gbm.predict(data[training_var][:], num_iteration= gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
def ZPidScore(gbm,data,n_jobs):
    training_var = ['NvtxReco',
                    'correctedScaledAverageMu',
                    'ele1_d0',
                    'ele1_d0Sig',
                    'ele1_ECIDSResult',
                    'ele1_charge',
                    'ele1_ePid_score',
                    'ele1_eIso_score',
                    'ele2_d0',
                    'ele2_d0Sig',
                    'ele2_ECIDSResult',
                    'ele2_charge',
                    'ele2_ePid_score',
                    'ele2_eIso_score',
                    'ele2_isFwd']

    score = gbm.predict(data[training_var][:], num_iteration=gbm.best_iteration, n_jobs=n_jobs)
    return logit(score)
