#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions, used throughout different files.
"""
###############################################################################
# Import packages and functions
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import lightgbm as lgb
import os
from datetime import timedelta
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error

###############################################################################
def h5ToDf(filename):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    with h5py.File(filename, "r") as hf :
        d = {}
        for name in list(hf.keys()):
            d[name] = np.array(hf[name][:])
        df = pd.DataFrame(data=d)
    return(df)

def h5ToNp(filename):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    with h5py.File(filename, "r") as hf :
        columns = list(hf.keys())
        n_rows = len(hf[columns[0]][:])
        arr = np.zeros((n_rows,len(columns)))
        for i, name in enumerate(columns):
            arr[:,i] = np.array(hf[name][:])
    return arr, columns

def getX(hf,columns):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    n_rows = len(hf[columns[0]][:])
    arr = np.zeros((n_rows,len(columns)))
    for i, name in enumerate(columns):
        arr[:,i] = np.array(hf[name][:])
    return arr


def mkdir (path):
    """
    Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        # print( "mdkir: Creating output directory:\n  {}".format(path) )
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass

    return
# Copy from: ~bhenckel/work/egamma_v2/scripts/MyFunctions/utils.py
def timediftostr(t):
    sec = timedelta(seconds=time() - t)
    #sec = sec - timedelta(microseconds=sec.microseconds)
    return str(sec)

# Copy from loading_functions (@author: stefan) from: ~bhenckel/work/egamma_v2/scripts/MyFunctions/loading_functions.py
def Build_Folder_Structure(directory_list):
    """ Create folders and parent folders recursively if they doesn't exist already."""
    for dirs in directory_list:
        try:
            os.makedirs(dirs)
        except OSError:
            continue

# Copy from: ~bhenckel/work/egamma_v2/scripts/MyFunctions/utils.py
def load_data(filepath, var_list, subsample, sets):
    data_dict = {}
    with h5py.File(filepath, 'r') as hf:
        for set in sets:

            dtype = [(var, str(hf[f'{set}/{var}'].dtype)) for var in var_list]
            nevents = hf[f'{set}/{var_list[0]}'].shape[0]

            if subsample[set] > nevents:
                subsample[set] = 1.0

            if subsample[set] > 1.0:
                data = np.recarray( (int(subsample[set]),), dtype = dtype )
            else:
                data = np.recarray( (int(nevents*subsample[set]),), dtype = dtype)

            for var in var_list:
                data[var] = hf[f'{set}/{var}'][:len(data)]

            data_dict[set] = data

    return data_dict

### For printing: ###
def header(str):
    print(f'')
    print(f'')
    print(f'--- {str} ---')

def footer(str,t):
    print(f'')
    print(f'Finished {str} - Time spent: {timediftostr(t)}.')


def print_col_names(all_cols,train_cols,label_cols):
    print(f'All columns in data:')
    for name in all_cols:
        print(f'        {name}')
    print(f'Training variables:')
    for name in train_cols:
        print(f'        {name}')
    print(f'Label variable:')
    print(f'        {label_cols}')
    print(f'')

###############################################################################
def train_valid_test_data(X, y, test_size=0.2, valid_size=0.2, rand_state=0, verbose=False):
    """
    Separate data into training, validation and test set, with truth variable {truth_var}.
    """
    test_valid_size = test_size + valid_size
    test_valid_split = valid_size/test_valid_size

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size = test_valid_size, random_state = rand_state)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size = test_valid_split, random_state = rand_state)

    if verbose:
        print(f"Shape of data set: y = {np.shape(y)}, X = {np.shape(X)}\n")
        print(f"Separate with test size {test_size} and validation size {valid_size}:")
        print(f"        Shape of Training set:   y = {np.shape(y_train)}, X = {np.shape(X_train)}")
        print(f"        Shape of Validation set: y = {np.shape(y_valid)}, X = {np.shape(X_valid)}")
        print(f"        Shape of Test set:       y = {np.shape(y_test)}, X = {np.shape(X_test)}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_test_data(x, y, test_size=0.2, rand_state=0, verbose=False):
    """
    Separate data into training, validation and test set, with truth variable {truth_var}.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = rand_state)

    if verbose:
        print(f'Removed variables: {drop_var}, truth variable: {truth_var}')
        print(f"        Shape of data set:       y = {np.shape(y)}, X = {np.shape(X)}")
        print(f"Separate with test size {test_size}:")
        print(f"        Shape of Training set:   y = {np.shape(y_train)}, X = {np.shape(X_train)}")
        print(f"        Shape of Test set:       y = {np.shape(y_test)}, X = {np.shape(X_test)}")

    return X_train, X_test, y_train, y_test


###############################################################################
def accuracy(pred, true):
    # Round prediction
    pred = np.rint(pred)
    # Accuracy
    acc = np.sum(pred==true)/np.size(true)
    return acc

###############################################################################

def auc_eval(preds, train_data):
    auc_value = -roc_auc_score(train_data.get_label(), preds, sample_weight = train_data.get_weight())
    return 'auc_eval', auc_value, False


###############################################################################
# Light GBM Random search with Cross Validation
def HyperOpt_RandSearch(lgb_train, params_set, params_grid,learning_rate=0.1, n_iter=10, n_fold=5, n_boost_round=200, e_stop_round=10, verbose=False):
    # Random seed
    rng = np.random.RandomState(2)
    print(f"learning_rate = {learning_rate}, n_fold = {n_fold}, n_boost_round = {n_boost_round}, e_stop_round = {e_stop_round}")

    # Create parameter lists by drawing n_iter parameters from param_grid
    param_list = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=rng))
    rounded_list = [dict((k, round(v, 3)) for (k, v) in d.items()) for d in param_list]

    # Array for storing cv_scores
    cv_scores = np.zeros((n_iter,6))

    for i in range(n_iter):
        # Header for printing and time calculation
        header = f"ITERATION [{i}/{n_iter}] "
        print(header+f"Optimizing...")
        start_time = time()

        # Parameters for this iteration
        params_cv = params_set
        params_cv.update(rounded_list[i])
        params_cv.update(learning_rate=learning_rate)

        # Check if 'max_depth' and 'num leaves' combination is valid
        if (params_cv['max_depth']>0) & (params_cv['num_leaves'] > 2**(params_cv['max_depth'])):
            print(header + f"IGNORE 'num_leaves' > 2^('max_depth'): {params_cv['num_leaves']} > 2^{params_cv['max_depth']} = {2**(params_cv['max_depth'])}")
            cv_scores[i] = np.array([999,999,999,params_cv['num_leaves'],params_cv['min_data_in_leaf'],params_cv['max_depth']])
            continue

        # Perform cross validation
        cv_results = lgb.cv(params_cv,
                            lgb_train,
                            feval = auc_eval,
                            nfold=n_fold,
                            num_boost_round=n_boost_round,
                            verbose_eval=verbose,
                            early_stopping_rounds=e_stop_round,
                            shuffle=False)
        # Save the best (ie. last) result
        best_results = np.array([cv_results['auc_eval-mean'][-1],
                                 cv_results['auc_eval-stdv'][-1],
                                 len(cv_results['auc_eval-mean']),
                                 params_cv['num_leaves'],
                                 params_cv['min_data_in_leaf'],
                                 params_cv['max_depth']
                                 ])
        # Update cv_scores
        cv_scores[i] = best_results
        elapsed_time = time() - start_time

        print(header+f"Cross validation done [{elapsed_time:.2f}s]- Best CV score: {best_results[0]:.6f} +/- {best_results[1]:.6f} - Best num_boost_round: {int(best_results[2])}")

    # Best iteration
    best = np.argmin(cv_scores[:,0])

    # Create a DataFrame with the cv_scores and the dictionary
    df = pd.DataFrame(cv_scores,columns=['cv_mean', 'cv_stdv', 'num_boost_round','num_leaves','min_data_in_leaf','max_depth'])
    df.insert(6, "params", rounded_list, True)

    return (best, df)

###############################################################################
def HeatMap_rand(fig,ax,val_mean, val_std, val1, val2, range1, range2):
    mean = np.zeros((len(range1),len(range2)))
    std = np.zeros((len(range1),len(range2)))

    for i in range(len(val_mean)):
        if val_mean[i]==999:
            continue
        arg_x = np.argwhere(np.round(range1,decimals=4)==np.round(val1[i],decimals=4))[0][0]
        arg_y = np.argwhere(np.round(range2,decimals=4)==np.round(val2[i],decimals=4))[0][0]

        mean[arg_x,arg_y] = val_mean[i]
        std[arg_x,arg_y] = val_std[i]
    Mean = np.ma.masked_where(mean == 0, mean)

    im = ax.imshow(Mean,aspect='auto')

    # We want to show all ticks...
    ax.set_yticks(np.arange(len(range1)))#,int(len(range1)/5)))
    ax.set_xticks(np.arange(len(range2)))#,int(len(range2)/5)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(range1)#[::int(len(range1)/5)])
    ax.set_xticklabels(range2)#[::int(len(range2)/5)])
    # Show only some of the ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels()[::5], visible=True)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_yticklabels()[::int((len(range1)-1)/8)], visible=True)

    fig.colorbar(im, ax=ax)

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(range1)):
    #     for j in range(len(range2)):
    #         if mean[i,j]!=0:
    #             t = np.round(mean[i, j],6)
    #             ax.text(j, i, t, ha="center", va="top", color="k")

