
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import seaborn as sn
import os
import pickle
from datetime import datetime
from time import time
from pathlib import Path
from itertools import combinations


"""Collection of functions used to estimate motor delay from cognitive
tasks using a fixed-point equation model."""


# import matplotlib.pyplot as plt

def word_complexities_from_rt(rt_mat, correct_mat, word_mat, niter = 200):
    # Initialise empty matrices
    tau_max = np.nanmax(rt_mat)
    complexity_mat = np.ones_like(rt_mat)
    vocabulary = np.unique(word_mat)
    vocabulary = vocabulary[ ~np.isnan(vocabulary) ]
    
    # Get the 3D word indexer array
    vocabulary = vocabulary[:, None, None] # equivalent to np.expand_dims
    word_indexer = word_mat[None, :, :] == vocabulary
    word_apprearance = np.nansum(word_indexer, axis=(1,2))

    last_complexity_estimate = np.ones_like(vocabulary)
    perf_messages = []
    for i in range(niter):
        #print(i)
        t_start = time()
        (perf, rt_mat, tau_max, correct_mat, complexity_mat, 
        word_indexer, word_apprearance, 
        last_complexity_estimate, avg_delta_complexity) = update_complexity_estimate(
            rt_mat, tau_max, correct_mat, complexity_mat,
            word_indexer, word_apprearance, last_complexity_estimate
        )
        t = time() - t_start
        #print(f"[{i:4d}] avg(∂C) = {avg_delta_complexity:.15f}" 
        #                     f" | Elapsed: {t:.5f}s") 
    word_complexity = last_complexity_estimate
    return perf, word_complexity, complexity_mat, vocabulary, word_indexer


def update_complexity_estimate(rt_mat, tau_max, correct_mat, complexity_mat,
                               word_indexer, word_apprearance, last_complexity_estimate):
    perf = get_performance(rt_mat,tau_max,correct_mat,complexity_mat)
    complexity_mat_3D = np.where(word_indexer, 
                                 np.expand_dims((1-perf), 0),
                                 np.array([[[0]]]))
    # word_complexity = np.nansum(complexity_mat_3D, axis=(1,2)) / word_apprearance
    word_complexity = (nansum_on_first_second_axis(complexity_mat_3D) / 
                       word_apprearance)
    word_complexity = np.expand_dims(np.expand_dims(word_complexity, -1),-1)
    complexity_3D = np.where(word_indexer, word_complexity, 0)
    complexity_mat_new = nansum_on_zero_axis(complexity_3D)
    avg_delta_complexity = np.mean(
        np.abs(word_complexity - last_complexity_estimate) / 
        last_complexity_estimate
    )
    last_complexity_estimate = word_complexity
    return (perf, rt_mat, tau_max, correct_mat, complexity_mat_new, word_indexer,
            word_apprearance, word_complexity, avg_delta_complexity)
    


def get_performance(rt_mat,tau_max,correct_mat,complexity_mat):
    return (1 - (rt_mat/tau_max)) * correct_mat * complexity_mat


def nansum_on_zero_axis(arr):
    out_arr = np.zeros(arr.shape[1:])
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            out_arr[i,j] = np.nansum(arr[:,i,j])
    return out_arr
    

def nansum_on_first_second_axis(arr):
    out_arr = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        out_arr[i] = np.nansum(arr[i,:,:])
    return out_arr

def get_abilities(perf):
    non_na_values = ~np.isnan(perf)
    ability = np.nansum(perf, axis=1) / np.sum(non_na_values, axis=1)
    return ability


    
def ordinal_encode(str_mat):
    if isinstance(str_mat, pd.DataFrame):
        str_mat = str_mat.values
    tmp = str_mat.copy()
    if tmp.dtype == "O":
        dummy_value = "-1"
    else:
        dummy_value = -1
    tmp[pd.isna(tmp)] = dummy_value
    lookupTable, int_word_mat = np.unique(tmp, return_inverse=True)
    nan_label = (lookupTable == dummy_value).nonzero()[0]
    print(nan_label)
    tmp_mat = int_word_mat.reshape(str_mat.shape).astype(float)
    if len(lookupTable[lookupTable == dummy_value])>0:
        tmp_mat[tmp_mat == nan_label] = np.nan
        lookupTable[lookupTable == dummy_value] = np.nan
        
    return lookupTable, tmp_mat
