from pathlib import Path
import numpy as np
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import requests

# import local package
from src.data.read_data import read_channel


def get_percentiles(data, step=10):
    """
    The function to calculate 11 percentiles of data distribution.

    Calculates the following percentiles :
    (0 = min, 10 , 20 , 30, 40, 50, 60, 70, 80, 90, 100=max ) for step = 10.

    Parameters:
        data (numpy array): distribution to get percentiles from.

    Returns:
        numpy array: percentiles
    """
    prc_list = np.arange(0, 100 + step, step)
    return np.percentile(data, prc_list)


def get_img_percentiles(channel, files, padding):
    """
    The function to calculate 11 percentiles of the intensity distribution for a list of images.

    Calculates the following percentiles for each image provided :
    (0 = min, 10 , 20 , 30, 40, 50, 60, 70, 80, 90, 100=max ).

    Parameters:
        channel (string): which channel to get. Either "red" or "green".
        files (array): images to get percentiles for.
        padding (3x3 array of int): how many pixels to crop away on each side, in the format
        [[z_top, z_bottom],[y_top,y_bottom],[x_left,x_right]].
    Returns:
        numpy array : 11 percentiles for each file ( one file in one row)
    """
    num_prc = 11
    prc_img = np.zeros((len(files), num_prc))
    for count, file_name in enumerate(files):
        print(count)
        my_img = read_channel(channel, file_name.as_posix(), padding)
        prc_img[count, :] = get_percentiles(my_img)
    return prc_img


def snr(signal, bg):
    """
    returns snr as (signal - bg)/bg
    """
    return (signal - bg) / bg


def subtract_bg(signal, bg):
    """
    returns normalised intensity as (signal - bg)
    """
    return signal - bg


def get_bg_stats_as_df(prc, bg_prc, study_id, int_df=None, snr_df=None, snr_hist_df=None):
    """
    Specifically for the dictionary data
    prc = syn[study_id]['lost']['prc']
    bg_prc : which prc to use as bg: 0 = min, 1 = 5% , 2 = 10%, 3 = 15%, 4 = 20%

    """
    all_snr = []

    if int_df is None:
        int_df = pd.DataFrame()
    if int_df is None:
        snr_df = pd.DataFrame()
    if snr_hist_df is None:
        snr_hist_df = pd.DataFrame()

    intensity_raw = prc[:, -1]  # last element = max
    bg = prc[:, bg_prc]

    intensity_bg_sub = subtract_bg(intensity_raw, bg)
    the_snr = snr(intensity_raw, bg)

    # take percentiles of that distribution
    int_prc = get_percentiles(intensity_bg_sub)
    snr_prc = get_percentiles(the_snr)

    # add to df
    int_df[f"{study_id}"] = int_prc
    snr_df[f"{study_id}"] = snr_prc

    new_hist_df = pd.DataFrame({f"{study_id}": the_snr})
    snr_hist_df = pd.concat([snr_hist_df, new_hist_df], axis=1)

    return int_df, snr_df, snr_hist_df, the_snr


def get_cdf(prc, bg_prc):  # , study_id, cdf_df=None):
    """
    Specifically for the dictionary data
    prc = syn[study_id]['lost']['prc']
    bg_prc : which prc to use as bg: 0 = min, 1 = 5% , 2 = 10%, 3 = 15%, 4 = 20%

    """
    int_cdf = np.zeros((51, 1))

    # if cdf_df is None:
    #    cdf_df = pd.DataFrame()

    intensity_raw = prc[:, -1]  # last element = max
    bg = prc[:, bg_prc]

    intensity_bg_sub = subtract_bg(intensity_raw, bg)

    for i_val, int_val in enumerate(range(0, 510, 10)):
        # get percentage of values below int_val
        int_cdf[i_val, 0] = np.mean(intensity_bg_sub < int_val)

    # add to df
    # cdf_df[f"{study_id}"] = int_cdf

    return int_cdf


def get_reduced_cdf(prc, bg_prc, reduce_by):  # , study_id, cdf_df=None):
    """
    Specifically for the dictionary data
    prc = syn[study_id]['lost']['prc']
    bg_prc : which prc to use as bg: 0 = min, 1 = 5% , 2 = 10%, 3 = 15%, 4 = 20%
    reduce_by : percentage to reduce the intensity
    """
    int_cdf = np.zeros((51, 1))

    intensity_raw = prc[:, -1]  # last element = max
    bg = prc[:, bg_prc]

    intensity_bg_sub = subtract_bg(intensity_raw, bg)

    snr_thr = min(snr(intensity_raw, bg))
    red_snr = (intensity_bg_sub * (1 - reduce_by / 100)) / bg

    # keep only the synapses that got "lost" due to reduction
    is_lost = red_snr < snr_thr
    intensity_lost = intensity_bg_sub[is_lost]

    # fraction lost
    frc_lost = intensity_lost.shape[0] / intensity_bg_sub.shape[0]

    for i_val, int_val in enumerate(range(0, 510, 10)):
        # get percentage of values below int_val
        if intensity_lost.size == 0:
            int_cdf[i_val, 0] = 0
        else:
            int_cdf[i_val, 0] = np.mean(intensity_lost < int_val)

    return int_cdf, frc_lost
