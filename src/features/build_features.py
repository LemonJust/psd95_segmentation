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


def get_percentiles(data):
    """
    The function to calculate 11 percentiles of data distribution.

    Calculates the following percentiles :
    (0 = min, 10 , 20 , 30, 40, 50, 60, 70, 80, 90, 100=max ).

    Parameters:
        data (numpy array): distribution to get percentiles from.

    Returns:
        numpy array: percentiles
    """
    prc_list = np.arange(0, 110, 10)
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
        numpy array : 11 percentiles for each files ( one file in one row)
    """
    num_prc = 11
    prc_img = np.zeros((len(files), num_prc))
    for count, file_name in enumerate(files):
        print(count)
        my_img = read_channel(channel, file_name.as_posix(), padding)
        prc_img[count, :] = get_percentiles(my_img)
    return prc_img
