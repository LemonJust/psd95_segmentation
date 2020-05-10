# -*- coding: utf-8 -*-
# %% Import
from pathlib import Path
import numpy as np
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import requests


# %% Define function
def read_channel(channel, file_name, padding):
    """
    Reads an image in one channel (red or green) from a file with two channels side-by-side.

    Parameters:
        channel (string): which channel to get. Either "red" or "green".
        file_name (string): full path to the image file.
        padding (3x3 array of int): how many pixels to crop away on each side, in the format
        [[z_top, z_bottom],[y_top,y_bottom],[x_left,x_right]].

    Returns:
        numpy array: image volume in ZYX order.
    """
    padding = np.array(padding)
    padding[:, 0] = -padding[:, 0]

    if channel == "red":
        roi = np.array([[0, 251], [0, 2048], [1025, 2048]]) - padding  # Z, Y, X
    elif channel == "green":
        roi = np.array([[0, 251], [0, 2048], [0, 1024]]) - padding
    else:
        raise ValueError('The channel should be either "red" or "green".')

    img = tif.imread(file_name)
    chn_img = img[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]]

    return chn_img

# %% Define function
def get_image_name(img_path):
    """
    The function to extract the name of the image from path.

    Returns the name of the image without the extension and path.
    Tailored for the image name format: "path-to-file/Image_1-0HN6.ome.tif",
    where "1-0HN6" is the image name being extracted.

    Parameters:
        img_path (pathlib.Path): path to image file

    Returns:
        string: image name
    """
    return img_path.as_posix().split(sep='/')[-1].split(sep='.')[0].split(sep='_')[-1]


def download_csv():
    """
    The function to download csv files.

    Shows box plots of data in the data frame, grouping y_axis according x_axis label.

    Parameters:
        df (pandas.DataFrame): data frame with the data
        x_axis (string): the name of the df column to use for grouping.
        y_axis (string): the name of the df column to group according to x_axis.

    Returns:
        ???: figure identifier TODO: fill in details
    """


