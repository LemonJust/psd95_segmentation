from pathlib import Path
import numpy as np
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import requests


def get_percentiles(data):
    """
    The function to calculate 11 percentiles of data distribution.

    Calculates the following percentiles :
    (0 = min, 10 , 20 , 30, 40, 50, 60, 70, 80, 90, 100=max ).

    Parameters:
        data (array): distribution to get percentiles from.

    Returns:
        numpy array: percentiles
    """
    prc_list = np.arange(0, 110, 10)
    return np.percentile(data, prc_list)


def extract_img_percentiles():
    channel = "green"
    padding = [[0, 0], [50, 500], [20, 20]]

    pallium_path = Path("D:/Code/repos/psd95_segmentation/data/raw/pallium")
    files = [x for x in pallium_path.glob('*tif') if x.is_file()]

    num_prc = 11
    prc_green = np.zeros((len(files), num_prc))

    for count, file_name in enumerate(files):
        print(count)
        my_img = read_channel(channel, file_name.as_posix(), padding)
        prc_green[count, :] = get_percentiles(my_img)