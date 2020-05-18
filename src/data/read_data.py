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

    Returns the name of the image without the extension and path to file.
    Tailored for the image name format: "path-to-file/Image_1-0HN6.ome.tif",
    where "1-0HN6" is the image name being extracted.

    Parameters:
        img_path (pathlib.Path): path to image file
        TODO : change to string path

    Returns:
        string: image name
    """
    return img_path.as_posix().split(sep='/')[-1].split(sep='.')[0].split(sep='_')[-1]


def download_url(url, save_file_name):
    """
    Downloads the link with redirect.

    Parameters:
        url (string): url to download.
        save_file_name (string): where to save the file.

    TODO : return status ?
    """
    data = requests.get(url, allow_redirects=True)
    open(save_file_name, 'wb').write(data.content)


def download_nuclear_csvs(df, img_names, nuc_path):
    """
    The function to download all nuclear csv files for the images in batch
    from set website : 'https://synapse.isrd.isi.edu' .

    Parameters:
        df (pandas.DataFrame): data frame with image names (in column 'Source Image')
            and urls to nuclear csv (in column 'Segments Filtered URL').
        img_names (List[str]): list of images wor which to get the csvs.
        nuc_path (string): path to the folder to save all the csvs to.

    TODO : return status ?
    """

    link_head = 'https://synapse.isrd.isi.edu'

    for count, img_name in enumerate(img_names):

        url = df.loc[df['Source Image'] == img_name, 'Segments Filtered URL'].iloc[0]
        if url.split('.')[1] == 'nuclei_only':
            url = link_head + url
            print(f"#{count} image {img_name} url {url}")

            download_url(url, nuc_path + img_name + '_nuclei_only.csv')


def download_synapse_csvs(df, img_names, syn_path):
    """
    The function to download all synapse csv files for the images in batch
    from set website : 'https://synapse.isrd.isi.edu' .
    Keep in mind you might not have access to some, then the downloaded files will be 1Kb size.

    Parameters:
        df (pandas.DataFrame): data frame with image names (in column 'Source Image'),
            segmentation RIDs (in column 'RID' )
            and urls to nuclear csv (in column 'Segments Filtered URL').
        img_names (List[str]): list of images wor which to get the csvs.
        syn_path (string): path to the folder to save all the csvs to.

    TODO : return status ?
    """

    link_head = 'https://synapse.isrd.isi.edu'

    for count, img_name in enumerate(img_names):
        segmentations = df.loc[df['Source Image'] == img_name,
                               ['RID', 'Segments Filtered URL']]
        for index, segmentation in segmentations.iterrows():
            rid = segmentation[0]
            url = segmentation[1]
            csv_type = url.split('.')[1]
            if csv_type == 'synapses_only' or csv_type == 'segments_only':
                url = link_head + url
                download_url(url, syn_path + img_name + '_' + rid + '_' + csv_type + '.csv')


def download_npz(df, rids, npz_path):
    """
    The function to download npz files for the rids in batch
    from set website : 'https://synapse.isrd.isi.edu' .

    Parameters:
        df (pandas.DataFrame): data frame with image names (in column 'Source Image'),
            segmentation RIDs (in column 'RID' )
            and urls to npz (in column 'Npz URL').
        rids (List[str]): list of rids for which to get the npzs.
        npz_path (string): path to the folder to save all the npzs to.

    TODO : return status ?
    """
    link_head = 'https://synapse.isrd.isi.edu'

    for count, rid in enumerate(rids):
        segmentations = df.loc[df['RID'] == rid,
                               ['Source Image', 'Npz URL']]
        for index, segmentation in segmentations.iterrows():
            img_name = segmentation[0]
            url = link_head + segmentation[1]

            download_url(url, npz_path + img_name + '_' + rid + '.npz')


def list_available_segmentations(syn_path):
    """
    Makes a list of segmentation RIDs available for each image
    by looking at the downloaded csv files for synapse segmentations.

    Tailored for segmentation csv name format : "syn_path/1-0HN6_1-0J02_synapses_only.csv",
    where "1-0HN6" is the image name and "1-0J02" is the segmentation RID.

    Parameters:
        syn_path (string) : path to the folder with all the csvs.

    Returns:
        (DataFrame) : segmentation RIDs per image ( row = image )
    """
    syn_files = [x for x in Path(syn_path).glob('*csv') if x.is_file()]
    num_syn_csv = {}
    csv_of_image = {}
    for csv_file in syn_files:
        img_of_csv = csv_file.as_posix().split(sep='/')[-1].split(sep='.')[0].split(sep='_')[0]
        rid_of_csv = csv_file.as_posix().split(sep='/')[-1].split(sep='.')[0].split(sep='_')[1]

        # keep track of how many csv for each image
        if img_of_csv in num_syn_csv:
            num_syn_csv[img_of_csv] = num_syn_csv[img_of_csv] + 1
            csv_of_image[img_of_csv].append(rid_of_csv)
        else:
            num_syn_csv[img_of_csv] = 1
            csv_of_image[img_of_csv] = [rid_of_csv]

    seg_rid_df = pd.DataFrame.from_dict(csv_of_image, orient='index', columns=['Syn RID 1', 'Syn RID 2'])
    return seg_rid_df


