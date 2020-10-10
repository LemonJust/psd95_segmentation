"""
Functions to load and manipulate cohort.
"""

import numpy as np
import pandas as pd
from numpy import load
from numpy import genfromtxt
import json
from synspy.analyze.util import load_segment_status_from_csv, dump_segment_info_to_csv
import tifffile as tif
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import cKDTree

import os.path
from os import path


def get_cohort(imgpare_csv, cohort_key_csv, img_resolution):
    """
    Get all the info from the image-pair-study csv. Synapse coordinates are in physical
    space, so you need to know the original resolution to go to pixel space.

    Parameters:
        imgpare_csv (string) : path and filename of the csv.
        img_resolution : image resolution in order XYZ. Set to None if you don't know it.

    Returns:
        TODO : change this describtion, it's wrong
        stu (dict) : array of centroids in tiff space in pixels, in order ZYX.
        Dict structure  0) study type : LR / NL / US / NS / CS
                        1) fish ID
                        2) lost / gain / uncB / uncA
                        3) xyz / int_core / int_vcn / xyz_pix
    """
    cohort_df, study_id = load_imgpairstudy_csv(imgpare_csv)

    # populate syn
    syn = import_imgpairstudy(study_id[0], cohort_df, resolution=img_resolution)

    for fish_id in study_id[1:]:
        syn = import_imgpairstudy(fish_id, cohort_df, syn=syn,
                                  resolution=img_resolution)

    # populate stu
    key_df = pd.read_csv(cohort_key_csv)

    stu = {}
    for study_type in ['LR', 'NL', 'NS', 'US', 'CS']:
        print(f'Doing {study_type}')
        stu[study_type] = {'study_id': [study_id
                                        for study_id in
                                        key_df.loc[key_df['study_type'] == study_type,
                                                   'study_id'].values[:]],
                           'subject_id':
                               [cohort_df.loc[cohort_df['study_id'] == study_id,
                                              'subject'].values[0]
                                for study_id in
                                key_df.loc[key_df['study_type'] == study_type,
                                           'study_id'].values[:]]}
    return syn, stu


def import_imgpairstudy(fish_id, cohort_df, syn=None, resolution=None):
    """
    Imports the coordinates , xyz in pixels, and intensity , int_core & int_vcn,
    from the image-pair-study csv produced by the synspy.
    To get the cohort_df run load_imgpairstudy_csv.

    resolution ( xyz) : get coordinates in pixels if resolution is provided
    syn : if None creates a new syn, if given adds to existing dictionary
    TODO : finish describtion
    """

    def create_syn_type(syn, df, syn_type, xyz_cols, int_core_col, int_vcn_col,
                        resolution=None):
        """ Helper function : populates syn[fish_id][syn_type] dictionary"""
        syn[fish_id][syn_type] = {}
        syn[fish_id][syn_type]["xyz"] = df.loc[:, xyz_cols].values
        syn[fish_id][syn_type]["int_core"] = df.loc[:, int_core_col].values
        syn[fish_id][syn_type]["int_vcn"] = df.loc[:, int_vcn_col].values
        # get coordinates in pixels if resolution is provided
        if resolution is not None:
            syn[fish_id][syn_type]["xyz_pix"] = syn[fish_id][syn_type]["xyz"] / resolution
        return syn

    # create new syn if none provided
    if syn is None:
        syn = {}
    # create fish entry
    syn[fish_id] = {}
    # get all the data for this fish
    is_unch = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 0.0)]
    is_lost = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 1.0)]
    is_gain = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 2.0)]
    # populate syn
    syn = create_syn_type(syn, is_lost, "lost", ["x1", "y1", "z1"], ["core1"],
                          ["vcn1"], resolution)
    syn = create_syn_type(syn, is_gain, "gain", ["x2", "y2", "z2"], ["core2"],
                          ["vcn2"], resolution)
    syn = create_syn_type(syn, is_unch, "uncB", ["x1", "y1", "z1"], ["core1"],
                          ["vcn1"], resolution)
    syn = create_syn_type(syn, is_unch, "uncA", ["x2", "y2", "z2"], ["core2"],
                          ["vcn2"], resolution)
    return syn


def get_imagepair(study_id, info_filename=None):
    """
    Returns tp1 and tp2 image names for study_id (as in cohort scv)
    """
    if info_filename is None:
        info_filename = 'D:/Code/repos/psd95_segmentation/data/raw/' \
                        'SynapticPairStudy_db.csv '
    info_df = pd.read_csv(info_filename)
    images = info_df.loc[info_df['RID'] == study_id, ['Image 1', 'Image 2']].values[:]

    return images[0]


def load_imgpairstudy_csv(csv_filename):
    """
    reads image-pair-study csv as a data frame and extracts study ids.
    """
    cohort_df = pd.read_csv(csv_filename)
    # get study ids
    study_id = cohort_df.drop_duplicates(subset='study_id', inplace=False)[
                   'study_id'].values[:]

    return cohort_df, study_id


def get_raw_coord(syn, img_table, ips_table, resolution):
    """
    Returns xyz coordinates in raw image space.
    resolution (list) : resolution of the original image in xyz: [0.26, 0.26, 0.4]
    """
    img_df = pd.read_csv(img_table)
    ips_df = pd.read_csv(ips_table)

    # def undo_transform():

    for stu_RID in syn.keys():
        img_tp1 = get_imagepair(stu_RID)[0]

        # get tp2--> tp1 transform
        t2_to_tp1 = ips_df.loc[ips_df['Image 1'] == img_tp1, 'Alignment'].values[0]
        t2_to_tp1 = np.array(json.loads(t2_to_tp1))
        # this transform is saved with the 0001 as row 4, so transpose
        t2_to_tp1 = np.transpose(t2_to_tp1)

        # get tp1 --> Template transform
        to_tem = img_df.loc[img_df['RID'] == img_tp1, 'Alignment'].values[0]
        to_tem = np.array(json.loads(to_tem))

        # lost
        coord = syn[stu_RID]['lost']['xyz']
        syn[stu_RID]['lost']['raw_xyz'] = np.matmul(np.c_[coord, np.ones(len(coord))],
                                                    np.linalg.inv(to_tem))[:, 0:3]
        syn[stu_RID]['lost']['raw_xyz_pix'] = np.round(syn[stu_RID]['lost']['raw_xyz'] /
                                                       resolution) + [0, 0, 1]
        # uncB
        coord = syn[stu_RID]['uncB']['xyz']
        syn[stu_RID]['uncB']['raw_xyz'] = np.matmul(np.c_[coord, np.ones(len(coord))],
                                                    np.linalg.inv(to_tem))[:, 0:3]
        syn[stu_RID]['uncB']['raw_xyz_pix'] = np.round(syn[stu_RID]['uncB']['raw_xyz'] /
                                                       resolution) + [0, 0, 1]
        # gain
        coord = syn[stu_RID]['gain']['xyz']
        coord = np.matmul(np.c_[coord, np.ones(len(coord))],
                          np.linalg.inv(to_tem))[:, 0:3]
        syn[stu_RID]['gain']['raw_xyz'] = np.matmul(np.c_[coord, np.ones(len(coord))],
                                                    np.linalg.inv(t2_to_tp1))[:, 0:3]

        syn[stu_RID]['gain']['raw_xyz_pix'] = np.round(syn[stu_RID]['gain']['raw_xyz'] /
                                                       resolution) + [0, 0, 1]
        # uncA
        coord = syn[stu_RID]['uncA']['xyz']
        coord = np.matmul(np.c_[coord, np.ones(len(coord))],
                          np.linalg.inv(to_tem))[:, 0:3]
        syn[stu_RID]['uncA']['raw_xyz'] = np.matmul(np.c_[coord, np.ones(len(coord))],
                                                    np.linalg.inv(t2_to_tp1))[:, 0:3]
        syn[stu_RID]['uncA']['raw_xyz_pix'] = np.round(syn[stu_RID]['uncA']['raw_xyz'] /
                                                       resolution) + [0, 0, 1]

    return syn
