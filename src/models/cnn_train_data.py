import pandas as pd
from pathlib import Path
import numpy as np
import tifffile as tif

from src.features.synapse_features import get_centroids_and_labels, get_one_volume


class CnnTrainingData:
    """
    This class performs data consolidation, preprocess and etc. for CNN input
    """

    def __str__(self):
        return "training data"

    def __init__(self, data_folder=None, info_csv=None):
        """
        Initialises CnnTrainingData object.

        Parameters:
            data_folder (string) : path to the data folder,
                    Ex. : "D:/Code/repos/psd95_segmentation/data"
            info_csv (string) : path to the csv that has img - rid pairs
                    image names under 'Source Image' , rid names under 'Syn RID 1'
        """

        self.data_folder = Path(data_folder)
        self.info_df = pd.read_csv(info_csv)

        self.rid = []
        self.img = {}

        self.path_rid = {}
        self.path_img = {}
        self.path_nuc = {}
        self.path_npz = {}

        self.shape = []
        self.volumes = {}
        self.labels = {}

    def append_rids(self, *argv):
        """
        Adds rids to the list of data.
        Generate all useful info: img, paths to files TODO : finish this
        """

        for rid in argv:
            self.rid.append(rid)

            img = self.get_img(rid)
            self.img[rid] = img

            self.path_rid[rid] = self.generate_path(rid, 'rid')
            self.path_img[rid] = self.generate_path(rid, 'img')
            self.path_nuc[rid] = self.generate_path(rid, 'nuc')
            self.path_npz[rid] = self.generate_path(rid, 'npz')

    def get_img(self, rid):
        """
        Sets up everything for the chosen rid:
        get's paths to segmentation files, nuclear files, images
        """
        img = self.info_df['Source Image'].values[self.info_df['Syn RID 1'] == '1-1E98'][0]
        return img

    def generate_path(self, rid, path_type):
        """
        Generates path according to the HARDCODED template

        Parameters:
            rid (string) : the RID you want to create synapse csv, npz,
                        nuclear of image path.
            path_type (string) : what kind of path : 'rid' / 'npz' / 'nuc' / 'img' .

        Returns:
            string : path you requested
        """
        # template paths
        # FIXME : this won't work if the folder structure/naming convention changes
        rid_path = Path('raw/csv/pallium/syn')
        img_path = Path('raw/img/pallium')
        nuc_path = Path('raw/csv/pallium/nuc')
        npz_path = Path('raw/npz/pallium/syn')

        if path_type == 'rid':
            file_name = self.img[rid] + "_" + rid + "_synapses_only.csv"
            file_path = self.data_folder / rid_path / file_name
        elif path_type == 'img':
            file_name = "Image_" + self.img[rid] + ".ome.tif"
            file_path = self.data_folder / img_path / file_name
        elif path_type == 'nuc':
            file_name = self.img[rid] + "_nuclei_only.csv"
            file_path = self.data_folder / nuc_path / file_name
        elif path_type == 'npz':
            file_name = self.img[rid] + "_" + rid + ".npz"
            file_path = self.data_folder / npz_path / file_name
        else:
            ValueError('who should be "rid" "img" or "nuc" ')

        return file_path

    def get_volumes_and_labels(self, padding, recrop=False):
        """
        Crops volumed according to padding around the center.

        Parameters:
        padding (list) : half radius of area around the center of synapse to consider.
                In ZYX order, so that [0,7,7] will result in input_shape=(15,15,1)
        """
        shape = 2 * np.array(padding) + 1

        # either recrop is True of self.volumes is empty:
        # crop volumes for all the entries (empty dictionaries are false)
        if recrop or (not self.volumes):
            self.shape = shape
            for rid in self.rid:
                self.volumes[rid], self.labels[rid] = self.crop_and_label(rid)

        # crop volumes only for the new entries
        else:
            assert self.shape == shape, "Padding has changed, do you want to recrop all the volumes?"

            for rid in self.rid:
                if rid in self.volumes:
                    pass
                else:
                    self.volumes[rid], self.labels[rid] = self.crop_and_label(rid)

    def crop_and_label(self, rid):
        """
        For all centroids crop the box around
        """
        centroids_tiff, labels = get_centroids_and_labels(self.path_rid[rid], self.path_npz[rid])
        num_centroids = centroids_tiff.shape[0]

        img_volume = tif.imread(self.path_img[rid])

        cropped_volumes = np.zeros((num_centroids, self.shape[0], self.shape[1], self.shape[2]))

        for count, centroid in enumerate(centroids_tiff):
            cropped_volumes[count, :, :, :] = get_one_volume(img_volume, centroid, self.shape)

        return cropped_volumes, labels


