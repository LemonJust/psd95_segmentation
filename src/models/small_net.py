from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
import tifffile as tif

from src.features.synapse_features import get_centroids_and_labels, get_one_volume


class ShapeNet:
    """
    This is a net used to extract shape features.
    It looks only onto a normalised intensity images, small area around the potential synapse.
    """
    # Class Attribute
    species = 'mammal'

    # Initializer / Instance Attributes
    def __init__(self, seed=None):
        self.seed = seed

        self.model = []
        self.initialise_model()

    def initialise_model(self):
        """
        Set up the network structure.
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(65, 65, 1), name='conv_1'))
        model.add(layers.Conv2D(16, kernel_size=3, activation='relu', name='conv_2'))
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu', name='conv_3'))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu', name='conv_4'))
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu', name='conv_5'))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu', name='conv_6'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        self.model = model

