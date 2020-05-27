import src
from src.models.cnn_train_data import CnnTrainingData
import numpy as np

# %%
folder = 'D:\\Code\\repos\\psd95_segmentation\\data'
info = folder + '\\raw\\rid-img_pairs.csv'
# %%
# the good data sets :
# '1-0CBE', '1-0EVJ', '1-0EWE', '1-0J1J', '1-0JNA', '1-0QTY', '1-0YR8', '1-0YRW'
train_data = CnnTrainingData(data_folder=folder, info_csv=info)
train_data.append_rids('1-1E98', '1-1DCG')

# %%
