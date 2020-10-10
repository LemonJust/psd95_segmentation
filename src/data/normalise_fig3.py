"""
This code normalises the data for fig3.
1. Using the Robust Scaler
"""
# %% Load data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

project_folder = 'D:/Code/repos/synapse-redistribution/'
data = pd.read_csv(f'{project_folder}psd95_verify/fig3_main_extended.csv')
# %% split into separate df, because of different number of rows there are nan otherwise
main = data[['main_x', 'main_y']].dropna()
main.head()
exta = data[['exta_x', 'exta_y']].dropna()
exta.head()
extb = data[['extb_x', 'extb_y']].dropna()
extb.head()
# %% 1. Robust Scaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
main_scaled = scaler.fit_transform(main)
exta_scaled = scaler.fit_transform(exta)
extb_scaled = scaler.fit_transform(extb)

print('main_scaled means (Spine Int, PSD95 Int): ', main_scaled.mean(axis=0))
print('exta_scaled means (Spine Int, PSD95 Int): ', exta_scaled.mean(axis=0))
print('extb_scaled means (Spine Int, PSD95 Int): ', extb_scaled.mean(axis=0))

print('main_scaled std (Spine Int, PSD95 Int): ', main_scaled.std(axis=0))
print('exta_scaled std (Spine Int, PSD95 Int): ', exta_scaled.std(axis=0))
print('extb_scaled std (Spine Int, PSD95 Int): ', extb_scaled.std(axis=0))

print('main_scaled Min (Spine Int, PSD95 Int): ', main_scaled.min(axis=0))
print('exta_scaled Min (Spine Int, PSD95 Int): ', exta_scaled.min(axis=0))
print('extb_scaled Min (Spine Int, PSD95 Int): ', extb_scaled.min(axis=0))

print('main_scaled Max (Spine Int, PSD95 Int): ', main_scaled.max(axis=0))
print('exta_scaled Max (Spine Int, PSD95 Int): ', exta_scaled.max(axis=0))
print('extb_scaled Max (Spine Int, PSD95 Int): ', extb_scaled.max(axis=0))
# %%
data_scaled = pd.concat([pd.DataFrame(main_scaled, columns=['main_x', 'main_y']),
                         pd.DataFrame(exta_scaled, columns=['exta_x', 'exta_y']),
                         pd.DataFrame(extb_scaled, columns=['extb_x', 'extb_y'])], axis=1)

data_scaled.to_csv(f'{project_folder}psd95_verify/fig3_main_extended_scale25to75.csv')
