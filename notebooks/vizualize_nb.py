#%% imports

import pandas as pd
from src.visualization.visualize import plot_boxplot

#%% md

### Load data frame to visualize

#%%

img_intensity_df = pd.read_csv('D:/Code/repos/psd95_segmentation/data/processed/intensity_over_years.csv')
img_intensity_df

#%% md

### Look at some boxplots

#%%

plot_feature = 'green_prc_100'
fig = plot_boxplot(img_intensity_df, 'Year', plot_feature)
print (type(fig))

#%% md

Save figure

#%%

fig.savefig(f'{plot_feature}.png')
