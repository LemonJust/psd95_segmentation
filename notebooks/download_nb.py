# %%

"""
This notebook helps you download the csv for the image files you have.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.read_data import get_image_name, download_nuclear_csvs, \
    download_synapse_csvs, \
    download_npz, list_available_segmentations

# %%
"""
1. Provide path with all the images

Before running it, I assume that you have already downloaded all the images of interest.
"""
pallium_path = Path("D:/Code/repos/psd95_segmentation/data/raw/img/pallium")

# get list of tif files in the folder
files = [x for x in pallium_path.glob('*tif') if x.is_file()]
print(files)

# %%
"""
2. Load a csv that has all the info for the images, segmentation , etc
"""

info_df = pd.read_csv('D:/Code/repos/psd95_segmentation/data/raw/karls_good2.csv')

info_year = info_df.loc[:, ['Source Image', 'Subject Issue Date']].sort_values(
    by=['Source Image'])
info_year.drop_duplicates(subset='Source Image', inplace=True)

# %%
"""
3. Get image file names and years
"""

names = []
for count, file_name in enumerate(files):
    img_name = get_image_name(file_name)

    # if you can't find a year - probably csv is off
    if info_year.loc[info_year['Source Image'] == img_name,
                     'Subject Issue Date'].empty:
        raise ValueError('Corrupted entry in the info_year.'
                         ' Check provided csv for errors. '
                         'Example: 1-1988 was changed to Jan-88.')
    else:
        year = info_year.loc[info_year['Source Image'] == img_name,
                             'Subject Issue Date'].values[0].split(sep='-')[0]

    print(f"#{count} image {img_name} year {year}")
    names.append(img_name)

# %%
"""
OPT 1. Download nuclear csv
"""
nuc_path = 'D:/Code/repos/psd95_segmentation/data/raw/csv/pallium/nuc/'
download_nuclear_csvs(info_df, names, nuc_path)

# %%
"""
OPT 2. Download synapse csv

Some files are of size 1Kb - you will probably want to get rid of those.
Also, some files are exact duplicates... 
"""
syn_path = 'D:/Code/repos/psd95_segmentation/data/raw/csv/pallium/syn/'
download_synapse_csvs(info_df, names, syn_path)

# %%
"""
OPT 3. Download npz files

before you do - figure out what RIDs you actually want.
"""
syn_path = 'D:/Code/repos/psd95_segmentation/data/raw/csv/pallium/syn/'
img_rid_df = list_available_segmentations(syn_path)
img_rid_df.to_csv('D:/Code/repos/psd95_segmentation/data/raw/rid-img_pairs.csv')

rids = np.squeeze(img_rid_df.loc[:, ['Syn RID 1']].values)

# %%
npz_path = 'D:/Code/repos/psd95_segmentation/data/raw/npz/pallium/syn/'
download_npz(info_df, rids, npz_path)
