# %% md

### Look at the intensities

# %%

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.read_data import get_image_name, download_nuclear_csvs, download_synapse_csvs
from src.features.build_features import get_img_percentiles


# %% path with all the images

pallium_path = Path("D:/Code/repos/psd95_segmentation/data/raw/img/pallium")

# get list of tif files in the folder
files = [x for x in pallium_path.glob('*tif') if x.is_file()]
print(files)

# %% md

### Load a csv that has all the info for the images, segmentation , etc

# %%

info_df = pd.read_csv('D:/Code/repos/psd95_segmentation/data/raw/karls_good2.csv')

info_year = info_df.loc[:, ['Source Image', 'Subject Issue Date']].sort_values(by=['Source Image'])
info_year.drop_duplicates(subset='Source Image', inplace=True)
info_year

# %% md

### Match the files you have with the years, extract file names

# %%

years = np.zeros((len(files), 1))
names = []
for count, file_name in enumerate(files):
    img_name = get_image_name(file_name)
    year = info_year.loc[info_year['Source Image'] == img_name,
                         'Subject Issue Date'].values[0].split(sep='-')[0]
    print(f"#{count} image {img_name} year {year}")
    years[count] = year
    names.append(img_name)

# %% md

### Get the intensity precentiles for green and red channels.

# %%

padding = [[0, 0], [50, 500], [20, 20]]
prc_green = get_img_percentiles("green", files, padding)
prc_red = get_img_percentiles("red", files, padding)

# %% md

Combine
names, year and prcentiles
into
a
data
frame

# %% merge year w percentile

img_intensity_df = pd.DataFrame(np.concatenate((years, prc_green, prc_red), axis=1),
                                columns=['Year', 'green_prc_0', 'green_prc_10',
                                         'green_prc_20', 'green_prc_30',
                                         'green_prc_40', 'green_prc_50',
                                         'green_prc_60', 'green_prc_70',
                                         'green_prc_80', 'green_prc_90',
                                         'green_prc_100',
                                         'red_prc_0', 'red_prc_10',
                                         'red_prc_20', 'red_prc_30',
                                         'red_prc_40', 'red_prc_50',
                                         'red_prc_60', 'red_prc_70',
                                         'red_prc_80', 'red_prc_90',
                                         'red_prc_100'])
img_intensity_df.insert(0, 'Name', names, True)

# %% md

### Figure out what time point it is.
If
the
old
image
name
ends
with a 6 or 5 then its tp 2; 3 or 2 is tp 1.

# %%

id_segments = []
time_point = []
for count, img_name in enumerate(names):

    id_segment = info_df.loc[info_df['Source Image'] == img_name, 'ID'].values[0]
    print(f"#{count} image {img_name} ID {id_segment}")
    id_segments.append(id_segment)

    if id_segment[-2] == '6' or id_segment[-2] == '5':
        time_point.append(2)
    elif id_segment[-2] == '3' or id_segment[-2] == '2':
        time_point.append(1)
    else:
        raise ValueError("Don't know what time point it is")

# add to the data frame
img_intensity_df.insert(1, 'tp', time_point, True)

# %% md

### Calculate and add nuclear percentiles to the data frame

# %%

nuc_prc = np.zeros((len(names), 5))

# for each image , read the corresponding csv and calculate pct
for count, img_name in enumerate(names):
    print(f"#{count} image {img_name}")

    csv_file = nuc_path + img_name + '_nuclei_only.csv'
    nuc_csv_df = pd.read_csv(csv_file, skiprows=[1])

    nuc_intensity = nuc_csv_df.loc[:, ['raw core']]
    nuc_prc[count, :] = np.percentile(nuc_intensity, [10, 25, 50, 75, 90])

nuc_prc_df = pd.DataFrame(nuc_prc, columns=['nuc_prc_10', 'nuc_prc_25',
                                            'nuc_prc_50', 'nuc_prc_75', 'nuc_prc_90'])
# add to the main df
img_intensity_df = pd.concat([img_intensity_df, nuc_prc_df], axis=1)

# %% md

### Calculate and add synapse percentiles to the data frame

# %%

syn_prc = np.zeros((len(names), 5))

for count, img_name in enumerate(names):
    done = 0
    segmentations = info_df.loc[info_df['Source Image'] == img_name,
                                ['RID', 'classifier_name', 'Segments Filtered URL']]
    for index, segmentation in segmentations.iterrows():
        rid = segmentation[0]
        cl_name = segmentation[1]
        url = segmentation[2]
        csv_type = url.split('.')[1]
        if csv_type == 'synapses_only':
            csv_file = syn_path + img_name + '_' + rid + '_' + csv_type + '.csv'

            # grab the first segmentation you have for this image
            if Path(csv_file).exists() and done == 0:
                done = 1

                syn_csv_df = pd.read_csv(csv_file, skiprows=[1])
                syn_intensity = syn_csv_df.loc[:, ['raw core']]
                syn_prc[count, :] = np.percentile(syn_intensity,
                                                  [10, 25, 50, 75, 90])

syn_prc_df = pd.DataFrame(syn_prc, columns=['syn_prc_10', 'syn_prc_25',
                                            'syn_prc_50', 'syn_prc_75', 'syn_prc_90'])

img_intensity_df = pd.concat([img_intensity_df, syn_prc_df], axis=1)

# %% md

### Save data frame

# %%


