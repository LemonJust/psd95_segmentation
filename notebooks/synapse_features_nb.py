"""
get synapse features into a table.
"""

from src.features.build_features import get_percentiles
from src.features.synapse_features import load_imgpairstudy_csv, get_cohort
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def plot_histo(stu, how_smart, fish_id):
    ax1 = plt.subplot(211)
    ax1.set_title(f'Hollow Intensity in {how_smart} {fish_id}')
    what_measure = 'int_vcn'

    lost = stu[how_smart][fish_id]['lost'][what_measure]
    gain = stu[how_smart][fish_id]['gain'][what_measure]
    uncB = stu[how_smart][fish_id]['uncB'][what_measure]
    uncA = stu[how_smart][fish_id]['uncA'][what_measure]
    ax1.hist([lost, gain, uncB, uncA], color=['b', 'r', 'g', 'y'], alpha=0.5,
             density=True)
    ax1.legend(['lost', 'gain', 'uncB', 'uncA'])
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    plt.xlabel('Intensity, a.u.')
    plt.ylabel('Syn. Fraction')

    ax2 = plt.subplot(223)
    lost = stu[how_smart][fish_id]['lost'][what_measure]
    uncB = stu[how_smart][fish_id]['uncB'][what_measure]
    ax2.hist([lost, uncB], color=['b', 'g'], alpha=0.5, density=True)
    ax2.legend(['lost', 'uncB'])
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel('Intensity, a.u.')
    plt.ylabel('Syn. Fraction')

    ax3 = plt.subplot(224)
    gain = stu[how_smart][fish_id]['gain'][what_measure]
    uncA = stu[how_smart][fish_id]['uncA'][what_measure]
    ax3.hist([gain, uncA], color=['r', 'y'], alpha=0.5, density=True)
    ax3.legend(['gain', 'uncA'])
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel('Intensity, a.u.')
    # plt.ylabel('Syn. Fraction')
    plt.show()


def get_prct(stu, how_smart, fish_id):
    what_measure = 'int_core'
    lost = stu[how_smart][fish_id]['lost'][what_measure]
    gain = stu[how_smart][fish_id]['gain'][what_measure]
    uncB = stu[how_smart][fish_id]['uncB'][what_measure]
    uncA = stu[how_smart][fish_id]['uncA'][what_measure]
    lost_prc = get_percentiles(lost)
    gain_prc = get_percentiles(gain)
    uncB_prc = get_percentiles(uncB)
    uncA_prc = get_percentiles(uncA)

    prc_list = np.arange(0, 110, 10)
    col_names = [f'prc_{p}' for p in prc_list]

    lost_df = pd.DataFrame(lost_prc[np.newaxis, :], columns=col_names)
    lost_df['fish_id'] = fish_id
    gain_df = pd.DataFrame(gain_prc[np.newaxis, :], columns=col_names)
    gain_df['fish_id'] = fish_id
    uncB_df = pd.DataFrame(uncB_prc[np.newaxis, :], columns=col_names)
    uncB_df['fish_id'] = fish_id
    uncA_df = pd.DataFrame(uncA_prc[np.newaxis, :], columns=col_names)
    uncA_df['fish_id'] = fish_id

    return lost_df, gain_df, uncB_df, uncA_df

# %% WHEN IMPORTED
if __name__ == "synapse_features_nb":
    print('synapse_features_nb imported')

# %% ALL THE CODE TO RUN
if __name__ == "__main__":
    # %%
    # Prepare the table with all the available data ( if you haven't done so already)
    info_df = load()

    # %%
    # prepare training subset
    train_list_df = make_dataset(info_csv,
                                 'ImgZfDsy20171201A3A',
                                 'ImgZfDsy20170915D3A',
                                 'ImgZfZdu20171201B3C',
                                 'ImgZfZdu20171020A3A',
                                 'ImgZfDsy20171207B3A',
                                 'ImgZfZdu20171121A3C')

    # save training subset for future reference
    # %%
    # extract synapse intensity in green
    # extract synapse intensity in red

    # %%
    csv_filename = 'D:/Code/repos/synapse-redistribution/get_data/data/1-1EWW_1' \
                   '-0YEY_SynapsePositions.csv '
    cohort_key = 'D:/Code/repos/synapse-redistribution/get_data/data/1-1EWW_CohortKey.csv'

    cohort_df, study_id = load_imgpairstudy_csv(csv_filename)

    # %% load all the synapses in
    img_resolution = [0.26, 0.26, 0.4]
    stu = get_cohort(csv_filename, cohort_key, img_resolution)
    pickle.dump(stu, open("data/interim/stu_cohort_EWW.p", "wb"))
    # %%
    lost_df = get_prct(stu, 'LR', '1-0CN6')
    # %%
    plot_histo(stu, 'LR', '1-0CN6')
    # %%
    from matplotlib.backends.backend_pdf import PdfPages

    pdffilepath = 'D:/Code/repos/psd95_segmentation/data/processed' \
                  '/hollow_intensity_lost_gained_vs_unchanged.pdf'
    pdf = PdfPages(pdffilepath)

    for how_smart in ['LR', 'NL', 'NS', 'US', 'CS']:
        for fish_id in stu[how_smart].keys():
            fig = plt.figure()
            plot_histo(stu, how_smart, fish_id)
            pdf.savefig(fig)

    # remember to close the object to ensure writing multiple plots
    pdf.close()

    # %%
    how_smart = 'LR'
    for i_fish, fish_id in enumerate(stu[how_smart].keys()):
        print(i_fish)
        if i_fish == 0:
            lost_df, gain_df, uncB_df, uncA_df = get_prct(stu, how_smart, fish_id)
        else:
            lost_df2, gain_df2, uncB_df2, uncA_df2 = get_prct(stu, how_smart, fish_id)
            lost_df = lost_df.append(lost_df2, ignore_index=True)
            gain_df = gain_df.append(gain_df2, ignore_index=True)
            uncB_df = uncB_df.append(uncB_df2, ignore_index=True)
            uncA_df = uncA_df.append(uncA_df2, ignore_index=True)

    path_to_save = 'C:/Users/nadtochi/Dropbox/2019_LearningPaper/Cohort_data' \
                   '/Intensity_percentiles/'
    lost_df = lost_df.set_index('fish_id')
    lost_df.to_csv(f'{path_to_save}{how_smart}_lost_prc.csv')
    gain_df = gain_df.set_index('fish_id')
    gain_df.to_csv(f'{path_to_save}{how_smart}_gain_prc.csv')
    uncB_df = uncB_df.set_index('fish_id')
    uncB_df.to_csv(f'{path_to_save}{how_smart}_unchBefore_prc.csv')
    uncA_df = uncA_df.set_index('fish_id')
    uncA_df.to_csv(f'{path_to_save}{how_smart}_unchAfter_prc.csv')

 # %% Karls code to get Kernels

import synspy.analyze.util

# these are params as used in our pipeline
gridsize = (0.4, 0.26, 0.26)

synaptic_footprints = (
    (2.75, 1.5, 1.5),
    (4.0, 2.75, 2.75),
    (3.0, 3.0, 3.0),
)
synapse_diam_microns, vicinity_diam_microns, redblur_microns = synaptic_footprints
kernels3d = synspy.analyze.util.prepare_kernels(gridsize, synapse_diam_microns, vicinity_diam_microns,
                                    redblur_microns)[1]
# these should be 3D numpy arrays
core, hollow, red, maxbox = kernels3d

# %%
core[3]
