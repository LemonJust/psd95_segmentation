"""
get synapse features into a table.
"""

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

#
# %%
csv_filename = 'D:/Code/repos/synapse-redistribution/get_data/data/1-1EWW_1' \
               '-0YEY_SynapsePositions.csv'
cohort_df = pd.read_csv(csv_filename)
# get study ids
study_id = cohort_df.drop_duplicates(subset='study_id', inplace=False)[
               'study_id'].values[:]


# %% load all the synapses in 
def create_syn_type(syn, df, syn_type, xyz_cols, int_core_col, int_vcn_col):
    syn[fish_id][syn_type] = {}
    syn[fish_id][syn_type]["xyz"] = df.loc[:, xyz_cols].values
    syn[fish_id][syn_type]["int_core"] = df.loc[:, int_core_col].values
    syn[fish_id][syn_type]["int_vcn"] = df.loc[:, int_vcn_col].values
    return syn

syn = {}
fish_id = study_id[0]
# get all the data for this fish
is_unch = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 0.0)]
is_lost = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 1.0)]
is_gain = cohort_df.loc[(cohort_df['study_id'] == fish_id) & (cohort_df['t'] == 2.0)]

syn[fish_id] = {}
syn = create_syn_type(syn, is_lost, "lost", ["x1", "y1", "z1"], ["core1"], ["vcn1"])
syn = create_syn_type(syn, is_gain, "gain", ["x2", "y2", "z2"], ["core2"], ["vcn2"])
syn = create_syn_type(syn, is_unch, "uncB", ["x1", "y1", "z1"], ["core1"], ["vcn1"])
syn = create_syn_type(syn, is_unch, "uncA", ["x2", "y2", "z2"], ["core2"], ["vcn2"])

