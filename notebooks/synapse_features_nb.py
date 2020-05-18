"""
get synapse features into a table.
"""

#%%
# Prepare the table with all the available data ( if you haven't done so already)
info_df = load()

#%%
# prepare training subset
train_list_df = make_dataset(info_csv,
                             'ImgZfDsy20171201A3A',
                             'ImgZfDsy20170915D3A',
                             'ImgZfZdu20171201B3C',
                             'ImgZfZdu20171020A3A',
                             'ImgZfDsy20171207B3A',
                             'ImgZfZdu20171121A3C')
# save training subset for future reference

#%%
# extract synapse intensity in green

# extract synapse intensity in red

#
