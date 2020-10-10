"""
This is a file where I'm trying to estimate intensity SNR .
"""

#%% Let's have a look at Karl's kernel
import synspy.analyze.util

# these are params as used in our pipeline
gridsize = (0.4, 0.26, 0.26)

synaptic_footprints = (
    (2.75, 1.5, 1.5),
    (4.0, 2.75, 2.75),
    (3.0, 3.0, 3.0),
)
synapse_diam_microns, vicinity_diam_microns, redblur_microns = synaptic_footprints
kernels3d = synspy.analyze.util.prepare_kernels(gridsize, synapse_diam_microns,
                                                vicinity_diam_microns, redblur_microns)[1]
# these should be 3D numpy arrays
core, hollow, red, maxbox = kernels3d

#%% Plot core and hollow slice by slice
import plotly.graph_objects as go

# will open plots in browser, but it's fine
import plotly.io as pio
pio.renderers.default = "browser"

fig = go.Figure()

config = dict({'scrollZoom': True})

fig.add_trace(
    go.Scatter(
        x=[1, 2, 3],
        y=[1, 3, 1]))

fig.show(config=config)

#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.show()

