from pathlib import Path
import numpy as np
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import requests
import plotly.graph_objs as go


def plot_histogram(data, xlim, ylim):
    """
    The function to plot a histogram.

    Parameters:
        data (numpy array): distribution to plot.
        xlim (array (2,1) float): lower and upper x limit to show.
        ylim (array (2,1) float): lower and upper y limit to show.
    """
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data.flatten(), bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=50, histtype='step')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity, a.u.')
    plt.ylabel('N pixels')
    plt.title('My Very Own Histogram')

    # Set a clean upper y-axis limit.
    plt.ylim((ylim[0], ylim[1]))
    plt.xlim((xlim[0], xlim[1]))
    plt.show()

def plotly_histo(x0,x1,binwidth,x0_label,x1_label,title):

    min_hist = min(min(x0), min(x1))
    max_hist = max(max(x0), max(x1))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0,
                               xbins=dict(start=min_hist, size=binwidth, end=max_hist),
                               name=x0_label,
                               histnorm='percent', ))

    fig.add_trace(go.Histogram(x=x1,
                               xbins=dict(start=min_hist, size=binwidth, end=max_hist),
                               name=x1_label,
                               histnorm='percent', ))

    # The two histograms are drawn on top of another
    fig.update_layout(
        barmode='overlay',
        title_text=title,  # title of plot
        xaxis_title_text='Intensity, a.u.',  # xaxis label
        yaxis_title_text='%',  # yaxis label
    )
    fig.update_traces(opacity=0.5)
    fig.show()


def plot_boxplot(df, x_axis, y_axis):
    """
    Shows box plots of data in the data frame, grouping y_axis according x_axis label.

    Parameters:
        df (pandas.DataFrame): data frame with the data
        x_axis (string): the name of the df column to use for grouping.
        y_axis (string): the name of the df column to group according to x_axis.

    Returns:
        matplotlib.figure.Figure': figure
    """
    ax = sns.boxplot(x=x_axis, y=y_axis, data=df)

    # Calculate number of obs per group & median to position labels
    medians = df.groupby([x_axis])[y_axis].median().values
    nobs = df[x_axis].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], medians[tick] + 0.03, nobs[tick],
                horizontalalignment='center', size='x-small', color='w', weight='semibold')
    fig = plt.gcf()
    plt.show()
    return fig
