#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests, fdrcorrection
import itertools
import matplotlib.ticker as mticker


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    d.pop('r2var', None)
    d.pop('r2null', None)
    return d


def load_data(out_dir, layers, subjs, rois):
    # Load the results in their own dictionaries and create a dataframe
    data_list = []
    files = glob.glob(f'{out_dir}/ROIPredictionAlexNet/*pkl')
    for f in files:
        data_list.append(load_pkl(f))
    df = pd.DataFrame(data_list)
    df['significant'] = 'ns'
    df.loc[df.p < 0.05, 'significant'] = '*'
    df = df[df.roi != 'TPJ']
    df['layer'] = pd.Categorical(df['layer'], ordered=True,
                                 categories=layers)
    df['sid'] = pd.Categorical(df['sid'], ordered=True,
                               categories=subjs)
    df['roi'] = pd.Categorical(df['roi'], ordered=True,
                               categories=rois)
    return df


def subj2shade(key):
    d = {'01': 1.0,
         '02': 0.8,
         '03': 0.6,
         '04': 0.4}
    return d[key]


out_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
process = 'PlotROIPredictionAlexNet'
Path(f'{figure_dir}/{process}').mkdir(exist_ok=True, parents=True)
Path(f'{out_dir}/{process}').mkdir(exist_ok=True, parents=True)
layers = [1, 2, 3, 4, 5]
subjs = ['01', '02', '03', '04']
rois = ['EVC',  'MT', 'FFA', 'PPA', 'EBA', 'LOC', 'pSTS', 'face-pSTS', 'aSTS']


df = load_data(out_dir, layers, subjs, rois)
df['p_corrected'] = 1
for roi, subj in itertools.product(rois, subjs):
    _, p_corrected = fdrcorrection(df.loc[(df.roi == roi) & (df.sid == subj), 'p'])
    df.loc[(df.roi == roi) & (df.sid == subj), 'p_corrected'] = p_corrected
df['significant'] = 'ns'
df.loc[df.p_corrected < 0.05, 'significant'] = '*'
df.to_csv(f'{out_dir}/{process}/roi_prediction.csv', index=False)
y_max = df.high_ci.max() + 0.04
_, axes = plt.subplots(1, len(rois), figsize=(int(len(rois)*6), 8))
for ax, roi in zip(axes, rois):
    sns.barplot(x='layer', y='r2', hue='sid',
                ax=ax, data=df[df.roi == roi], color=[0.8, 0, 0.8])
    for bar, (subj, layer) in zip(ax.patches, itertools.product(subjs, layers)):
        color = np.array(bar.get_facecolor())
        color[:-1] = color[:-1] * subj2shade(subj)
        y1 = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'low_ci'].item()
        y2 = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'high_ci'].item()
        sig = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'significant'].item()
        x = bar.get_x() + 0.1
        ax.plot([x, x], [y1, y2], 'k')
        if sig != 'ns':
            ax.scatter(x, y_max - 0.02, marker='o', color=color, edgecolors=[0.2, 0.2, 0.2])
        bar.set_color(color)
        bar.set_edgecolor([0.2, 0.2, 0.2])

    ax.set_xlabel('AlexNet Layer', fontsize=22)
    ax.legend([], [], frameon=False)
    ax.set_ylim([0, y_max])
    ax.set_title(roi, fontsize=30)
    ax.set_ylabel(f'Explained variance ($r^2$)', fontsize=22)

    # Plot vertical lines to separate the bars
    ax.vlines(np.arange(0.5, len(layers) - 0.5),
              ymin=0, ymax=y_max - (y_max / 20),
              colors='lightgray')

    # Change the ytick font size
    label_format = '{:,.2f}'
    y_ticklocs = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
    ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=20)

    # Remove lines on the top and left of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Change the xaxis font size and colors
    ax.set_xticklabels(layers,
                       fontsize=20, ha='right')

plt.tight_layout()
plt.savefig(f'{figure_dir}/{process}/feature_prediction.png')
