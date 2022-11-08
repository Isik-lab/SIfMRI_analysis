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
from statsmodels.stats.multitest import multipletests
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
    df['layer'] = pd.Categorical(df['layer'], ordered=True,
                                 categories=layers)
    df['sid'] = pd.Categorical(df['sid'], ordered=True,
                               categories=subjs)
    df['roi'] = pd.Categorical(df['roi'], ordered=True,
                               categories=rois)
    return df


out_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
process = 'PlotROIPredictionAlexNet'
Path(f'{figure_dir}/{process}').mkdir(exist_ok=True, parents=True)
layers = [1, 2, 3, 4, 5]
subjs = ['01', '02', '03', '04']
rois = ['EVC',  'MT', 'FFA', 'PPA', 'EBA', 'LOC', 'aSTS', 'face-pSTS', 'pSTS', 'TPJ']


df = load_data(out_dir, layers, subjs, rois)
y_max = df.high_ci.max()
for roi in rois:
    _, ax = plt.subplots()
    sns.barplot(x='layer', y='r2', hue='sid',
                ax=ax, data=df.loc[df.roi == roi])
    for bar, (subj, layer) in zip(ax.patches, itertools.product(subjs, layers)):
        # color = cat2color(layer)
        # color[:-1] = color[:-1] * subj2shade(subj)
        y1 = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'low_ci'].item()
        y2 = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'high_ci'].item()
        sig = df.loc[(df.roi == roi) & (df.sid == subj) & (df.layer == layer), 'significant'].item()
        x = bar.get_x() + 0.1
        ax.plot([x, x], [y1, y2], 'k')
        if sig != 'ns':
            ax.scatter(x, y_max + 0.02, marker='o', color='k')
    plt.title(roi)
    plt.savefig(f'{figure_dir}/{process}/{roi}.png')
