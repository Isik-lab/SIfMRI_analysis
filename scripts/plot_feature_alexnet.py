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
from statsmodels.stats.multitest import fdrcorrection


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    d.pop('r2var', None)
    d.pop('r2null', None)
    return d


def load_data(out_dir, layers, features):
    # Load the results in their own dictionaries and create a dataframe
    data_list = []
    files = glob.glob(f'{out_dir}/FeaturePermutation/*pkl')
    for f in files:
        data_list.append(load_pkl(f))
    df = pd.DataFrame(data_list)
    df['layer'] = pd.Categorical(df['layer'], ordered=True,
                                 categories=layers)
    df['feature'] = pd.Categorical(df['feature'], ordered=True,
                                    categories=features)
    return df


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['transitivity'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['agent_distance'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['facingness'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['joint_action'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['communication'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['valence'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    d['arousal'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    if key is not None:
        return d[key]
    else:
        return d


def layer2shade(key):
    d = {1: 1.0,
         2: 0.8,
         3: 0.6,
         4: 0.4,
         5: 0.2,
         'moten': 0}
    return d[key]


out_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
process = 'PlotFeatureAlexNet'
Path(f'{figure_dir}/{process}').mkdir(exist_ok=True, parents=True)
Path(f'{out_dir}/{process}').mkdir(exist_ok=True, parents=True)
layers = [1, 2, 3, 4, 5, 'moten']
features = ['indoor', 'expanse', 'transitivity',
            'agent_distance', 'facingness',
            'joint_action', 'communication',
            'valence', 'arousal']

df = load_data(out_dir, layers, features)
df['p_corrected'] = 1
for feature in features:
    _, p_corrected = fdrcorrection(df.loc[df.feature == feature, 'p'])
    df.loc[df.feature == feature, 'p_corrected'] = p_corrected
df['significant'] = 'ns'
df.loc[df.p_corrected < 0.05, 'significant'] = '*'
df.to_csv(f'{out_dir}/{process}/feature_prediction.csv', index=False)
_, axes = plt.subplots(1, len(features), figsize=(int(len(features)*6), 8))
y_max = df.high_ci.max() + 0.04
for ax, feature in zip(axes, features):
    sns.barplot(x='layer', y='r2', ax=ax, data=df[df.feature == feature])
    for bar, layer in zip(ax.patches, layers):
        color = feature2color(feature)
        y1 = df.loc[(df.feature == feature) & (df.layer == layer), 'low_ci'].item()
        y2 = df.loc[(df.feature == feature) & (df.layer == layer), 'high_ci'].item()
        sig = df.loc[(df.feature == feature) & (df.layer == layer), 'significant'].item()
        x = bar.get_x() + 0.45
        ax.plot([x, x], [y1, y2], 'k', linewidth=3)
        if sig != 'ns':
            ax.scatter(x, y_max - 0.02, marker='o', color='k')
        bar.set_color(color)
        bar.set_edgecolor([0.2, 0.2, 0.2])

    ax.set_xlabel('')
    ax.legend([], [], frameon=False)
    ax.set_ylim([0, y_max])
    ax.set_title(feature.replace('_', ' '), fontsize=26)
    ax.set_ylabel(f'Explained variance ($r^2$)', fontsize=22)

    # Change the ytick font size
    label_format = '{:,.2f}'
    y_ticklocs = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
    ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=20)

    # Remove lines on the top and left of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Change the xaxis font size and colors
    ax.set_xticklabels(['conv1', 'conv2', 'conv3',
                        'conv4', 'conv5', 'moten'],
                       fontsize=20, rotation=45,
                       ha='right')

plt.tight_layout()
plt.savefig(f'{figure_dir}/{process}/feature_prediction.png')
