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


def load_data(out_dir, layers, features):
    # Load the results in their own dictionaries and create a dataframe
    data_list = []
    files = glob.glob(f'{out_dir}/FeaturePermutation/*pkl')
    for f in files:
        data_list.append(load_pkl(f))
    df = pd.DataFrame(data_list)
    df['significant'] = 'ns'
    df.loc[df.p < 0.05, 'significant'] = '*'
    df['layer'] = pd.Categorical(df['layer'], ordered=True,
                                 categories=layers)
    df['features'] = pd.Categorical(df['feature'], ordered=True,
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
         5: 0.2}
    return d[key]


out_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim'
figure_dir = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures'
process = 'PlotFeatureAlexNet'
Path(f'{figure_dir}/{process}').mkdir(exist_ok=True, parents=True)
Path(f'{out_dir}/{process}').mkdir(exist_ok=True, parents=True)
layers = [1, 2, 3, 4, 5]
features = ['indoor', 'expanse', 'transitivity', 'agent_distance', 'facingness', 'joint_action', 'communication',
            'valence', 'arousal']

df = load_data(out_dir, layers, features)
df.to_csv(f'{out_dir}/{process}/feature_prediction.csv', index=False)
y_max = df.high_ci.max()
_, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x='feature', y='r2', hue='layer', ax=ax, data=df)
for bar, (layer, feature) in zip(ax.patches, itertools.product(layers, features)):
    color = feature2color(feature)
    color[:-1] = color[:-1] * layer2shade(layer)
    y1 = df.loc[(df.feature == feature) & (df.layer == layer), 'low_ci'].item()
    y2 = df.loc[(df.feature == feature) & (df.layer == layer), 'high_ci'].item()
    sig = df.loc[(df.feature == feature) & (df.layer == layer), 'significant'].item()
    x = bar.get_x() + 0.1
    ax.plot([x, x], [y1, y2], 'k')
    if sig != 'ns':
        ax.scatter(x, y_max - 0.02, marker='o', color=color)
    bar.set_color(color)
ax.legend([], [], frameon=False)
ax.set_xticklabels(features,
                   fontsize=20,
                   rotation=45, ha='right')
plt.savefig(f'{figure_dir}/{process}/feature_prediction.png')
