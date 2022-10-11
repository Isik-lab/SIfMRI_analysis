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


def multiple_comp_correct(arr):
    out = multipletests(arr, alpha=0.05, method='fdr_bh')[1]
    return out


def model2cat():
    d = dict()
    d['indoor'] = 'visual'
    d['expanse'] = 'visual'
    d['object'] = 'visual'
    d['agent distance'] = 'primitive'
    d['facingness'] = 'primitive'
    d['joint action'] = 'social'
    d['communication'] = 'social'
    d['valence'] = 'affective'
    d['arousal'] = 'affective'
    return d


def subj2shade(key):
    d = {'01': 1.0,
         '02': 0.8,
         '03': 0.6,
         '04': 0.4}
    return d[key]


def roi2color(key=None):
    d = {'EVC': np.array([1.0, 0.48627450980392156, 0.0, 0.8]),
         'MT': np.array([0.10196078431372549, 0.788235294117647, 0.2196078431372549, 0.8]),
         'EBA': np.array([1.0, 1.0, 0, 0.8]),
         'FFA': np.array([1.0, 1.0, 0, 0.8]),
         'PPA': np.array([1.0, 1.0, 0, 0.8]),
         'LOC': np.array([1.0, 1.0, 0, 0.8]),
         'STS-Face': np.array([0.0, 0.8431372549019608, 1.0, 0.8]),
         'aSTS-SI': np.array([1.0, 0.7686274509803922, 0.0, 0.8]),
         'pSTS-SI': np.array([1.0, 0.7686274509803922, 0.0, 0.8]),
         }
    if key is not None:
        return d[key]
    else:
        return d


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        # self.models = ['indoor', 'expanse', 'object',
        #                'agent distance', 'facingness',
        #                'joint action', 'communication',
        #                'valence', 'arousal']
        self.subjs = ['01', '02', '03', '04']
        self.rois = ['EVC', 'MT', 'FFA', 'PPA', 'EBA', 'LOC', 'pSTS-SI', 'STS-Face', 'aSTS-SI']
        self.roi_cmap = sns.color_palette("hls")[2:7]

    def load_data(self):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/*pkl')
        for f in files:
            data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)

        # Remove TPJ
        df = df[df.roi != 'TPJ']

        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)

        # Make sid categorical
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)

        # Perform FDR correction and make a column for how the marker should appear
        df['p_corrected'] = 1
        for roi in self.rois:
            for subj in self.subjs:
                rows = (df.sid == subj) & (df.roi == roi)
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
        df['significant'] = 'ns'
        df.loc[(df['p_corrected'] < 0.05) & (df['p_corrected'] >= 0.01), 'significant'] = '*'
        df.loc[(df['p_corrected'] < 0.01) & (df['p_corrected'] >= 0.001), 'significant'] = '**'
        df.loc[(df['p_corrected'] < 0.001), 'significant'] = '**'
        return df

    def plot_results(self, df):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='poster', style='white', rc=custom_params)
        _, ax = plt.subplots(1, figsize=(10, 5))

        df['reliability'] = 0
        print(df.head())
        sns.barplot(x='roi', y='reliability',
                    hue='sid', color=[0.7, 0.7, 0.7],
                    edgecolor=".2",
                    ax=ax, data=df)
        sns.barplot(x='roi', y='r2',
                    hue='sid', palette='gray',
                    ax=ax, data=df)

        y_max = 0.7
        ax.set_ylim([0, y_max])

        ax.set_xlabel('')
        ax.set_xticklabels(self.rois,
                           fontsize=20,
                           rotation=45, ha='right')
        for ticklabel, pointer in zip(self.rois, ax.get_xticklabels()):
            color = roi2color(ticklabel)
            color[-1] = 1.
            pointer.set_color(color)
            pointer.set_weight('bold')

        # Plot vertical lines to separate the bars
        for x in np.arange(0.5, len(self.rois) - 0.5):
            ax.plot([x, x], [0, y_max - (y_max / 20)], '0.8')

        # Manipulate the color and add error bars
        for bar, (subj, roi) in zip(ax.patches[int(len(ax.patches) / 2):],
                                    itertools.product(self.subjs, self.rois)):
            color = roi2color(roi)
            color[:-1] = color[:-1] * subj2shade(subj)
            y1 = df.loc[(df.sid == subj) & (df.roi == roi),
                        'low_ci'].item()
            y2 = df.loc[(df.sid == subj) & (df.roi == roi),
                        'high_ci'].item()
            sig = df.loc[(df.sid == subj) & (df.roi == roi),
                         'significant'].item()
            x = bar.get_x() + 0.1
            ax.plot([x, x], [y1, y2], 'k')
            bar.set_color(color)
        ax.legend([], [], frameon=False)
        ax.set_ylabel('Explained variance ($r^2$)', fontsize=22)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/fullmodel_results.pdf')

    def run(self):
        data = self.load_data()
        data.to_csv(f'{self.out_dir}/{self.process}/roi_prediction.csv', index=False)
        print(data.head())
        self.plot_results(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotROIPrediction(args).run()


if __name__ == '__main__':
    main()
