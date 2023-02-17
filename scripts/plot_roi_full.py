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
from src import tools


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    # d.pop('r2var', None)
    # d.pop('r2null', None)
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


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.stream = args.stream
        self.n_perm = args.n_perm
        self.individual = args.individual
        self.reliability_mean = args.reliability_mean
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        self.subjs = ['01', '02', '03', '04']
        self.all_rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA', 'PPA', 'face-pSTS', 'pSTS', 'aSTS']
        if self.individual:
            self.out_prefix = 'individual_'
        else:
            self.out_prefix = 'group_'

        if self.stream == 'lateral':
            self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS-SI', 'STS-Face', 'aSTS-SI']
            self.roi_cmap = sns.color_palette("hls")[:len(self.rois)]
            self.out_prefix += 'lateral-rois_full-model'
        else:
            self.rois = ['FFA', 'PPA']
            self.roi_cmap = sns.color_palette("hls")[:len(self.rois)]
            self.out_prefix += 'ventral-rois_full-model'

    def load_group_reliability(self):
        data_list = []
        for roi in self.all_rois:
            files = glob.glob(f'{self.out_dir}/ROIPrediction/*roi-{roi}*reliability*pkl')
            r2 = np.zeros(len(files))
            for i, f in enumerate(files):
                in_data = load_pkl(f)
                r2[i] = in_data['reliability']
            data = {'roi': roi, 'reliability_min': r2.min(), 'reliability_max': r2.max(), 'reliability_mean': r2.mean()}
            data_list.append(data)
        df = pd.DataFrame(data_list)
        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)
        df = df.loc[df['roi'].isin(self.rois)]
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)
        return df

    def load_group_data(self, name):
        data_list = []
        for roi in self.all_rois:
            files = glob.glob(f'{self.out_dir}/ROIPrediction/*{name}*roi-{roi}*pkl')
            r2 = 0
            r2null = np.zeros(self.n_perm)
            r2var = np.zeros(self.n_perm)
            for f in files:
                in_data = load_pkl(f)
                r2 += in_data['r2']
                r2null += in_data['r2null']
                r2var += in_data['r2var']
            r2 /= len(files)
            r2null /= len(files)
            r2var /= len(files)
            data = {'roi': roi, 'r2': r2}
            data['p'] = tools.calculate_p(r2null, r2, self.n_perm, 'greater')
            data['low_ci'], data['high_ci'] = np.percentile(r2var, [2.5, 97.5])
            data_list.append(data)
        df = pd.DataFrame(data_list)
        df = df.loc[df.roi != 'TPJ'] #drop TPJ

        df['p_corrected'] = multiple_comp_correct(df['p'])
        df['significant'] = 'ns'
        df.loc[(df['p_corrected'] < 0.05) & (df['p_corrected'] >= 0.01), 'significant'] = '*'
        df.loc[(df['p_corrected'] < 0.01) & (df['p_corrected'] >= 0.001), 'significant'] = '**'
        df.loc[(df['p_corrected'] < 0.001), 'significant'] = '***'

        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)
        df = df.loc[df['roi'].isin(self.rois)]
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)
        return df

    def load_individual_data(self, name):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/*{name}*pkl')
        for f in files:
            data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)

        # Remove TPJ and rename some ROIs
        df.drop(columns=['unique_variance', 'feature', 'category'], inplace=True)

        if 'reliability' not in name:
            # Perform FDR correction and make a column for how the marker should appear
            df.drop(columns=['reliability', 'r2null', 'r2var'], inplace=True)

            df['p_corrected'] = 1
            for subj in self.subjs:
                rows = (df.sid == subj)
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
            df['significant'] = 'ns'
            df.loc[(df['p_corrected'] < 0.05) & (df['p_corrected'] >= 0.01), 'significant'] = '*'
            df.loc[(df['p_corrected'] < 0.01) & (df['p_corrected'] >= 0.001), 'significant'] = '**'
            df.loc[(df['p_corrected'] < 0.001), 'significant'] = '**'

        # Make sid categorical
        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)
        df = df.loc[df['roi'].isin(self.rois)]
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)
        return df

    def plot_group_results(self, df, font=6):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='talk', style='whitegrid', rc=custom_params)
        if self.stream == 'lateral':
            _, ax = plt.subplots(1, figsize=(6.5, 2))
        else:
            _, ax = plt.subplots(1, figsize=(2, 2))
        sns.barplot(x='roi', y='r2',
                    color=[0.87, 0.67, 0.87], edgecolor=[0.2, 0.2, 0.2],
                    ax=ax, data=df)

        if self.reliability_mean:
            y_max = df.reliability_mean.max() + 0.04
        else:
            y_max = df.reliability_max.max() + 0.04
        ax.set_ylim([0, y_max])

        ax.set_xlabel('')
        ax.set_xticklabels(self.rois, fontsize=font)

        # Change the ytick font size
        label_format = '{:,.2f}'
        y_ticklocs = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
        ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=font)
        ax.set_ylabel('Explained variance ($r^2$)', fontsize=font + 2)

        for bar, roi in zip(ax.patches, self.rois):
            y1 = df.loc[(df.roi == roi), 'low_ci'].item()
            y2 = df.loc[(df.roi == roi), 'high_ci'].item()
            sig = df.loc[(df.roi == roi), 'significant'].item()
            width = bar.get_width()
            x = bar.get_x() + (width/2)
            #Plot noise ceiling
            if self.reliability_mean:
                r1 = df.loc[df.roi == roi, 'reliability_mean'].item()
                ax.plot([x-(width/2), x+(width/2)], [r1, r1], color='k', alpha=0.1)
            else:
                r1 = df.loc[df.roi == roi, 'reliability_min'].item()
                r2 = df.loc[df.roi == roi, 'reliability_max'].item()
                ax.fill_between([x-(width/2), x+(width/2)], r1, r2,
                                color='k', alpha=0.25, edgecolor=[1, 1, 1, 0])
            #Plot error bars
            ax.plot([x, x], [y1, y2], 'k', linewidth=1.5)
            if sig != 'ns':
                ax.text(x, y_max - 0.02, sig,
                        horizontalalignment='center')
        ax.legend([], [], frameon=False)
        ax.set_xlabel('')
        ax.set_ylabel('Explained variance ($r^2$)', fontsize=font+2)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/{self.out_prefix}.svg')

    def plot_individual_results(self, df, font=6):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='paper', style='whitegrid', rc=custom_params)
        if self.stream == 'lateral':
            _, ax = plt.subplots(1, figsize=(6.5, 2))
        else:
            _, ax = plt.subplots(1, figsize=(2.5, 2))

        sns.barplot(x='roi', y='reliability',
                    hue='sid', color=[0.7, 0.7, 0.7],
                    edgecolor=".2",
                    ax=ax, data=df)
        sns.barplot(x='roi', y='r2',
                    hue='sid', color=[0.8, 0, 0.8],
                    ax=ax, data=df)

        y_max = df.reliability.max() + 0.05
        ax.set_ylim([0, y_max])

        ax.set_xlabel('')
        ax.set_xticklabels(self.rois, fontsize=font)

        # Change the ytick font size
        label_format = '{:,.2f}'
        y_ticklocs = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
        ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=font)
        ax.set_ylabel('Explained variance ($r^2$)', fontsize=font+2)

        # Plot vertical lines to separate the bars
        # ax.vlines(np.arange(0.5, len(self.rois) - 0.5),
        #           ymin=0, ymax=np.round(y_max),
        #           colors='lightgray', alpha=0.5)

        # Manipulate the color and add error bars
        for bar, (subj, roi) in zip(ax.patches[int(len(ax.patches) / 2):],
                                    itertools.product(self.subjs, self.rois)):
            color = np.array(bar.get_facecolor())
            color[:-1] = color[:-1] * subj2shade(subj)
            y1 = df.loc[(df.sid == subj) & (df.roi == roi),
                        'low_ci'].item()
            y2 = df.loc[(df.sid == subj) & (df.roi == roi),
                        'high_ci'].item()
            sig = df.loc[(df.sid == subj) & (df.roi == roi),
                         'significant'].item()
            x = bar.get_x() + 0.1
            ax.plot([x, x], [y1, y2], 'k', linewidth=1.5)
            if sig != 'ns':
                ax.scatter(x, y_max - 0.02, marker='o', color=color, s=8, edgecolors=[0.2, 0.2, 0.2])
            bar.set_facecolor(color)
            bar.set_edgecolor((.2, .2, .2))
        ax.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/{self.out_prefix}.svg')

    def run(self):
        if self.individual:
            data = self.load_individual_data('full-model')
            data = data.merge(self.load_individual_data('reliability'),
                              left_on=['sid', 'roi'],
                              right_on=['sid', 'roi'])
            self.plot_individual_results(data)
        else:
            data = self.load_group_data('full-model')
            data = data.merge(self.load_group_reliability(), on='roi')
            self.plot_group_results(data)
        data.to_csv(f'{self.out_dir}/{self.process}/{self.out_prefix}.csv', index=False)
        print(data.head())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, default='lateral')
    parser.add_argument('--reliability_mean', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--individual', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_perm', type=int, default=10000)
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
