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
    return d


def multiple_comp_correct(arr):
    out = multipletests(arr, alpha=0.05, method='fdr_bh')[1]
    return out


def subj2shade(key):
    d = {'01': 1.0,
         '02': 0.8,
         '03': 0.6,
         '04': 0.4}
    return d[key]


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['object'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['agent distance'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['facingness'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['joint action'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['communication'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['valence'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    d['arousal'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
    if key is not None:
        return d[key]
    else:
        return d


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.n_perm = args.n_perm
        self.individual = args.individual
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.y_max = 0
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        self.all_rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA', 'PPA', 'face-pSTS', 'pSTS', 'aSTS']
        self.all_features = ['indoor', 'expanse', 'transitivity', 'agent_distance',
                         'facingness', 'joint_action', 'communication', 'valence', 'arousal']
        self.features = ['indoor', 'expanse', 'object', 'agent distance',
                         'facingness', 'joint action', 'communication', 'valence', 'arousal']
        self.file_rename_map = {key: val for key, val in zip(self.all_features, self.features)}
        self.subjs = ['01', '02', '03', '04']
        if self.individual:
            self.out_prefix = 'individual_'
        else:
            self.out_prefix = 'group_'

        if args.stream == 'lateral':
            self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS-SI', 'STS-Face', 'aSTS-SI']
            self.out_prefix += 'lateral-rois_'
        else:
            self.rois = ['FFA', 'PPA']
            self.out_prefix += 'ventral-rois_'

        if args.unique_variance:
            self.y_label = 'Unique variance'
            if args.include_nuisance:
                self.file_id = 'dropped-featurewithnuisance'
                self.out_prefix += 'dropped-featurewithnuisance'
            else:
                self.file_id = 'dropped-feature'
                self.out_prefix += 'dropped-feature'
        else:
            assert False, 'not implemented'

    def load_group_data(self, name):
        print(name)
        data_list = []
        for feature, roi in itertools.product(self.all_features, self.all_rois):
            files = glob.glob(f'{self.out_dir}/ROIPrediction/*{name}-{feature}*roi-{roi}*pkl')
            print(f'{name}-{feature}*roi-{roi}', len(files))
            if files:
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
                data = {'roi': roi, 'r2': r2, 'feature': feature}
                data['p'] = tools.calculate_p(r2null, r2, self.n_perm, 'greater')
                data['low_ci'], data['high_ci'] = np.percentile(r2var, [2.5, 97.5])
                data_list.append(data)
        df = pd.DataFrame(data_list)
        df = df.loc[df.roi != 'TPJ']

        df['p_corrected'] = 1
        for roi in self.all_rois:
                rows = (df.roi == roi)
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
        df['significant'] = 'ns'
        df.loc[(df['p_corrected'] < 0.05) & (df['p_corrected'] >= 0.01), 'significant'] = '*'
        df.loc[(df['p_corrected'] < 0.01) & (df['p_corrected'] >= 0.001), 'significant'] = '**'
        df.loc[(df['p_corrected'] < 0.001), 'significant'] = '***'

        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)
        df.replace(self.file_rename_map, inplace=True)
        self.y_max = tools.round_decimals_up(df.high_ci.max(), 1) + 0.05
        print(self.y_max)
        df = df.loc[df['roi'].isin(self.rois)]
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)
        df['feature'] = pd.Categorical(df['feature'], ordered=True,
                                   categories=self.features)
        return df

    def load_individual_data(self, name):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        for feature in self.all_features:
            files = glob.glob(f'{self.out_dir}/ROIPrediction/*{name}-{feature}_roi*pkl')
            print(f'{name}-{feature}', len(files))
            for f in files:
                data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)

        # Perform FDR correction and make a column for how the marker should appear
        df.drop(columns=['reliability'], inplace=True)
        df.replace({'transitivity': 'object',
                    'agent_distance': 'agent distance',
                    'joint_action': 'joint action'}, inplace=True)
        df['feature'] = pd.Categorical(df['feature'], ordered=True,
                                        categories=self.features)
        #Remove TPJ before multiple comparisons correction
        df = df.loc[df.roi != 'TPJ']

        df['p_corrected'] = 1
        for roi in self.all_rois:
            for subj in self.subjs:
                rows = (df.sid == subj) & (df.roi == roi)
                print(roi, subj, np.sum(rows))
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
        df['significant'] = 'ns'
        df.loc[(df['p_corrected'] < 0.05) & (df['p_corrected'] >= 0.01), 'significant'] = '*'
        df.loc[(df['p_corrected'] < 0.01) & (df['p_corrected'] >= 0.001), 'significant'] = '**'
        df.loc[(df['p_corrected'] < 0.001), 'significant'] = '**'

        # Rename some ROIs
        df.replace({'face-pSTS': 'STS-Face',
                    'pSTS': 'pSTS-SI',
                    'aSTS': 'aSTS-SI'},
                   inplace=True)

        # Make sid categorical
        self.y_max = tools.round_decimals_up(df.high_ci.max(), 1) + 0.05
        print(self.y_max)
        df = df.loc[df['roi'].isin(self.rois)]
        df.drop(columns=['unique_variance', 'category', 'r2var', 'r2null'], inplace=True)
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)
        df['roi'] = pd.Categorical(df['roi'], ordered=True,
                                   categories=self.rois)
        return df

    def plot_group_results(self, df):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(context='poster', style='white', rc=custom_params)
        _, axes = plt.subplots(1, len(self.rois), figsize=(int(len(self.rois) * 6), 8))

        for i, (ax, roi) in enumerate(zip(axes, self.rois)):
            cur_df = df.loc[df.roi == roi]
            sns.barplot(x='feature', y='r2', palette='gray', saturation=0.8,
                        data=cur_df,
                        ax=ax)
            ax.set_title(roi, fontsize=26)
            ax.set_xlabel('')
            ax.set_ylim([0, self.y_max])

            # Change the ytick font size
            label_format = '{:,.2f}'
            y_ticklocs = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
            ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=20)

            # Change the xaxis font size and colors
            ax.set_xticklabels(self.features,
                               fontsize=20,
                               rotation=45, ha='right')
            for ticklabel, pointer in zip(self.features, ax.get_xticklabels()):
                color = feature2color(ticklabel)
                # color[-1] = 1.
                pointer.set_color(color)
                pointer.set_weight('bold')

            # Remove the yaxis label from all plots except the two leftmost plots
            if i == 0:
                ax.set_ylabel(f'{self.y_label} ($r^2$)', fontsize=22)
            else:
                ax.set_ylabel('')

            # Manipulate the color and add error bars
            for bar, feature in zip(ax.patches, self.features):
                color = feature2color(feature)
                y1 = cur_df.loc[(cur_df.feature == feature), 'low_ci'].item()
                y2 = cur_df.loc[(cur_df.feature == feature), 'high_ci'].item()
                sig = cur_df.loc[(cur_df.feature == feature), 'significant'].item()
                width = bar.get_width()
                x = bar.get_x() + (width/2)
                ax.plot([x, x], [y1, y2], 'k')
                if sig != 'ns':
                    ax.text(x, self.y_max - 0.02, sig,
                            horizontalalignment='center',
                            fontsize=16)
                bar.set_color(color)
            ax.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/{self.out_prefix}.pdf')

    def plot_individual_results(self, df):
        _, axes = plt.subplots(1, len(self.rois), figsize=(int(len(self.rois)*6), 8))
        sns.set_theme(font_scale=2)
        for i, (ax, roi) in enumerate(zip(axes, self.rois)):
            cur_df = df.loc[df.roi == roi]
            sns.barplot(x='feature', y='r2',
                        hue='sid', palette='gray', saturation=0.8,
                        data=cur_df,
                        ax=ax).set(title=roi)
            ax.set_xlabel('')
            ax.set_ylim([0, self.y_max])

            # Change the ytick font size
            label_format = '{:,.2f}'
            y_ticklocs = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
            ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=20)

            # Remove lines on the top and left of the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Change the xaxis font size and colors
            ax.set_xticklabels(self.features,
                               fontsize=20,
                               rotation=45, ha='right')
            for ticklabel, pointer in zip(self.features, ax.get_xticklabels()):
                color = feature2color(ticklabel)
                # color[-1] = 1.
                pointer.set_color(color)
                pointer.set_weight('bold')

            # Remove the yaxis label from all plots except the two leftmost plots
            if i == 0 or i == len(self.rois):
                ax.set_ylabel(f'{self.y_label} ($r^2$)', fontsize=22)
            else:
                ax.set_ylabel('')

            # Plot vertical lines to separate the bars
            ax.vlines(np.arange(0.5, len(self.features) - 0.5),
                      ymin=0, ymax=self.y_max - (self.y_max / 20),
                      colors='lightgray')

            # Manipulate the color and add error bars
            for bar, (subj, feature) in zip(ax.patches, itertools.product(self.subjs, self.features)):
                color = feature2color(feature)
                color[:-1] = color[:-1] * subj2shade(subj)
                y1 = cur_df.loc[(cur_df.sid == subj) & (cur_df.feature == feature), 'low_ci'].item()
                y2 = cur_df.loc[(cur_df.sid == subj) & (cur_df.feature == feature), 'high_ci'].item()
                sig = cur_df.loc[(cur_df.sid == subj) & (cur_df.feature == feature), 'significant'].item()
                x = bar.get_x() + 0.1
                ax.plot([x, x], [y1, y2], 'k')
                if sig != 'ns':
                    ax.scatter(x, self.y_max - 0.02, marker='o', color=color)
                bar.set_color(color)

            ax.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/{self.out_prefix}.pdf')

    def run(self):
        if self.individual:
            data = self.load_individual_data(self.file_id)
            self.plot_individual_results(data)
        else:
            data = self.load_group_data(self.file_id)
            self.plot_group_results(data)
        data.to_csv(f'{self.out_dir}/{self.process}/{self.out_prefix}.csv', index=False)
        print(data.head())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, default='lateral')
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--individual', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--include_nuisance', action=argparse.BooleanOptionalAction, default=False)
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
