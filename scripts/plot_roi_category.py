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


def subj2shade(key):
    d = {'01': 1.0,
         '02': 0.8,
         '03': 0.6,
         '04': 0.4}
    return d[key]


def cat2color(key=None):
    d = dict()
    d['scene & object'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['social primitives'] = np.array([0.51953125, 0.34375, 0.953125, 0.8])
    d['social'] = np.array([0.44921875, 0.8203125, 0.87109375, 0.8])
    d['affective'] = np.array([0.8515625, 0.32421875, 0.35546875, 0.8])
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
        self.categories = ['scene & object', 'social primitives', 'social', 'affective']
        self.subjs = ['01', '02', '03', '04']
        self.rois = ['EVC', 'MT', 'EBA', 'STS-Face', 'STS-SI']
        self.hemis = ['lh', 'rh']

    def load_reliability(self):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/*reliability.pkl')
        for f in files:
            data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)
        df.replace({'face-pSTS': 'STS-Face',
                    'SI-pSTS': 'STS-SI'},
                   inplace=True)
        df.rename(columns={'r2': 'reliability'}, inplace=True)
        return df

    def load_data(self):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROICategory/*category*pkl')
        for f in files:
            if not 'None' in f:
                data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)
        df.to_csv(f'{self.out_dir}/{self.process}/roi_category.csv', index=False)

        # Replace names with how I want them to show on axis
        df.replace({'social_primitive': 'social primitives',
                    'scene_object': 'scene & object'},
                   inplace=True)
        df.replace({'face-pSTS': 'STS-Face',
                    'SI-pSTS': 'STS-SI'},
                   inplace=True)

        # Make model and sid categorical
        df['category'] = pd.Categorical(df['category'], ordered=True,
                                     categories=self.categories)
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)

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

        # Remove TPJ
        df = df[df.roi != 'TPJ']
        # df.to_csv(f'{self.out_dir}/{self.process}/ro i_prediction.csv', index=False)
        return df

    def plot_results(self, df):
        _, axes = plt.subplots(2, len(self.rois), figsize=(30, 10))
        axes = axes.flatten()
        sns.set_theme(font_scale=2)
        for i, (ax, (hemi, roi)) in enumerate(zip(axes,
                                                  itertools.product(self.hemis, self.rois))):
            if hemi == 'lh':
                title = f'l{roi}'
            else:
                title = f'r{roi}'

            sns.barplot(x='category', y='r2',
                        hue='sid', palette='gray', saturation=0.8,
                        data=df.loc[(df.roi == roi) & (df.hemi == hemi)],
                        ax=ax).set(title=title)
            ax.set_xlabel('')
            y_max = df.loc[(df.roi == roi) & (df.hemi == hemi), 'high_ci'].max() + 0.04
            ax.set_ylim([0, y_max])

            # Change the ytick font size
            label_format = '{:,.2f}'
            y_ticklocs = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
            ax.set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=20)

            # Remove lines on the top and left of the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Change the xaxis font size and colors
            if i < len(self.rois):
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(self.categories,
                                   fontsize=20,
                                   rotation=45, ha='right')
                for ticklabel, pointer in zip(self.categories, ax.get_xticklabels()):
                    color = cat2color(ticklabel)
                    color[-1] = 1.
                    pointer.set_color(color)
                    pointer.set_weight('bold')

            # Remove the yaxis label from all plots except the two leftmost plots
            if i == 0 or i == len(self.rois):
                ax.set_ylabel('Unique variance ($r^2$)', fontsize=22)
            else:
                ax.set_ylabel('')

            # Plot vertical lines to separate the bars
            ax.vlines(np.arange(0.5, len(self.categories)-0.5),
                      ymin=0, ymax=y_max-(y_max/20),
                      colors='lightgray')

            # Manipulate the color and add error bars
            for bar, (subj, category) in zip(ax.patches,
                                          itertools.product(self.subjs, self.categories)):
                color = cat2color(category)
                color[:-1] = color[:-1] * subj2shade(subj)
                y1 = df.loc[(df.sid == subj) & (df.category == category) & (df.hemi == hemi) & (df.roi == roi),
                            'low_ci'].item()
                y2 = df.loc[(df.sid == subj) & (df.category == category) & (df.hemi == hemi) & (df.roi == roi),
                            'high_ci'].item()
                sig = df.loc[(df.sid == subj) & (df.category == category) & (df.hemi == hemi) & (df.roi == roi),
                            'significant'].item()
                x = bar.get_x() + 0.1
                ax.plot([x, x], [y1, y2], 'k')
                if sig != 'ns':
                    ax.scatter(x, y_max-0.02, marker='o', color=color)
                bar.set_color(color)

            ax.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/roi_category.pdf')

    def run(self):
        data = self.load_data()
        # reliability = self.load_reliability()
        # data = data.merge(reliability,
        #                   left_on=['hemi', 'sid', 'roi'],
        #                   right_on=['hemi', 'sid', 'roi'],
        #                   how='left')
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
