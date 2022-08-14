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


def mk_palette():
    pal = {
        'visual': '#F5DD40',
        'primitive': '#8558F4',
        'social': '#73D2DF',
        'affective': '#DA535B',
    }
    return pal


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        # Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        self.models = ['indoor', 'expanse', 'object',
                       'agent distance', 'facingness',
                       'joint action', 'communication',
                       'valence', 'arousal']
        self.subjs = ['01', '02', '03', '04']
        self.rois = ['EVC', 'MT', 'EBA', 'face-pSTS', 'SI-pSTS', 'TPJ']
        self.hemis = ['lh', 'rh']

    def load_data(self):
        # Load the results in their own dictionaries and create a dataframe
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/*pkl')
        for f in files:
            data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)

        # Replace names with how I want them to show on axis
        df.replace({'transitivity': 'object',
                    'agent_distance': 'agent distance',
                    'joint_action': 'joint action'},
                   inplace=True)
        # Using replacement make a column with the different categories
        df['model_cat'] = df.model.replace(model2cat())

        # Make model and sid categorical
        df['model'] = pd.Categorical(df['model'], ordered=True,
                                     categories=self.models)
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)

        # Perform FDR correction and make a column for the location that the marker should appear
        df['p_corrected'] = 1
        for roi in self.rois:
            for subj in self.subjs:
                rows = (df.sid == subj) & (df.roi == roi)
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
        df['significant'] = np.nan
        df.loc[df['p_corrected'] < 0.05, 'significant'] = 0.055

        return df

    def plot_results(self, df):
        _, axes = plt.subplots(2, 6, figsize=(30, 10))
        axes = axes.flatten()
        sns.set_theme(font_scale=1.5)
        palette = mk_palette()
        for i, (ax, (hemi, roi)) in enumerate(zip(axes,
                                                  itertools.product(self.hemis, self.rois))):
            if hemi == 'lh':
                title = f'l{roi}'
            else:
                title = f'r{roi}'

            sns.barplot(x='model', y='r2',
                        hue='sid', palette='gray',
                        data=df.loc[(df.roi == roi) & (df.hemi == hemi)],
                        ax=ax).set(title=title)
            ax.set_ylim([0, 0.06])
            sns.swarmplot(x='model', y='significant', hue='sid',
                          palette='gray',
                          data=df.loc[(df.roi == roi) & (df.hemi == hemi)],
                          ax=ax)
            ax.legend([], [], frameon=False)
            ax.set_xlabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < 6:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(self.models,
                                   fontsize=16,
                                   rotation=45, ha='right')
            if i == 0 or i == 6:
                ax.set_ylabel('Unique variance (r^2)', fontsize=18)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/roi_results.pdf')

    def run(self):
        data = self.load_data()
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
