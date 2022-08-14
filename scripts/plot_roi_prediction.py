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


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    d.pop('r2var', None)
    d.pop('r2null', None)
    return d


def multiple_comp_correct(arr):
    out = multipletests(arr, alpha=0.05, method='fdr_bh')[1]
    return out


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.hemi = args.hemi
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        # Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        self.models = ['indoor', 'expanse', 'transitivity',
                       'agent_distance', 'facingness',
                       'joint_action', 'communication',
                       'valence', 'arousal']
        self.subjs = ['01', '02', '03', '04']
        self.rois = ['EVC', 'MT', 'EBA', 'face-pSTS', 'SI-pSTS', 'TPJ']

    def load_data(self):
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/*hemi-{self.hemi}*pkl')
        for f in files:
            data_list.append(load_pkl(f))
        df = pd.DataFrame(data_list)
        df['model'] = pd.Categorical(df['model'], ordered=True,
                                     categories=self.models)
        df['sid'] = pd.Categorical(df['sid'], ordered=True,
                                   categories=self.subjs)
        df['p_corrected'] = 1
        for roi in self.rois:
            for subj in self.subjs:
                rows = (df.sid == subj) & (df.roi == roi)
                df.loc[rows, 'p_corrected'] = multiple_comp_correct(df.loc[rows, 'p'])
        df['significant'] = np.nan
        df.loc[df['p_corrected'] < 0.05, 'significant'] = 0.05
        return df

    def plot_results(self, df):
        _, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        sns.set_theme(font_scale=1.5)
        for i, (ax, roi) in enumerate(zip(axes, self.rois)):
            sns.barplot(x='model', y='r2', hue='sid',
                        data=df.loc[df.roi == roi],
                        ax=ax).set(title=roi)
            ax.set_ylim([0, 0.055])
            sns.swarmplot(x='model', y='significant', hue='sid',
                          data=df.loc[df.roi == roi], ax=ax)
            ax.legend([], [], frameon=False)
            ax.set_xlabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(self.models,
                                   fontsize=16,
                                   rotation=45, ha='right')
            if i == 0 or i == 3:
                ax.set_ylabel('Unique variance (r^2)', fontsize=18)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/hemi-{self.hemi}.pdf')

    def run(self):
        data = self.load_data()
        print(data.head())
        self.plot_results(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hemi', type=str, default='rh')
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
