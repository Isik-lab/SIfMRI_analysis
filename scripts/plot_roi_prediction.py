#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
from pathlib import Path
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def load_pkl(file):
    d = pickle.load(open(file, 'rb'))
    d.pop('r2var', None)
    d.pop('r2null', None)
    return d


class PlotROIPrediction:
    def __init__(self, args):
        self.process = 'PlotROIPrediction'
        self.hemi = args.hemi
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        # Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

    def load_data(self):
        data_list = []
        files = glob.glob(f'{self.out_dir}/ROIPrediction/sub-01*hemi-{self.hemi}*pkl')
        for f in files:
            data_list.append(load_pkl(f))
        return pd.DataFrame(data_list)

    def plot_results(self, df):
        rois = ['EVC', 'MT', 'EBA', 'face-pSTS', 'SI-pSTS', 'TPJ']
        _, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        for ax, roi in zip(axes, rois):
            sns.catplot(x='model', y='r2', hue='sid',
                        ax=ax,
                        data=df.loc[df.roi == roi],
                        title=roi)
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
