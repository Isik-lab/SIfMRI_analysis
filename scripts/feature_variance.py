#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.custom_plotting import feature_colors, custom_palette


class FeatureVariance():
    def __init__(self, args):
        self.process = 'FeatureVariance'
        self.set = args.set
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def plot(self, df, title, color):
        sns.set(style='white', context='talk', rc={'figure.figsize': (5, 5)})
        fig, ax = plt.subplots()
        sns.histplot(x=title, data=df, color=color)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, int(len(df)*.8)])
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(title.capitalize())
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/{title}_set-{self.set}.pdf')
        plt.close(fig)

    def load_annotations(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        return df.drop('video_name', axis=1)

    def run(self):
        palette = custom_palette(rgb=False)
        colors = feature_colors()
        df = self.load_annotations()
        for feature in df.columns:
            print(feature)
            color = colors[feature]
            rgb = palette[color]
            self.plot(df, feature, rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeatureVariance(args).run()

if __name__ == '__main__':
    main()
