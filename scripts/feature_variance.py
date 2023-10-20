#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.custom_plotting import feature_colors, custom_palette
import matplotlib.ticker as mticker


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

    def plot(self, df, title, color, ax):
        sns.histplot(x=title, data=df, color=color, ax=ax)
        ax.set_xlim([0, 1])
        # ax.set_ylim([0, int(len(df)*.8)])
        ax.set_xlabel('Rating')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title.capitalize(), fontsize=8)

    def load_annotations(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        if self.set != 'both':
            subset = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
            df = df.merge(subset)
        df.sort_values(by=['video_name'], inplace=True)
        df.rename(columns={'transitivity': 'object'}, inplace=True)
        return df.drop(columns=['video_name', 'indoor', 'dominance', 'cooperation', 'intimacy'])

    def run(self):
        palette = custom_palette(rgb=False)
        colors = feature_colors()
        df = self.load_annotations()
        sns.set(style='white', context='paper', rc={'figure.figsize': (2.5, 4)})
        fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
        ax = ax.flatten()
        for i, feature in enumerate(df.columns):
            color = colors[feature]
            rgb = palette[color]
            self.plot(df, feature, rgb, ax[i])
            ax[i].set_ylim([0, 100])

            if i % 2 == 0:
                #edit y ticks
                label_format = '{:.0f}'
                plt.locator_params(axis='y', nbins=3)
                y_ticklocs = ax[i].get_yticks().tolist()
                ax[i].yaxis.set_major_locator(mticker.FixedLocator(y_ticklocs))
                ax[i].set_yticklabels([label_format.format(x) for x in y_ticklocs], fontsize=6)
                ax[i].set_ylabel('Count', fontsize=8)

            if feature == 'valence' or 'arousal':
                #edit y ticks
                label_format = '{:.1f}'
                plt.locator_params(axis='x', nbins=3)
                x_ticklocs = ax[i].get_xticks().tolist()
                ax[i].xaxis.set_major_locator(mticker.FixedLocator(x_ticklocs))
                ax[i].set_xticklabels([label_format.format(x) for x in x_ticklocs], fontsize=6)

        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/set-{self.set}.svg')


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
