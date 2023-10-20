#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
from src.tools import perm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')


def q_map(q):
    if 0.01 < q < 0.05:
        out = '*'
    elif 0.001 < q < 0.01:
        out = '**'
    elif q < 0.001:
        out = '***'
    else:
        out = ''
    return out


def feature2color(key=None):
    d = dict()
    d['indoor'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
    d['spatial expanse'] = np.array([0.95703125, 0.86328125, 0.25, 0.8])
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


def plot_feature_corr(faces, title=None, out_dir=None):
    print(out_dir)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='paper', style='white', rc=custom_params)
    _, ax = plt.subplots(figsize=(3, 2.5))
    sns.barplot(x='feature', y='r', data=faces,
                errorbar=None,
                zorder=1, color='gray')
    ymin, ymax = ax.get_ylim()

    faces.set_index('feature', inplace=True)
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        ax.text(x, ymax,
                faces.loc[label._text, 'sig_text'],
                fontsize=10, horizontalalignment='center')

    for bar, label in zip(ax.patches, ax.get_xticklabels()):
        color = feature2color(label._text)
        bar.set_color(color)
    ax.set_ylim([ymin, ymax + (ymax * .2)])
    ax.set_xlabel('')
    ax.set_ylabel('Correlation ($r$)')
    plt.title(title)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir)
    plt.close('all')


class FeatureCorrelations:
    def __init__(self, args):
        self.process = 'FeatureCorrelations'
        self.data_dir = args.data_dir
        self.n_perm = args.n_perm
        self.precomputed = args.precomputed
        self.H0 = 'two_tailed'
        self.out_dir = f'{args.out_dir}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.features = ['indoor', 'expanse', 'transitivity', 'agent distance', 'facingness',
                         'joint action', 'communication', 'valence', 'arousal']
        self.plotting_features = ['indoor', 'spatial expanse', 'object', 'agent distance', 'facingness',
                                  'joint action', 'communication', 'valence', 'arousal']
        self.plot_feature_rename = {old: new for old, new in zip(self.features, self.plotting_features)}

    def save(self, df, name):
        df.to_csv(f'{self.out_dir}/{self.process}/{name}.csv', index=False)

    def load(self, name):
        df = pd.read_csv(f'{self.out_dir}/{self.process}/{name}.csv')
        df.feature = pd.Categorical(df.feature,
                                    categories=self.plotting_features,
                                    ordered=True)
        df.fillna('', inplace=True)
        return df

    def compute_correlation(self, df, face_parm):
        ps = []
        out_df = []
        for feature in tqdm(self.features, total=len(self.features)):
            r, p, rnull = perm(df[feature].to_numpy(), df[face_parm].to_numpy(),
                                       n_perm=self.n_perm,
                                       H0=self.H0, square=False, verbose=False)
            ps.append(p)
            out_df.append({'feature': feature, 'r': r, 'p': p,
                           'q': 1, 'sig': False, 'sig_text': ''})
        out_df = pd.DataFrame(out_df).set_index('feature')
        sigs, qs = fdrcorrection(ps)

        for feature, (sig, q) in zip(self.features, zip(sigs, qs)):
            out_df.loc[feature, 'q'] = q
            out_df.loc[feature, 'sig'] = sig
            out_df.loc[feature, 'sig_text'] = q_map(q)
        out_df.reset_index(drop=False, inplace=True)
        out_df.replace(self.plot_feature_rename, inplace=True)
        out_df.feature = pd.Categorical(out_df.feature,
                                        categories=self.plotting_features,
                                        ordered=True)
        self.save(out_df, face_parm)
        return out_df

    def load_annotations(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        faces = pd.read_csv(f'{self.data_dir}/annotations/face_annotations.csv')
        df = df.merge(faces, on='video_name')
        df = df.drop(columns=['video_name', 'cooperation', 'dominance', 'intimacy'])
        return df

    def run(self):
        df = self.load_annotations()

        if not self.precomputed:
            face_area = self.compute_correlation(df, 'face_area')
            face_centrality = self.compute_correlation(df, 'face_centrality')
        else:
            face_area = self.load('face_area')
            face_centrality = self.load('face_centrality')

        plot_feature_corr(face_area, title='Face area', out_dir=f'{self.figure_dir}/face_area.svg')
        plot_feature_corr(face_centrality, title='Face centrality', out_dir=f'{self.figure_dir}/face_centrality.svg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_perm', type=int, default=int(5e3))
    parser.add_argument('--precomputed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeatureCorrelations(args).run()


if __name__ == '__main__':
    main()
