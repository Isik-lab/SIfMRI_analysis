#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from src.custom_plotting import custom_seaborn_cmap, feature_categories


def plot_feature_correlation(cur, ax):
    cmap = custom_seaborn_cmap()
    order = cur.columns.to_list()
    cur = cur.sort_values(by='Spearman rho', ascending=False)
    sns.barplot(y='Spearman rho', x='Feature',
                hue='category',
                # order=order,
                data=cur, ax=ax,
                palette=cmap,
                dodge=False)
    plt.title('')
    ax.tick_params(axis='x', labelrotation=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel(r'Correlation ($\rho$)')
    ax.set_ylim([-1, 0.7])
    legend = ax.get_legend()
    legend.remove()


def plot_video_loadings(loading, videos, ax):
    indices = np.flip(np.argsort(loading))
    df = pd.DataFrame({'Videos': videos[indices],
                       'Loading': loading[indices]})
    sns.barplot(y='Loading', x='Videos',
                data=df, ax=ax, color='gray',
                dodge=False)
    plt.title('')
    ax.tick_params(axis='x', labelrotation=90, labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    legend = ax.legend()
    legend.remove()


def pca(X, n_components):
    model = PCA(whiten=True, n_components=n_components, random_state=0)
    out = model.fit_transform(X)
    return out, model.components_, model.explained_variance_ratio_


class FeaturePCA():
    def __init__(self, args):
        self.process = 'FeaturePCA'
        self.set = args.set
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
        self.n_components = args.n_components
        if self.n_components > 1:
            self.n_components = int(self.n_components)

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)

        new = df.copy()
        # new['motion energy'] = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{self.set}_avg.npy')
        # new['AlexNet conv2'] = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{self.set}_avg.npy')
        names = new.columns.to_list()
        names.remove('video_name')
        names = np.array(names)

        df = df.drop(columns=['video_name'])
        categories = pd.read_csv(f'{self.data_dir}/annotations/{self.set}_categories.csv')
        return names, df, new, categories.action_categories.to_numpy()

    def plot_variance(self, vals):
        fig, ax = plt.subplots()
        ax.plot(vals, '.-r')
        ax.plot(np.ones(len(vals))*0.8, 'k', alpha=0.3)
        ax.plot(np.ones(11) * np.sum(vals <= 0.8), np.arange(0, 1.1, 0.1), '--k')
        ax.plot(np.ones(len(vals)) * 0.95, 'k', alpha=0.3)
        ax.plot(np.ones(11) * np.sum(vals <= 0.95), np.arange(0, 1.1, 0.1), '--k')
        plt.ylim([0, 1.05])
        plt.xlabel('PCs')
        plt.ylabel('Explained variance')
        plt.savefig(f'{self.figure_dir}/explained_variance_set-{self.set}.pdf')

    def PC_to_features(self, features, feature_names, vid_comp, explained_variance):
        categories = feature_categories()
        df = pd.DataFrame()
        for iPC in range(vid_comp.shape[-1]):
            d = dict()
            for feature in feature_names:
                rho, _ = spearmanr(features[feature], vid_comp[:, iPC])
                d['Feature'] = feature
                d['Spearman rho'] = rho
                d['PC'] = [iPC]
                d['category'] = categories[feature]
                d['Explained variance'] = [explained_variance[iPC]]
                df = pd.concat([df, pd.DataFrame(d)])
        df.category = pd.Categorical(df.category,
                              categories=['scene', 'object', 'social primitive', 'social', 'low-level model'],
                              ordered=True)
        df.to_csv(f'{self.out_dir}/{self.process}/PCs_set-{self.set}.csv', index=False)
        return df

    def plot_PC_results(self, df, videos, vid_comp):
        sns.set(style='white', context='poster')
        for i, iname in enumerate(np.unique(df.PC)):
            # _, ax = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1, 1.5]})
            _, ax = plt.subplots(1, 1, figsize=(7, 7))
            plot_feature_correlation(df[df['PC'] == iname], ax)
            # plot_video_loadings(vid_comp[:, i], videos, ax[1])
            plt.xticks(rotation=90)
            ev = df.loc[df['PC'] == iname, "Explained variance"].unique()[0]
            plt.suptitle(f'PC {i + 1} \n Explained variance = {np.round(ev*100):.0f}%', fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{self.figure_dir}/PC{str(i).zfill(2)}_set-{self.set}.pdf')
            plt.close()

    def videos(self, vid_comp, df):
        def copy_videos(inds_, df_, pc, part):
            for i, ind in enumerate(inds_):
                name = df_.loc[ind, 'video_name']
                Path(f'{self.figure_dir}/videos/PC-{pc}').mkdir(exist_ok=True, parents=True)
                shutil.copyfile(f'{self.data_dir}/videos/{name}',
                                f'{self.figure_dir}/videos/PC-{pc}/{part}-{i}_{name}')

        for icomp in range(self.n_components):
            inds = np.argsort(vid_comp[:, icomp])
            copy_videos(inds[:5], df, icomp, 'bottom')
            copy_videos(inds[-5:], df, icomp, 'top')

    def run(self):
        feature_names, features, features_low, videos = self.load_features()
        vid_comp, comp_feature, explained_variance = pca(features, self.n_components)
        self.videos(vid_comp, features_low)
        self.plot_variance(explained_variance.cumsum())
        df = self.PC_to_features(features_low, feature_names, vid_comp, explained_variance)
        self.plot_PC_results(df, videos, vid_comp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=float, default=4)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeaturePCA(args).run()

if __name__ == '__main__':
    main()
