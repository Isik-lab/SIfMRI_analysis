#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
from src import custom_plotting as cp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nilearn import datasets, surface
import nibabel as nib
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def plot_feature_correlation(cur, ax):
    cur = cur.sort_values(by='Spearman rho', ascending=False)
    sns.barplot(y='Spearman rho', x='Feature',
                data=cur, ax=ax, color='gray')
    plt.title('')
    ax.tick_params(labelrotation=90)


def plot_video_loadings(loading, videos, ax):
    indices = np.flip(np.argsort(loading))
    df = pd.DataFrame({'Videos': videos[indices],
                       'Loading': loading[indices]})
    sns.barplot(y='Loading', x='Videos',
                data=df, ax=ax, color='gray')
    plt.title('')
    ax.tick_params(labelrotation=90)


def pca(X, n_components):
    model = PCA(whiten=True, n_components=n_components)
    out = model.fit_transform(X)
    return out, model.components_, model.explained_variance_ratio_


def mkcmap():
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    return cmap


class VoxelPCA():
    def __init__(self, args):
        self.process = 'VoxelPCA'
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
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
        self.cmap = mkcmap()

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df = df.merge(train)
        df['motion energy'] = np.load(f'{self.out_dir}/of_activations/of_adelsonbergen_avg.npy')
        df['AlexNet conv2'] = np.load(f'{self.out_dir}/alexnet_activations/alexnet_conv2_avg.npy')
        df['AlexNet conv5'] = np.load(f'{self.out_dir}/alexnet_activations/alexnet_conv5_avg.npy')
        df.sort_values(by=['video_name'], inplace=True)
        new = df.drop(columns=['video_name'])

        categories = pd.read_csv(f'{self.data_dir}/annotations/train_categories.csv')
        return np.array(new.columns), df, categories.action_categories.to_numpy()

    def load_neural(self, n_subjects=4):
        X = []
        mask = np.load(f'{self.out_dir}/Reliability/sub-all_reliability-mask.npy').astype('bool')
        for sid_ in range(n_subjects):
            sid = str(sid_ + 1).zfill(2)
            betas = np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_train-data.npy')

            # Filter the beta values to the reliable voxels
            betas = betas[mask, :]

            # Mean center the activation within subject
            offset_subject = betas.mean()
            betas -= offset_subject

            if type(X) is list:
                X = betas.T
            else:
                X = np.hstack([X, betas.T])
        return StandardScaler().fit_transform(X), np.sum(mask)

    def load_mask(self):
        im = nib.load(f'{self.out_dir}/Reliability/sub-all_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.out_dir}/Reliability/sub-all_reliability-mask.npy')
        return mask, im

    def vol_to_surf(self, im, hemi):
        return surface.vol_to_surf(im, surf_mesh=self.fsaverage[f'pial_{hemi}'], radius=2.)

    def plot_brain(self, stat, mask, im):
        volume = cp.mkNifti(stat, mask, im)
        texture = {'left': self.vol_to_surf(volume, 'left'),
                   'right': self.vol_to_surf(volume, 'right')}
        cp.plot_surface_stats(self.fsaverage, texture,
                              cmap=self.cmap, threshold=1,
                              output_file=f'{self.figure_dir}/brain_PCs.pdf')

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
        plt.savefig(f'{self.figure_dir}/explained_variance.pdf')

    def PC_to_features(self, features, feature_names, vid_comp):
        df = pd.DataFrame()
        for iPC in range(vid_comp.shape[-1]):
            d = dict()
            for feature in feature_names:
                rho, _ = spearmanr(features[feature], vid_comp[:, iPC])
                d['Feature'] = feature
                d['Spearman rho'] = rho
                d['PC'] = [iPC]
                df = pd.concat([df, pd.DataFrame(d)])
        df.to_csv(f'{self.out_dir}/{self.process}/PCs.csv', index=False)
        return df

    def plot_PC_results(self, df, videos, vid_comp):
        for i, iname in enumerate(np.unique(df.PC)):
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_feature_correlation(df[df['PC'] == iname], ax[0])
            plot_video_loadings(vid_comp[:, i], videos, ax[1])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{self.figure_dir}/PC{str(i).zfill(2)}.pdf')

    def run(self):
        mask, im = self.load_mask()
        feature_names, features, videos = self.load_features()
        neural, n_voxels = self.load_neural()
        vid_comp, comp_vox, explained_variance = pca(neural, self.n_components)
        self.plot_variance(explained_variance.cumsum())
        df = self.PC_to_features(features, feature_names, vid_comp)
        self.plot_PC_results(df, videos, vid_comp)
        vox = np.argmax(comp_vox.reshape((-1, 4, n_voxels)).mean(axis=-2), axis=0) + 1
        self.plot_brain(vox, mask, im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--n_components', type=float, default=100)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    VoxelPCA(args).run()

if __name__ == '__main__':
    main()
