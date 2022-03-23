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
import seaborn as sns
from src.custom_plotting import custom_nilearn_cmap, custom_seaborn_cmap, feature_categories


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
    legend = ax.legend()
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
    model = PCA(whiten=True, n_components=n_components)
    out = model.fit_transform(X)
    return out, model.components_, model.explained_variance_ratio_


class VoxelPCA():
    def __init__(self, args):
        self.process = 'VoxelPCA'
        self.set = args.set
        self.STS = args.STS
        if self.STS:
            self.out_name = 'STS'
        else:
            self.out_name = 'reliable_voxels'
        self.n_subjects = 4
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

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        # df['motion energy'] = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{self.set}_avg.npy')
        # df['AlexNet conv2'] = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{self.set}_avg.npy')
        # df['AlexNet conv5'] = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv5_set-{self.set}_avg')
        df.sort_values(by=['video_name'], inplace=True)
        new = df.drop(columns=['video_name'])
        categories = pd.read_csv(f'{self.data_dir}/annotations/{self.set}_categories.csv')
        return np.array(new.columns), df, categories.action_categories.to_numpy()

    def load_neural(self, mask):
        X = []
        for sid_ in range(self.n_subjects):
            sid = str(sid_ + 1).zfill(2)
            betas = np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_{self.set}-data.npy')

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

    def load_roi_mask(self, n_voxels):
        mask = np.zeros(n_voxels, dtype='bool')
        for sid_ in range(1, self.n_subjects + 1):
            sid_ = str(sid_).zfill(2)
            cur = nib.load(f'{self.data_dir}/ROI_masks/sub-{sid_}/sub-{sid_}_task-SIpSTS_region-SIpSTS_mask.nii.gz')
            cur = np.array(cur.dataobj, dtype='bool').flatten()
            mask += cur
        return mask

    def load_mask(self):
        im = nib.load(f'{self.out_dir}/Reliability/sub-all_stat-rho_statmap.nii.gz')
        if self.STS:
            mask = self.load_roi_mask(np.prod(im.shape))
        else:
            mask = np.load(f'{self.out_dir}/Reliability/sub-all_reliability-mask.npy')
        return mask, im

    def vol_to_surf(self, im, hemi):
        return surface.vol_to_surf(im, surf_mesh=self.fsaverage[f'pial_{hemi}'], radius=2.)

    def plot_brain(self, stat, mask, im):
        cmap = sns.color_palette('Paired', as_cmap=True)
        volume = cp.mkNifti(stat, mask, im)
        texture = {'left': self.vol_to_surf(volume, 'left'),
                   'right': self.vol_to_surf(volume, 'right')}
        cp.plot_surface_stats(self.fsaverage, texture,
                              cmap=cmap, threshold=1,
                              output_file=f'{self.figure_dir}/brain_PCs_set-{self.set}_{self.out_name}.pdf')

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
        plt.savefig(f'{self.figure_dir}/explained_variance_set-{self.set}_{self.out_name}.pdf')

    def PC_to_features(self, features, feature_names, vid_comp):
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
                df = pd.concat([df, pd.DataFrame(d)])
        df.category = pd.Categorical(df.category,
                              categories=['scene', 'object', 'social primitive', 'social'],
                              ordered=True)
        df.to_csv(f'{self.out_dir}/{self.process}/PCs_set-{self.set}_{self.out_name}.csv', index=False)
        return df

    def plot_PC_results(self, df, videos, vid_comp):
        sns.set(style='whitegrid', context='talk')
        for i, iname in enumerate(np.unique(df.PC)):
            _, ax = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1, 1.5]})
            plot_feature_correlation(df[df['PC'] == iname], ax[0])
            plot_video_loadings(vid_comp[:, i], videos, ax[1])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{self.figure_dir}/PC{str(i).zfill(2)}_set-{self.set}_{self.out_name}.pdf')
            plt.close()

    def run(self):
        # Load neural data and do PCA
        mask, im = self.load_mask()
        neural, n_voxels = self.load_neural(mask)
        vid_comp, comp_vox, explained_variance = pca(neural, self.n_components)

        # Plot on the brain
        vox = np.argmax(comp_vox.reshape((-1, 4, n_voxels)).mean(axis=-2), axis=0) + 1
        self.plot_brain(vox, mask, im)

        # Interpret the PCs
        feature_names, features, videos = self.load_features()
        self.plot_variance(explained_variance.cumsum())
        df = self.PC_to_features(features, feature_names, vid_comp)
        self.plot_PC_results(df, videos, vid_comp)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--n_components', type=float, default=10)
    parser.add_argument('--STS', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    VoxelPCA(args).run()

if __name__ == '__main__':
    main()
