#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import shutil

import nilearn.image
import numpy as np
import pandas as pd
from pathlib import Path

from src import custom_plotting as cp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nilearn import datasets, surface
import nibabel as nib
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from src.custom_plotting import custom_nilearn_cmap, custom_seaborn_cmap, \
    feature_categories, custom_pca_cmap, roi_paths


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
    ax.set_ylim([-.5, 0.5])
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
    model = PCA(whiten=True, n_components=n_components, random_state=0)
    out = model.fit_transform(X)
    return out, model.components_, model.explained_variance_ratio_


class VoxelPCA():
    def __init__(self, args):
        self.process = 'VoxelPCA'
        self.set = args.set
        self.roi = args.roi
        self.parcel = args.parcel
        assert not np.all([self.roi, self.parcel]), "Either an ROI or a parcel can be defined, not both"
        if self.roi is not None:
            self.out_name = self.roi
        elif self.parcel is not None:
            self.out_name = self.parcel
        else:
            self.out_name = 'reliable_voxels'
        self.n_subjects = 4
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}/{self.out_name}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        self.n_components = args.n_components
        if self.n_components > 1:
            self.n_components = int(self.n_components)
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df['motion energy'] = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{self.set}_avg.npy')
        df['AlexNet conv2'] = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{self.set}_avg.npy')
        df.sort_values(by=['video_name'], inplace=True)
        new = df.drop(columns=['video_name'])
        categories = pd.read_csv(f'{self.data_dir}/annotations/{self.set}_categories.csv')
        return np.array(new.columns), df, categories.action_categories.to_numpy()

    def load_neural(self, mask):
        X = []
        for sid_ in range(self.n_subjects):
            sid = str(sid_ + 1).zfill(2)
            betas = np.load(f'{self.out_dir}/GroupRuns/sub-{sid}/sub-{sid}_{self.set}-data.npy')

            # Filter the beta values to the reliable voxels or to the roi within subject
            if self.roi is not None:
                betas = betas[mask[f'sub-{sid}'], :]
            else:
                betas = betas[mask, :]

            # Mean center the activation within subject
            offset_subject = betas.mean()
            betas -= offset_subject

            if type(X) is list:
                X = betas.T
            else:
                X = np.hstack([X, betas.T])
        return StandardScaler().fit_transform(X)

    def load_roi_mask(self):
        roi_mask = dict()
        n_voxels = dict()
        for sid_ in range(1, self.n_subjects + 1):
            sid = str(sid_).zfill(2)
            files = glob.glob(f'{self.data_dir}/ROI_masks/sub-{sid}/sub-{sid}_*{self.roi}*nii.gz')
            if len(files) > 1:
                for f in files:
                    if 'nooverlap' in f:
                        file = f
            else:
                file = files[0]
            cur = np.array(nib.load(file).dataobj, dtype='bool').flatten()
            roi_mask[f'sub-{sid}'] = cur
            n_voxels[f'sub-{sid}'] = np.sum(cur)
        return roi_mask, n_voxels

    def load_parcel_mask(self, im):
        arr = []
        for hemi in ['left', 'right']:
            paths = roi_paths(hemi)
            vol = nib.load(paths[self.parcel])
            vol = nilearn.image.resample_to_img(vol, im, interpolation='nearest')
            if type(arr) is list:
                arr = np.array(vol.dataobj)
            else:
                arr += np.array(vol.dataobj)
        return arr.astype('bool').flatten()

    def load_mask(self):
        im = nib.load(f'{self.out_dir}/Reliability/sub-all_set-test_stat-rho_statmap.nii.gz')
        n_voxels = None
        if self.roi is not None:
            mask, n_voxels = self.load_roi_mask()
        elif self.parcel is not None:
            mask = self.load_parcel_mask(im)
        else:
            mask = np.load(f'{self.out_dir}/Reliability/sub-all_set-test_reliability-mask.npy').astype('bool')
        return mask, n_voxels, im

    def vol_to_surf(self, im, hemi):
        return surface.vol_to_surf(im, surf_mesh=self.fsaverage[f'pial_{hemi}'], radius=2.)

    def plot_brain(self, stat, mask, im, sid):
        # cmap = sns.color_palette('Paired', as_cmap=True)
        cmap = custom_pca_cmap(self.n_components)
        volume = cp.mkNifti(stat, mask, im)
        texture = {'left': self.vol_to_surf(volume, 'left'),
                   'right': self.vol_to_surf(volume, 'right')}
        if self.parcel is not None:
            hemis = ['left', 'right']
            modes = ['lateral']
        else:
            hemis = ['left', 'right']
            modes = ['lateral', 'ventral']
        cp.plot_surface_stats(self.fsaverage, texture,
                              cmap=cmap, threshold=1,
                              hemis=hemis,
                              modes=modes,
                              output_file=f'{self.figure_dir}/sub-{sid}_set-{self.set}_PCs.pdf')

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
                d['Explained variance'] = explained_variance[iPC]
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
            plt.suptitle(f'PC {i+1} \n Explained variance = {np.round(ev*100):.0f}', fontsize=20)
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
        # Load neural data and do PCA
        mask, n_voxels, im = self.load_mask()
        neural = self.load_neural(mask)
        vid_comp, comp_vox, explained_variance = pca(neural, self.n_components)

        # Plot on the brain
        if self.roi is None:
            vox = np.argmax(comp_vox.reshape((-1, 4, np.sum(mask))).mean(axis=-2), axis=0) + 1
            self.plot_brain(vox, mask, im, 'all')

            sub_vox = comp_vox.reshape((-1, 4, np.sum(mask)))
            for i in range(self.n_subjects):
                vox = np.argmax(sub_vox[:, i, :], axis=0) + 1
                self.plot_brain(vox, mask, im, str(i+1).zfill(2))

        # Interpret the PCs
        feature_names, features, videos = self.load_features()
        self.videos(vid_comp, features)
        self.plot_variance(explained_variance.cumsum())
        df = self.PC_to_features(features, feature_names, vid_comp, explained_variance)
        self.plot_PC_results(df, videos, vid_comp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--parcel', type=str, default=None)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--n_components', type=float, default=10)
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
