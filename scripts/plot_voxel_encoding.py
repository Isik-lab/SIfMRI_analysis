#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from statsmodels.stats.multitest import multipletests
from src import custom_plotting as cm
from nilearn import surface
from pathlib import Path


def save(arr, out_name, mode='npy'):
    if mode == 'npy':
        np.save(out_name, arr)
    elif mode == 'nii':
        nib.save(arr, out_name)


def correct(ps_, rs_, p_crit=5e-2):
    sig_bool, ps_corrected, _, _ = multipletests(ps_, alpha=p_crit, method='fdr_bh')
    indices = np.where(sig_bool)[0]
    return sig_bool, rs_[indices].min()


def filter_r(rs, ps):
    ps, threshold = correct(ps, rs)
    ps = np.invert(ps)
    indices = np.where(ps)[0]
    rs[indices] = 0.
    rs_mask = np.copy(rs)
    rs_mask[rs != 0.] = 1.
    return rs, rs_mask, threshold


class PlotEncoding():
    def __init__(self, args):
        self.process = 'PlotEncoding'
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.stat_dir = args.stat_dir
        self.mask_dir = args.mask_dir
        self.annotation_dir = args.annotation_dir
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
        self.features = []
        self.feature = args.feature
        self.separate_features = args.separate_features
        self.individual_features = args.individual_features
        self.group_features = args.group_features
        self.overall = args.overall
        self.control = args.control
        self.pca_before_regression = args.pca_before_regression
        if self.overall:
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = 'overall'
            self.threshold = None
        elif self.group_features:
            self.cmap = cm.custom_nilearn_cmap()
            self.out_name = 'grouped'
            self.threshold = 1.
        elif self.separate_features:
            self.cmap = sns.color_palette('Paired', as_cmap=True)
            self.out_name = 'separate'
            self.threshold = 1.
        elif self.individual_features:
            assert self.feature is not None, "Must define an input feature"
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = f'individual_features/{self.feature}'
            self.threshold = None
        else:
            raise RuntimeError ("One of overall, group_features, separate_features, or individual_features must be True")
        print(self.out_name)
        self.figure_dir = f'{args.figure_dir}/{self.process}/{self.control}/{self.out_name}'
        path = Path(self.figure_dir)
        path.mkdir(parents=True, exist_ok=True)

    def load_features(self):
        df = pd.read_csv(f'{self.annotation_dir}/annotations.csv')
        train = pd.read_csv(f'{self.annotation_dir}/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name'], inplace=True)
        features = np.array(df.columns)
        self.features = np.array([feature.replace(' ', '_') for feature in features])

    def preference_maps(self, mask, mask_im):
        rs = np.load(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs-filtered.npy').astype(
            'bool')
        rs = cm.mkNifti(rs, mask, mask_im, nii=False)

        base = f'{self.stat_dir}/sub-{self.sid}_feature-XXX_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs.npy'
        pred = []
        for feature in self.features:
            arr = np.load(base.replace('XXX', feature))
            arr = np.expand_dims(arr, axis=1)
            if type(pred) is list:
                pred = arr
            else:
                pred = np.hstack([pred, arr])
        preference = np.argmax(pred, axis=1)
        # Make the argmax indexed at 1
        preference += 1

        # Make the preference values into a volume mask
        volume = cm.mkNifti(preference, mask, mask_im, nii=False)
        volume[~rs] = 0
        volume = volume.astype('float')
        return nib.Nifti1Image(volume.reshape(mask_im.shape), affine=mask_im.affine)

    def overall_prediction(self, mask, mask_im):
        rs = np.load(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs.npy')
        ps = np.load(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_ps.npy')

        # Filter the r-values, set threshold, and save output
        rs, rs_mask, threshold = filter_r(rs, ps)
        self.threshold = threshold
        np.save(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs-filtered.npy',
            rs)
        np.save(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs-mask.npy',
            rs_mask)
        return cm.mkNifti(rs, mask, mask_im)

    def individual_feature_prediction(self, mask, mask_im):
        rs = np.load(
            f'{self.stat_dir}/sub-{self.sid}_feature-{self.feature}_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs.npy')
        ps = np.load(
            f'{self.stat_dir}/sub-{self.sid}_feature-{self.feature}_control-{self.control}_pca_before_regression-{self.pca_before_regression}_ps.npy')

        # Filter the r-values, set threshold, and save output
        rs, rs_mask, threshold = filter_r(rs, ps)
        self.threshold = threshold
        np.save(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs-filtered.npy',
            rs)
        np.save(
            f'{self.stat_dir}/sub-{self.sid}_feature-all_control-{self.control}_pca_before_regression-{self.pca_before_regression}_rs-mask.npy',
            rs_mask)
        return cm.mkNifti(rs, mask, mask_im)

    def load(self):
        mask_im = nib.load(f'{self.mask_dir}/sub-all_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.mask_dir}/sub-all_reliability-mask.npy')
        if self.overall:
            volume = self.overall_prediction(mask, mask_im)
        elif self.separate_features or self.group_features:
            volume = self.preference_maps(mask, mask_im)
        else:
            volume = self.individual_feature_prediction(mask, mask_im)
        texture = {'left': surface.vol_to_surf(volume, self.fsaverage['pial_left'],
                                               interpolation='nearest'),
                   'right': surface.vol_to_surf(volume, self.fsaverage['pial_right'],
                                                interpolation='nearest')}
        return volume, texture

    def run(self):
        # load reliability files
        self.load_features()
        volume, texture = self.load()
        if self.overall or self.individual_features:
            vmax = cm.get_vmax(texture)
            if self.threshold >= vmax:
                vmax = self.threshold + 0.1
        else:
            vmax = None
        cm.plot_surface_stats(self.fsaverage, texture,
                              cmap=self.cmap,
                              output_file=f'{self.figure_dir}/sub-{self.sid}.png',
                              threshold=self.threshold,
                              vmax=vmax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--control', type=str, default='conv2')
    parser.add_argument('--feature', type=str, default='None')
    parser.add_argument('--separate_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--group_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overall', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--individual_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mask_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/Reliability')
    parser.add_argument('--annotation_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw/annotations')
    parser.add_argument('--stat_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotEncoding(args).run()


if __name__ == '__main__':
    main()
