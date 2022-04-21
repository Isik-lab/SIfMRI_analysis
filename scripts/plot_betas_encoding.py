#!/usr/bin/env python
# coding: utf-8

import argparse
import glob

import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, plotting
from statsmodels.stats.multitest import multipletests
from src import custom_plotting as cp
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
    # remove nan
    indices = np.isnan(rs)
    rs[indices] = 0
    ps[indices] = 1

    # correct
    ps, threshold = correct(ps, rs)
    ps = np.invert(ps)
    indices = np.where(ps)[0]
    rs[indices] = 0.
    rs_mask = np.copy(rs)
    rs_mask[rs != 0.] = 1.
    return rs, rs_mask, threshold


class PlotVoxelEncoding():
    def __init__(self, args):
        self.process = 'PlotVoxelEncoding'
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.roi_parcel = args.roi_parcel
        self.noise_ceiling_set = args.noise_ceiling_set
        self.stat_dir = args.stat_dir
        self.mask_dir = args.mask_dir
        self.annotation_dir = args.annotation_dir
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
        if len(args.feature) == 1:
            self.feature = args.feature[0]
        else:
            self.feature = ''
            for feature in args.feature:
                self.feature += feature.replace(' ', '').capitalize()
        self.predict_individual_features = args.predict_individual_features
        self.model_individual_features = args.model_individual_features
        self.group_features = args.group_features
        self.overall = args.overall
        self.control = args.control
        self.pca_before_regression = False
        if self.overall:
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = 'overall'
            self.threshold = None
        elif self.group_features:
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = f'grouped_prediction/{self.feature}'
            self.threshold = None
        elif self.predict_individual_features:
            assert self.feature is not None, "Must define an input feature"
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = f'predict_individual_features/{self.feature}'
            self.threshold = None
        else: #self.model_individual_features:
            assert self.feature is not None, "Must define an input feature"
            self.cmap = sns.color_palette('magma', as_cmap=True)
            self.out_name = f'model_individual_features/{self.feature}'
            self.threshold = None
        self.figure_dir = f'{args.figure_dir}/{self.process}/{self.control}/{self.out_name}'
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def load_features(self):
        df = pd.read_csv(f'{self.annotation_dir}/annotations.csv')
        df.drop(columns=['video_name'], inplace=True)
        features = np.array(df.columns)
        self.features = np.array([feature.replace(' ', '_') for feature in features])

    def load_noise_ceiling(self, mask):
        noise_ceiling = np.load(f'{self.mask_dir}/sub-{self.sid}_set-{self.noise_ceiling_set}_stat-rho_statmap.npy')
        inds = np.where(noise_ceiling < 0.)[0]
        noise_ceiling[inds] = np.nan
        inds = np.where(mask)[0]
        noise_ceiling = noise_ceiling[inds]
        print(np.nanmin(noise_ceiling))
        return noise_ceiling

    # def preference_maps(self, mask, mask_im):
    #     # name = f'{self.stat_dir}/sub-{self.sid}_model-full_predict-all_control-{self.control}_pca-{self.pca_before_regression}_rs-mask.npy'
    #     # rs = np.load(name).astype('bool')
    #
    #     base = f'{self.stat_dir}/sub-{self.sid}_model-*_predict-all_control-{self.control}_pca-{self.pca_before_regression}_rs.npy'
    #     files = glob.glob(base)
    #     pred = []
    #     for file in files:
    #         arr = np.load(file)
    #         arr = np.expand_dims(arr, axis=1)
    #         if type(pred) is list:
    #             pred = arr
    #         else:
    #             pred = np.hstack([pred, arr])
    #     preference = np.argmax(pred, axis=1)
    #     # Make the argmax indexed at 1
    #     preference += 1
    #
    #     # Make the preference values into a volume mask
    #     preference[~rs] = 0
    #     volume = cp.mkNifti(preference, mask, mask_im, nii=False)
    #     volume = volume.astype('float')
    #     return nib.Nifti1Image(volume.reshape(mask_im.shape), affine=mask_im.affine)

    def grouped_prediction(self, mask, mask_im):
        base = f'{self.stat_dir}/sub-{self.sid}_model-full_predict-{self.feature}_control-{self.control}_pca-{self.pca_before_regression}'
        rs = np.load(f'{base}_rs.npy')
        ps = np.load(f'{base}_ps.npy')

        # Normalize by noise ceiling
        rs = rs / self.load_noise_ceiling(mask)
        rs, rs_mask, threshold = filter_r(rs, ps)
        self.threshold = threshold
        np.save(f'{base}_rs-filtered.npy', rs)
        np.save(f'{base}_rs-mask.npy', rs_mask)
        return cp.mkNifti(rs, mask, mask_im)

    def overall_prediction(self, mask, mask_im):
        base = f'{self.stat_dir}/sub-{self.sid}_model-full_predict-all_control-{self.control}_pca-{self.pca_before_regression}'
        rs = np.load(f'{base}_rs.npy')
        ps = np.load(f'{base}_ps.npy')

        # Normalize by noise ceiling
        rs = rs / self.load_noise_ceiling(mask)
        rs, rs_mask, threshold = filter_r(rs, ps)
        self.threshold = threshold
        np.save(f'{base}_rs-filtered.npy', rs)
        np.save(f'{base}_rs-mask.npy', rs_mask)
        return cp.mkNifti(rs, mask, mask_im)

    def individual_features(self, mask, mask_im):
        if self.predict_individual_features:
            base = f'{self.stat_dir}/sub-{self.sid}_model-full_predict-{self.feature}_control-{self.control}_pca-{self.pca_before_regression}'
        else: #self.model_individual_features
            base = f'{self.stat_dir}/sub-{self.sid}_model-{self.feature}_predict-all_control-{self.control}_pca-{self.pca_before_regression}'

        rs = np.load(f'{base}_rs.npy')
        ps = np.load(f'{base}_ps.npy')

        # Filter the r-values, set threshold, and save output
        rs = rs / self.load_noise_ceiling(mask)
        rs, rs_mask, threshold = filter_r(rs, ps)
        self.threshold = threshold
        return cp.mkNifti(rs, mask, mask_im)

    def load(self):
        mask_im = nib.load(f'{self.mask_dir}/sub-{self.sid}_set-test_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.mask_dir}/sub-{self.sid}_set-test_reliability-mask.npy')
        if self.overall:
            volume = self.overall_prediction(mask, mask_im)
        elif self.group_features:
            volume = self.grouped_prediction(mask, mask_im)
        else:
            volume = self.individual_features(mask, mask_im)
        print(np.nanmax(volume.dataobj))
        # view = plotting.view_img(volume, threshold=0)
        # view.open_in_browser()
        texture = {'left': surface.vol_to_surf(volume, self.fsaverage['pial_left'],
                                               interpolation='nearest'),
                   'right': surface.vol_to_surf(volume, self.fsaverage['pial_right'],
                                                interpolation='nearest')}
        print(np.nanmax(texture['right']))
        return volume, texture

    def run(self):
        # load reliability files
        self.load_features()
        volume, texture = self.load()
        if self.threshold >= 1.:
            vmax = self.threshold + 0.1
        else:
            vmax = 1.
        cp.plot_surface_stats(self.fsaverage, texture,
                              roi=self.roi_parcel,
                              cmap=self.cmap,
                              modes=['lateral', 'ventral'],
                              output_file=f'{self.figure_dir}/sub-{self.sid}.png',
                              threshold=self.threshold,
                              vmax=vmax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--roi_parcel', action='append', default=[])
    parser.add_argument('--control', type=str, default='conv2')
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--noise_ceiling_set', type=str, default='train')
    parser.add_argument('--feature', '-p', action='append', default=[])
    parser.add_argument('--group_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overall', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--predict_individual_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model_individual_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mask_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/Reliability')
    parser.add_argument('--annotation_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw/annotations')
    parser.add_argument('--stat_dir', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim/VoxelPermutation')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotVoxelEncoding(args).run()


if __name__ == '__main__':
    main()
