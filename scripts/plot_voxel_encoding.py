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


def save(arr, out_name, mode='npy'):
    if mode == 'npy':
        np.save(out_name, arr)
    elif mode == 'nii':
        nib.save(arr, out_name)


def mk_cmap(n_features=12):
    # colors = ['blue', 'purple', 'yellow']
    # colors = ['#48D4E1', '#8C55FD', '#FADC00']
    c1 = tuple(np.array([72., 212., 225.]) / 256)
    c2 = tuple(np.array([140., 85., 253.]) / 256)
    c3 = tuple(np.array([250., 220., 0.]) / 256)

    cmap = sns.color_palette('Paired', n_features, as_cmap=True)
    colors = []
    for c in range(n_features):
        # if c <= 1 or c == 4:
        if c <= 4:
            colors.append(c1)
            cmap._lut[c] = list(c1) + [1.]
        elif c > 4 and c <= 6:
        # elif c == 2 or c == 3:
            colors.append(c2)
            cmap._lut[c] = list(c2) + [1.]
        else:
            colors.append(c3)
            cmap._lut[c] = list(c3) + [1.]
    cmap.colors = tuple(colors)

    cmap._lut[cmap._i_over] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_under] = [0., 0., 0., 0.]
    cmap._lut[cmap._i_bad] = [0., 0., 0., 0.]
    return cmap


def correct(ps_, rs_, p_crit=1e-2):
    sig_bool, ps_corrected, _, _ = multipletests(ps_, alpha=p_crit, method='fdr_by')
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
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)
        self.features = []
        self.separate_features = args.separate_features
        self.overall_prediction = args.overall_prediction
        self.no_control_model = args.no_control_model
        self.pca_before_regression = args.pca_before_regression
        if not self.separate_features:
            if self.overall_prediction:
                self.cmap = sns.color_palette('magma', as_cmap=True)
                self.out_name = f'{self.figure_dir}/sub-{self.sid}_overall.png'
                self.threshold = None
            else:
                self.cmap = mk_cmap()
                self.out_name = f'{self.figure_dir}/sub-{self.sid}_grouped.png'
                self.threshold = 1.
        else:
            self.cmap = sns.color_palette('Paired', as_cmap=True)
            self.out_name = f'{self.figure_dir}/sub-{self.sid}_separate.png'
            self.threshold = 1.
        print(self.out_name)

    def load_features(self):
        df = pd.read_csv(f'{self.annotation_dir}/annotations.csv')
        train = pd.read_csv(f'{self.annotation_dir}/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name'], inplace=True)
        features = np.array(df.columns)
        self.features = np.array([feature.replace(' ', '_') for feature in features])

    def load(self):
        mask_im = nib.load(f'{self.mask_dir}/sub-all_stat-rho_statmap.nii.gz')
        mask = np.load(f'{self.mask_dir}/sub-all_reliability-mask.npy')

        if not self.overall_prediction:
            rs = np.load(f'{self.stat_dir}/sub-all/sub-all_feature-all_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_rs-filtered.npy').astype('bool')
            rs = cm.mkNifti(rs, mask, mask_im, nii=False)

            base = f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-XXX_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_rs.npy'
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
            volume = nib.Nifti1Image(volume.reshape(mask_im.shape), affine=mask_im.affine)
        else:
            rs = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-all_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_rs.npy')
            ps = np.load(f'{self.stat_dir}/sub-{self.sid}/sub-{self.sid}_feature-all_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_ps.npy')

            #Filter the r-values, set threshold, and save output
            rs, rs_mask, threshold = filter_r(rs, ps)
            self.threshold = threshold
            np.save(f'{self.stat_dir}/sub-{self.sid}/sub-all_feature-all_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_rs-filtered.npy', rs)
            np.save(f'{self.stat_dir}/sub-{self.sid}/sub-all_feature-all_include_control-{self.no_control_model}_pca_before_regression-{self.pca_before_regression}_rs-mask.npy', rs_mask)
            volume = cm.mkNifti(rs, mask, mask_im)

        texture = {'left': self.vol_to_surf(volume, 'left'),
                   'right': self.vol_to_surf(volume, 'right')}
        return volume, texture

    def run(self):
        # load reliability files
        self.load_features()
        volume, texture = self.load()
        cm.plot_surface_stats(self.fsaverage, texture,
                              cmap=self.cmap,
                              output_file=self.out_name,
                              threshold=self.threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--mesh', type=str, default='fsaverage5')
    parser.add_argument('--separate_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overall_prediction', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--no_control_model', action=argparse.BooleanOptionalAction, default=True)
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
