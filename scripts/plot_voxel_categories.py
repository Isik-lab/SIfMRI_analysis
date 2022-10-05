#!/usr/bin/env python
# coding: utf-8

import glob
import os
import numpy as np
import pandas as pd
from nilearn import plotting, surface, image
from pathlib import Path
import argparse
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt


def roi2contrast(roi):
    d = dict()
    d['MT'] = 'motionVsStatic'
    d['face-pSTS'] = 'facesVsObjects'
    d['EBA'] = 'bodiesVsObjecs'
    d['PPA'] = 'scenesVsObjects'
    d['TPJ'] = 'beliefVsPhoto'
    d['SI-pSTS'] = 'interactVsNoninteract'
    d['EVC'] = 'EVC'
    return d[roi]


def model2vmax(model):
    d = {'arousal': 0.15,
         'communication': 0.1,
         'agent_distance': 0.075,
         'transitivity': 0.075}
    return d[model]


def roi_cmap():
    d = {
        'MT': (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
        'EBA': (1.0, 1.0, 0.0),
        'face-pSTS': (0.0, 0.8431372549019608, 1.0),
        'SI-pSTS': (1.0, 0.7686274509803922, 0.0)}
    return list(d.values())


class PlotVoxelCategory:
    def __init__(self, args):
        self.process = 'PlotVoxelCategory'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.ROIs = args.ROIs
        self.overwrite = args.overwrite
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.cmap = sns.color_palette('magma', as_cmap=True)
        self.rois = ['MT', 'EBA', 'face-pSTS', 'SI-pSTS']
        self.roi_cmap = roi_cmap()
        im = nib.load(
            f'{self.data_dir}/betas_3mm_zscore/sub-{self.sid}/sub-{self.sid}_space-T1w_desc-test-fracridge_data.nii.gz')
        self.im_shape = im.shape[:-1]
        self.affine = im.affine
        self.header = im.header
        del im

    def compute_surf_stats(self, hemi_, stat_):
        file = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}-stat-{stat_}_hemi-{hemi_}.mgz'
        if self.overwrite or not os.path.exists(file):
            cmd = '/Applications/freesurfer/bin/mri_vol2surf '
            cmd += f'--src {self.out_dir}/CategoryVoxelPermutation/sub-{self.sid}_category-{self.category}_{stat_}.nii.gz '
            cmd += f'--out {file} '
            cmd += f'--regheader sub-{self.sid} '
            cmd += f'--hemi {hemi_} '
            cmd += '--projfrac 1'
            os.system(cmd)
        return surface.load_surf_data(file)

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated', \
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi_, stat_):
        if hemi_ == 'lh':
            hemi_name = 'left'
        else:
            hemi_name = 'right'

        # vmax = np.nanmax(surf_map)-0.05
        _, axes = plt.subplots(3, figsize=(5, 15), subplot_kw={'projection': '3d'})
        for ax, view in zip(axes, ['lateral', 'ventral', 'medial']):
            plotting.plot_surf_roi(surf_mesh=surf_mesh,
                                   roi_map=surf_map,
                                   bg_map=bg_map,
                                   vmax=0.4,
                                   vmin=0.,
                                   axes=ax,
                                   cmap=self.cmap,
                                   hemi=hemi_name,
                                   colorbar=True,
                                   view=view)
        plt.savefig(f'{self.figure_dir}/sub-{self.sid}_category-{self.category}_stat-{stat_}_hemi-{hemi_}.jpg')

    def plot_one_hemi(self, hemi_, stat_):
        surface_data = self.compute_surf_stats(hemi_, stat_)
        inflated, sulcus = self.load_surf_mesh(hemi_)
        self.plot_stats(inflated, sulcus, surface_data, hemi_, stat_)

    def nib_transform(self, r_, nii=True):
        unmask = np.load(
            f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.npy').astype(
            'bool')
        i = np.where(unmask)
        if r_.ndim < 2:
            r_unmasked = np.zeros(unmask.shape)
            r_unmasked[i] = r_
            r_unmasked = r_unmasked.reshape(self.im_shape)
        else:
            r_ = r_.T
            r_unmasked = np.zeros((unmask.shape + (r_.shape[-1],)))
            r_unmasked[i, ...] = r_
            r_unmasked = r_unmasked.reshape((self.im_shape + (r_.shape[-1],)))
            print(r_unmasked.shape)

        if nii:
            r_unmasked = nib.Nifti1Image(r_unmasked, self.affine, self.header)
        return r_unmasked

    def run(self):
        for hemi in ['lh', 'rh']:
            for stat in ['r2', 'r2filtered']:
                self.plot_one_hemi(hemi, stat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default=1)
    parser.add_argument('--category', type=str, default='affective')
    parser.add_argument('--ROIs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    PlotVoxelCategory(args).run()


if __name__ == '__main__':
    main()
