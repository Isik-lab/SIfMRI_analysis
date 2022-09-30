#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from nilearn import plotting, surface
import nibabel as nib
import seaborn as sns
import src.tools as tools
from pathlib import Path
import matplotlib.pyplot as plt


def save_np_and_nib(np_arr, shape, affine, header, out):
    np.save(f'{out}.npy', np_arr)
    im = nib.Nifti1Image(np.nan_to_num(np_arr).reshape(shape), affine, header)
    nib.save(im, f'{out}.nii.gz')


class Reliability():
    def __init__(self, args):
        self.process = 'Reliability'
        self.sid = str(int(args.s_num)).zfill(2)
        self.set = args.set
        self.space = args.space
        self.step = args.step
        self.zscore_ses = args.zscore_ses
        self.smooth = args.smooth
        if self.smooth:
            if self.zscore_ses:
                self.beta_path = 'betas_3mm_zscore'
            else: #self.smoothing and not self.zscore_ses:
                self.beta_path  = 'betas_3mm_nozscore'
        else:
            if self.zscore_ses:
                self.beta_path  = 'betas_0mm_zscore'
            else: #not self.smoothing and not self.zscore_ses
                self.beta_path  = 'betas_0mm_nozscore'
        if self.set == 'test':
            self.threshold = 0.235
        else:  # self.set == 'train'
            self.threshold = 0.117
        self.precomputed = args.precomputed
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.cmap = sns.color_palette('magma', as_cmap=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.output_file = f'{self.out_dir}/{self.process}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_stat-r_statmap'
        print(vars(self))

    def load_even_or_odd(self, name):
        im = nib.load(
            f'{self.data_dir}/{self.beta_path}/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}-{name}_data.nii.gz')
        return np.array(im.dataobj).reshape((-1, im.shape[-1])).T, im.shape[:-1], im.affine, im.header

    def load_betas(self):
        print('loading betas...')
        even, shape, affine, header = self.load_even_or_odd('even')
        odd, _, _, _ = self.load_even_or_odd('odd')
        return even, odd, shape, affine, header

    def load_anatomy(self):
        if self.space == 'T1w':
            anat = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-preproc_T1w.nii.gz')
            brain_mask = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-brain_mask.nii.gz')
        else:
            anat = nib.load(
                f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-preproc_T1w.nii.gz')
            brain_mask = nib.load(
                f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-brain_mask.nii.gz')
        return tools.mask_img(anat, brain_mask)

    def vol2surf(self, filename, hemi):
        cmd = '/Applications/freesurfer/bin/mri_vol2surf '
        cmd += f'--src {filename}.nii.gz '
        cmd += f'--out {filename}_hemi-{hemi}.mgz '
        cmd += f'--regheader sub-{self.sid} '
        cmd += f'--hemi {hemi} '
        cmd += '--projfrac 1'
        os.system(cmd)
        return

    def compute_reliability_mask(self, r_map, shape, affine, header, mask_name):
        r_mask = np.zeros_like(r_map, dtype='int')
        r_mask[(r_map >= self.threshold) & (~np.isnan(r_map))] = 1
        save_np_and_nib(r_mask, shape, affine, header, mask_name)

    def compute_reliability(self):
        even, odd, shape, affine, header = self.load_betas()

        # Compute the correlation
        print('computing the correlation')
        r_map = tools.corr2d(even, odd)
        r_map[r_map < 0] = 0  # Filter out the negative values
        save_np_and_nib(r_map, shape, affine, header, self.output_file)

        # Reiliability mask
        mask_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_reliability-mask'
        self.compute_reliability_mask(r_map, shape, affine, header, mask_name)

        # Transform the volume to the surface
        self.vol2surf(self.output_file, 'lh')
        self.vol2surf(self.output_file, 'rh')

    def load_surf_data(self, filename, hemi):
        return surface.load_surf_data(f'{filename}_hemi-{hemi}.mgz')

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated', \
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi):
        file = f'{self.figure_dir}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_hemi-{hemi}.jpg'
        if hemi == 'rh':
            hemi = 'right'
        else:
            hemi = 'left'
        _, ax = plt.subplots(1, figsize=(10, 10),
                               subplot_kw={'projection': '3d'})
        plotting.plot_surf_roi(surf_mesh=surf_mesh,
                               roi_map=surf_map,
                               bg_map=bg_map,
                               ax=ax,
                               threshold=self.threshold,
                               vmax=1.,
                               cmap=self.cmap,
                               hemi=hemi,
                               view='lateral',
                               colorbar=True,
                               output_file=file)

    def plot_each_hemi(self, filename):
        for hemi in ['lh', 'rh']:
            surface_data = self.load_surf_data(filename, hemi)
            inflated, sulcus = self.load_surf_mesh(hemi)
            self.plot_stats(inflated, sulcus, surface_data, hemi)

    def run(self):
        if not self.precomputed:
            print('computing correlation')
            self.compute_reliability()
        self.plot_each_hemi(self.output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='1')
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--precomputed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--zscore_ses', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--smooth', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    Reliability(args).run()


if __name__ == '__main__':
    main()
