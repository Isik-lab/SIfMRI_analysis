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


class Reliability():
    def __init__(self, args):
        self.process = 'Reliability'
        self.sid = str(int(args.s_num)).zfill(2)
        self.set = args.set
        self.space = args.space
        self.step = args.step
        if self.set == 'test':
            self.threshold = 0.235
        else:  # self.set == 'train'
            self.threshold = 0.117
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.cmap = sns.color_palette('magma', as_cmap=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{args.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def load_betas(self, name):
        im = nib.load(
            f'{self.data_dir}/betas/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}-{name}_data.nii.gz')
        return np.array(im.dataobj).reshape((-1, im.shape[-1])).T, im.shape[:-1], im.affine, im.header

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

    def compute_surf_stats(self, filename, hemi):
        cmd = '/Applications/freesurfer/bin/mri_vol2surf '
        cmd += f'--src {filename}.nii.gz '
        cmd += f'--out {filename}_hemi-{hemi}.mgz '
        cmd += f'--regheader sub-{self.sid} '
        cmd += f'--hemi {hemi} '
        cmd += '--projfrac 1'
        os.system(cmd)
        return surface.load_surf_data(f'{filename}_hemi-{hemi}.mgz')

    def load_surf_mesh(self, hemi):
        return f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.inflated', \
               f'{self.data_dir}/freesurfer/sub-{self.sid}/surf/{hemi}.sulc'

    def plot_stats(self, surf_mesh, bg_map, surf_map, hemi):
        file = f'{self.figure_dir}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_hemi-{hemi}.pdf'
        if hemi == 'rh':
            hemi = 'right'
        else:
            hemi = 'left'
        plotting.plot_surf_roi(surf_mesh=surf_mesh,
                               roi_map=surf_map,
                               bg_map=bg_map,
                               threshold=self.threshold,
                               vmax=1.,
                               cmap=self.cmap,
                               hemi=hemi,
                               view='lateral',
                               output_file=file)

    def plot_one_hemi(self, filename, hemi):
        surface_data = self.compute_surf_stats(filename, hemi)
        inflated, sulcus = self.load_surf_mesh(hemi)
        self.plot_stats(inflated, sulcus, surface_data, hemi)

    def run(self):
        print('loading betas...')
        even, shape, affine, header = self.load_betas('even')
        odd, _, _, _ = self.load_betas('odd')

        # Compute the correlation
        print('computing the correlation')
        r_map = tools.corr2d(even, odd)
        r_map[r_map < 0] = 0  # Filter out the negative values

        # Make the array into a nifti image and save
        print('saving reliability nifti')
        r_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_stat-r_statmap'
        r_im = nib.Nifti1Image(np.nan_to_num(r_map).reshape(shape), affine, header)
        nib.save(r_im, f'{r_name}.nii.gz')
        np.save(f'{r_name}.npy', r_map)

        # Save the mask
        print('saving reliability mask')
        r_mask = np.zeros_like(r_map, dtype='int')
        r_mask[(r_map >= self.threshold) & (~np.isnan(r_map))] = 1
        mask_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_reliability-mask.npy'
        np.save(mask_name, r_mask)

        # Plot in the volume
        print('saving figures')
        anatomy = self.load_anatomy()
        figure_name = f'{self.figure_dir}/sub-{self.sid}_space-{self.space}_desc-{self.set}-{self.step}_reliability.png'
        plotting.plot_stat_map(r_im, anatomy,
                               symmetric_cbar=False,
                               threshold=self.threshold,
                               display_mode='mosaic',
                               cmap=sns.color_palette('magma', as_cmap=True),
                               output_file=figure_name)

        # Plot on the surface
        print('saving surface figures')
        for hemi in ['lh', 'rh']:
            self.plot_one_hemi(r_name, hemi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--step', type=str, default='fracridge')
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
