#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn import plotting, image, datasets, surface
import nibabel as nib
from src import tools
import seaborn as sns
import src.custom_plotting as cm


class Reliability():
    def __init__(self, args):
        self.process = 'Reliability'
        self.n_subjects = 4
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.set = args.set
        if self.set == 'test':
            self.threshold = 0.279
        else: #self.set == 'train'
            self.threshold = 0.139
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{args.out_dir}/{self.process}'):
            os.mkdir(f'{args.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=args.mesh)

    def rm_EVC(self, mask):
        if self.sid == 'all':
            evc = nib.load(f'{self.data_dir}/ROI_masks/sub-01/sub-01_region-EVC_mask_nooverlap.nii.gz')
        else:
            evc = nib.load(f'{self.data_dir}/ROI_masks/sub-{self.sid}/sub-{self.sid}_region-EVC_mask_nooverlap.nii.gz')
        evc = np.array(evc.dataobj, dtype='bool')
        evc = np.invert(evc)
        mask = mask.reshape(evc.shape)

        # remove EVC
        inds = np.where(np.invert(evc))
        mask[inds] = 0

        return mask.flatten()

    def brain_indices(self, affine, shape):
        mask = datasets.load_mni152_brain_mask()
        mask = image.resample_img(mask, target_affine=affine,
                                  target_shape=shape,
                                  interpolation='nearest')
        mask = np.array(mask.dataobj, dtype='bool').flatten()
        return np.invert(mask)

    def load_test_betas(self, n_voxels, n_conds):
        even = np.zeros((n_voxels, n_conds))
        odd = np.zeros_like(even)
        if self.sid == 'all':
            for i in tqdm(range(1, self.n_subjects+1), total=self.n_subjects):
                i = str(i).zfill(2)

                # Load the data
                arr = np.load(f'{self.out_dir}/GroupRuns/sub-{i}/sub-{i}_test-data.npy')
                print(arr.shape)
                even += arr[..., 1::2].mean(axis=-1)
                odd += arr[..., ::2].mean(axis=-1)

            # Get the average of the even and odd runs across subjects
            even /= self.n_subjects
            odd /= self.n_subjects
        else:
            arr = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_test-data.npy')
            even += arr[..., 1::2].mean(axis=-1)
            odd += arr[..., ::2].mean(axis=-1)
        return even, odd

    def load_betas(self):
        if self.sid == 'all':
            even = None
            for si in tqdm(range(self.n_subjects), total=self.n_subjects):
                sub = str(si+1).zfill(2)
                # Load the data
                seven = np.load(f'{self.out_dir}/GroupRuns/sub-{sub}/sub-{sub}_{self.set}-even-data.npy')
                sodd = np.load(f'{self.out_dir}/GroupRuns/sub-{sub}/sub-{sub}_{self.set}-odd-data.npy')

                if even is None:
                    even = np.zeros((self.n_subjects, seven.shape[0], seven.shape[1]))
                    odd = np.zeros_like(even)

                even[si, ...] = seven
                odd[si, ...] = sodd

            # Get the average of the even and odd runs across subjects
            even = even.mean(axis=0)
            odd = odd.mean(axis=0)
        else:
            even = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_{self.set}-even-data.npy')
            odd = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_{self.set}-odd-data.npy')
        return even, odd

    def run(self):
        videos = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        n_conds = len(videos)

        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-01/sub-01_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)
        affine = im.affine

        print('loading betas...')
        even, odd = self.load_betas(n_voxels, n_conds)

        # Remove signal coming from outside the brain
        brain_mask = self.brain_indices(affine, vol)
        even[:, brain_mask] = 0.
        odd[:, brain_mask] = 0.
        
        # Compute the correlation
        print('computing the correlation')
        r_map = tools.corr2d(even, odd)

        # Make the array into a nifti image and save
        print('saving reliability nifti')
        r_map = np.nan_to_num(r_map)
        r_im = nib.Nifti1Image(np.array(r_map).reshape(vol), affine)
        r_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_set-{self.set}_stat-rho_statmap.nii.gz'
        nib.save(r_im, r_name)

        #Save the numpy array
        print('saving reliability numpy arr')
        r_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_set-{self.set}_stat-rho_statmap.npy'
        np.save(r_name, r_map)

        #Save the mask
        print('saving reliability mask')
        r_mask = np.zeros_like(r_map, dtype='int')
        r_mask[(r_map >= self.threshold) & (~np.isnan(r_map))] = 1
        r_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_set-{self.set}_reliability-mask.npy'
        np.save(r_name, r_mask)

        # remove the EVC and save
        r_mask = self.rm_EVC(r_mask)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_set-{self.set}_reliability-mask_noEVC.npy', r_mask)

        # Plot in the volume
        print('saving figures')
        cmap = sns.color_palette('magma', as_cmap=True)
        plotting.plot_stat_map(r_im, display_mode='ortho',
                               threshold=self.threshold,
                               symmetric_cbar=False,
                               output_file=f'{self.figure_dir}/sub-{self.sid}_view-volume_set-{self.set}_stat-rho_statmap.pdf',
                               cmap=cmap)

        # Plot on the surface
        name = f'{self.figure_dir}/sub-{self.sid}_view-surface_set-{self.set}_stat-rho_statmap.pdf'
        texture = {'left': surface.vol_to_surf(r_im, self.fsaverage['pial_left'],
                                               interpolation='linear'),
                   'right': surface.vol_to_surf(r_im, self.fsaverage['pial_right'],
                                                interpolation='linear')}
        # vmax = cm.get_vmax(texture)
        vmax = 0.8
        cm.plot_surface_stats(self.fsaverage, texture,
                              threshold=self.threshold,
                              modes=['lateral'],
                              cmap=cmap,
                              output_file=name,
                              vmax=vmax)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--mesh', type=str, default='fsaverage5')
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

