#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from nilearn import datasets, surface, plotting
import matplotlib.pyplot as plt
import nibabel as nib

import tools

class group_reliability():
    def __init__(self, args):
        self.process = 'group_reliability'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{args.out_dir}/{self.process}'):
            os.mkdir(f'{args.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def run(self, threshold=0.279, group='test', n_subjs=4):
        test_videos = pd.read_csv(f'{self.data_dir}/annotations/{group}.csv')
        nconds = len(test_videos)
        
        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-01/sub-01_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)
        affine = im.affine
        
        even = np.zeros((n_voxels, nconds)) 
        odd = np.zeros_like(even)
        print('loading betas...')
        for sid_ in tqdm(range(1, n_subjs+1), total=n_subjs):
            sid = str(sid_).zfill(2)
                         
            # Load the data
            arr = np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_test-data.npy')
            even += arr[..., 1::2].mean(axis=-1)
            odd += arr[..., ::2].mean(axis=-1)
            
        # Get the average of the even and odd runs across subjects
        even /= n_subjs
        odd /= n_subjs
        
        # Compute the correlation
        print('computing the correlation')
        r_map = tools.corr2d(even.T, odd.T)

        # Make the array into a nifti image and save
        print('saving reliability nifti')
        r_im = nib.Nifti1Image(np.array(r_map).reshape(vol), affine)
        r_name = f'{self.out_dir}/{self.process}/sub-all_stat-rho_statmap.nii.gz'
        nib.save(r_im, r_name)

        #Save the numpy array
        print('saving reliability numpy arr')
        r_name = f'{self.out_dir}/{self.process}/sub-all_stat-rho_statmap.npy'
        np.save(r_name, r_map)
        
        #Save the mask
        print('saving reliability mask')
        r_mask = np.zeros_like(r_map, dtype='int')
        r_mask[(r_map >= threshold) & (~np.isnan(r_map))] = 1
        r_name = f'{self.out_dir}/{self.process}/sub-all_reliability-mask.npy'
        np.save(r_name, r_mask)

        # Plot in the volume
        print('saving figures')
        plotting.plot_stat_map(r_im, display_mode='ortho',
                               threshold=threshold,
                               output_file=f'{self.figure_dir}/sub-all_view-volume_stat-rho_statmap.pdf')

        # Plot on the surface
        name = f'{self.figure_dir}/sub-all_view-surface_stat-rho_statmap.pdf'
        plotting.plot_img_on_surf(r_im, inflate=True,
                                  threshold=threshold,
                                  output_file=name,
                                  views=['lateral', 'medial', 'ventral'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    group_reliability(args).run()

if __name__ == '__main__':
    main()

