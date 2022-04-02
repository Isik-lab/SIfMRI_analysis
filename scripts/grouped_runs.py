#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

class GroupRuns():
    def __init__(self, args):
        self.sid = str(args.s_num).zfill(2)
        self.process = 'GroupRuns'
        self.set = args.set
        self.data_dir = args.data_dir
        self.real_trials_per_run = 50
        self.runs_per_repeat = 6
        self.out_dir = f'{args.out_dir}/{self.process}/sub-{self.sid}'
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        videos = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        nconds = len(videos)
        
        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-{self.sid}/sub-{self.sid}_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)

        # Initialize an empty array
        arr = np.zeros((n_voxels, nconds, 2))
        files = glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*beta.npy')
        files = [file for file in files if 'crowd' not in file]
        total_repeats = len(files) / self.runs_per_repeat / self.real_trials_per_run
        if self.set == 'test':
            even_denom = total_repeats
            odd_denom = total_repeats
            total_repeats = total_repeats * 2
        else:
            even_denom = np.floor(total_repeats / 2)
            odd_denom = np.ceil(total_repeats / 2)
        for ci, cond in enumerate(videos.video_name):
            print(f'{ci+1}: {cond}')
            cond = cond.split('.mp4')[0]
            # Get all the files for the current condition
            files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*cond-{cond}*beta.npy'))
            for ri, file in enumerate(files):
                if (ri+1) % 2 == 0:
                    arr[..., ci, 0] += np.load(file).flatten()
                else:
                    arr[..., ci, 1] += np.load(file).flatten()

        #even/odd
        odd = arr[..., 0] / odd_denom
        even = arr[..., 1] / even_denom

        #overall average
        arr = (arr[..., 0] + arr[..., 1]) / total_repeats

        # Save the subject data
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-even-data.npy', even)
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-odd-data.npy', odd)
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.set}-data.npy', arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    GroupRuns(args).run()

if __name__ == '__main__':
    main()


