#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import numpy as np
import pandas as pd

import nibabel as nib

class grouped_runs():
    def __init__(self, args):
        self.sid = sid = str(args.s_num).zfill(2)
        self.process = 'grouped_runs'
        self.run_type = args.run_type
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}/sub-{self.sid}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        videos = pd.read_csv(f'{self.data_dir}/annotations/{self.run_type}.csv')
        nconds = len(videos)
        
        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-{self.sid}/sub-{self.sid}_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)
        
        # Save info about number of runs and conditions
        files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*beta.npy'))
        if self.run_type == 'test':
            nruns = int((len(files) / 6) * 2)
            arr = np.zeros((n_voxels, nconds, nruns))
        else:
            nruns = int(len(files) / 6)
            arr = np.zeros((n_voxels, nconds))
        
        # Initialize an empty array
        
        for ci, cond in enumerate(videos.video_name):
            print(f'{ci}: {cond}')
            cond = cond.split('.mp4')[0]
            # Get all the files for the current condition
            files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*cond-{cond}*beta.npy'))
            for ri, file in enumerate(files):
                if self.run_type == 'test': 
                    arr[..., ci, ri] = np.load(file).flatten()
                else:
                    arr[..., ci] += np.load(file).flatten()
                    
        # Save the subject data
        if self.run_type =='train':
                arr[..., ci] /= nruns
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.run_type}-data.npy', arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--run_type', '-r', type=str)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    args = parser.parse_args()
    grouped_runs(args).run()

if __name__ == '__main__':
    main()


