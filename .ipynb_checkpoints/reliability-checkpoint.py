import argparse
import os
import glob
import numpy as np
import pandas as pd

from nilearn import surface, datasets, plotting
import matplotlib.pyplot as plt
import nibabel as nib

def corr2d(a, b):
    a_m = a - a.mean(axis=0)
    b_m = b - b.mean(axis=0)

    r = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        r[i] = (a_m[i, :] @ b_m[i, :]) / (np.sqrt((a_m[i, :] @ a_m[i, :]) * (b_m[i, :] @ b_m[i, :])))
    return r

class reliability():
    def __init__(self, args):
        self.process = 'reliability'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
            
            
    def run(self):
        test_videos = pd.read_csv(f'{self.data_dir}/annotations/test.csv')
        nconds = len(test_videos)
        
        # Load an ROI file to get meta data about the images
        im = nib.load(f'{self.data_dir}/ROI_masks/sub-01/sub-01_region-EVC_mask.nii.gz')
        vol = im.shape
        n_voxels = np.prod(vol)
        affine = im.affine
        
        for i in range(1,5):
            sid = str(i).zfill(2)
        
            files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{sid}/*beta.npy'))

            # Find the test runs
            runs = []
            for f in files:
                if test_videos.iloc[0,0].split('.mp4')[0] in f:
                    runs.append(f.split('run-')[-1].split('_')[0])

            # Save info about number of runs and conditions
            nruns = len(runs)
            half = int(nruns/2)

            for ri, run in enumerate(runs):
                # Get all the files for the current run
                files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*run-{run}*beta.npy'))
                # Initialize an empty array for the current run
                arr = np.zeros((n_voxels, nconds, nruns))
                fi = 0
                # Append all conditions to the current array, except for the crowd condition
                for f in files:
                    if not 'crowd' in f:
                        arr[..., fi, ri] = np.load(f).flatten()
                        fi += 1
            
            # Save the subject data
            np.save(f'{self.out_dir}/sub-{sid}_test-data.npy', arr)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    reliability(args).run()

if __name__ == '__main__':
    main()

