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
        affine = im.affine
        
        files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*beta.npy'))

        # Find the test runs
        runs = []
        for f in files:
            if videos.iloc[0,0].split('.mp4')[0] in f:
                runs.append(f.split('run-')[-1].split('_')[0])

        # Save info about number of runs and conditions
        nruns = len(runs)
        half = int(nruns/2)
        
        # Initialize an empty array
        arr = np.zeros((n_voxels, nconds, nruns))
        for ri, run in enumerate(runs):
            # Get all the files for the current run
            files = sorted(glob.glob(f'{self.data_dir}/betas/sub-{self.sid}/*run-{run}*beta.npy'))
            # Append all conditions to the current array, except for the crowd condition
            fi = 0
            for f in files:
                if not 'crowd' in f:
                    arr[..., fi, ri] = np.load(f).flatten()
                    fi += 1

        # Save the subject data
        np.save(f'{self.out_dir}/sub-{self.sid}_{self.run_type}-data.npy', arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--run_type', '-r', type=str, default='test')
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    args = parser.parse_args()
    grouped_runs(args).run()

if __name__ == '__main__':
    main()

