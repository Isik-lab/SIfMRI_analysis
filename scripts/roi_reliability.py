#!/usr/bin/env python
# coding: utf-8

from glob import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pickle
import os


class ROIRelibility:
    def __init__(self, args):
        self.process = 'ROIPrediction'
        self.sid = str(args.s_num).zfill(2)
        self.roi = args.roi
        self.step = args.step
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_reliability.pkl'
        self.reliability_file = f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-{self.step}_stat-r_statmap.nii.gz'
        self.reliability_mask_file = f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-{self.step}_reliability-mask.nii.gz'
        print(vars(self))
        print()

    def load_roi_mask(self, reliability_mask):
        # Loop through the two hemispheres
        roi_mask = []
        for hemi in ['lh', 'rh']:
            hemi_mask_file = glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*roi-{self.roi}*hemi-{hemi}*mask.nii.gz')[0]
            hemi_mask = nib.load(hemi_mask_file).get_fdata().astype('bool')
            roi_mask.append(hemi_mask)
        roi_mask = np.sum(roi_mask, axis=0).astype('bool')
        return np.logical_and(roi_mask, reliability_mask)
    
    def load_files(self):
        # Load the reliability file and the reliability mask
        reliability = nib.load(self.reliability_file).get_fdata()
        reliability_mask = nib.load(self.reliability_mask_file).get_fdata().astype('bool')
        roi_mask = self.load_roi_mask(reliability_mask)
        
        # square the values because the reliability map is just the correlation
        out_data = reliability[roi_mask] ** 2 
        return out_data.mean()

    def save_results(self, d):
        f = open(self.out_file_name, "wb")
        pickle.dump(d, f)
        f.close()

    def add_info2data(self, data):
        data['sid'] = self.sid
        data['roi'] = self.roi
        data['category'] = None
        data['feature'] = None
        data['unique_variance'] = False

    def run(self):
        data = dict()
        data['reliability'] = self.load_files()
        self.add_info2data(data)
        self.save_results(data)
        print(f"reliability = {data['reliability']:4f} \n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--roi', type=str, default='EVC')
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    ROIRelibility(args).run()

if __name__ == '__main__':
    main()
