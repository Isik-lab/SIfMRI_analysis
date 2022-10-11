#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from src import tools
import pickle


def mask_img(img, mask):
    if type(img) is str:
        img = nib.load(img)
    if type(mask) is str:
        mask = nib.load(mask)

    arr = np.array(img.dataobj)
    mask = np.array(mask.dataobj, dtype=bool)
    return arr[mask]


class ROIPrediction:
    def __init__(self, args):
        self.process = 'ROIPrediction'
        self.sid = str(args.s_num).zfill(2)
        self.roi = args.roi
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_reliability.pkl'
        print(vars(self))

    def get_file_name(self):
        top = f'{self.out_dir}/Reliability'
        file_name = f'{top}/sub-{self.sid}_space-T1w_desc-test-fracridge_stat-r_statmap.nii.gz'
        return file_name

    def load_files(self):
        out_data = None
        for hemi in ['lh', 'rh']:
            roi_mask = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*roi-{self.roi}*hemi-{hemi}*mask.nii.gz')[0]
            one_hemi = mask_img(self.get_file_name(), roi_mask)
            if out_data is None:
                out_data = one_hemi ** 2 # square the values because the reliability map is just the correlation
            else:
                out_data = np.concatenate([out_data, one_hemi ** 2])
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
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    ROIPrediction(args).run()

if __name__ == '__main__':
    main()
