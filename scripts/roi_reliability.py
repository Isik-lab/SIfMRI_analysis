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


def roi2contrast(roi):
    d = dict()
    d['MT'] = 'motionVsStatic'
    d['face-pSTS'] = 'facesVsObjects'
    d['EBA'] = 'bodiesVsObjecs'
    d['PPA'] = 'scenesVsObjects'
    d['TPJ'] = 'beliefVsPhoto'
    d['SI-pSTS'] = 'interactVsNoninteract'
    d['EVC'] = 'EVC'
    return d[roi]


class ROIPrediction:
    def __init__(self, args):
        self.process = 'ROIPrediction'
        self.sid = str(args.s_num).zfill(2)
        self.hemi = args.hemi
        self.roi = args.roi
        self.contrast = roi2contrast(self.roi)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.roi_mask = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*{self.contrast}*{self.hemi}*mask.nii.gz')[0]
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_hemi-{self.hemi}_reliability.pkl'
        print(vars(self))

    def get_file_name(self, var):
        top = f'{self.out_dir}/Reliability'
        file_name = f'{top}/sub-{self.sid}_space-T1w_desc-test-fracridge_stat-r_statmap.nii.gz'
        return file_name

    def load_files(self):
        data = dict()
        for key in ['r2']:
            file = mask_img(self.get_file_name(key), self.roi_mask) # Mask the reliability to the roi
            masked_file = file ** 2 # square the values because the reliability map is just the correlation
            data[key] = masked_file.mean(axis=0)
            print(f'loaded {key}')
        return data

    def save_results(self, d):
        f = open(self.out_file_name, "wb")
        pickle.dump(d, f)
        f.close()

    def add_info2data(self, d):
        d['sid'] = self.sid
        d['hemi'] = self.hemi
        d['roi'] = self.roi
        return d

    def run(self):
        data = self.load_files()
        data = self.add_info2data(data)
        self.save_results(data)
        print(f"r2 = {data['r2']:4f} \n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--hemi', type=str, default='rh')
    parser.add_argument('--roi', type=str, default='EVC')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    ROIPrediction(args).run()

if __name__ == '__main__':
    main()
