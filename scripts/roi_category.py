#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from src import tools
import pickle
from sys import getsizeof


def mask_img(img, mask):
    if type(img) is str:
        img = nib.load(img)
        img = np.array(img.dataobj)

    if type(mask) is str:
        mask = nib.load(mask)

    mask = np.array(mask.dataobj, dtype=bool)
    return img[mask].squeeze()


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


class ROICategory:
    def __init__(self, args):
        self.process = 'ROICategory'
        self.sid = str(args.s_num).zfill(2)
        self.hemi = args.hemi
        self.category = args.category
        self.roi = args.roi
        self.contrast = roi2contrast(self.roi)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.roi_file = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*{self.contrast}*{self.hemi}*mask.nii.gz')[0]
        self.reliability_file = f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz'
        self.roi_mask = mask_img(self.roi_file, self.reliability_file).astype('bool')
        print(self.roi_mask.shape)
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}_roi-{self.roi}_hemi-{self.hemi}.pkl'
        print(vars(self))

    def get_file_name(self, name):
        top = f'{self.out_dir}/CategoryVoxelPermutation'
        if 'null' not in name and 'var' not in name:
            file_name = f'{top}/sub-{self.sid}_category-{self.category}_{name}.nii.gz'
        else:
            file_name = f'{top}/dist/sub-{self.sid}_category-{self.category}_{name}.npy'
        return file_name

    def load_files(self, data, key):
        file = self.get_file_name(key)
        if 'npy' in file:
            reliable_data = np.load(file)
            roi_data = reliable_data[:, self.roi_mask]
            roi_mean_data = np.nanmean(roi_data, axis=1)
        else:
            reliable_data = mask_img(file, self.reliability_file)
            roi_data = reliable_data[self.roi_mask]
            roi_mean_data = np.nanmean(roi_data)
        print(reliable_data.shape)
        print(roi_data.shape)
        print(roi_mean_data.shape)
        data[key] = roi_mean_data
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
        d['category'] = self.category
        return d

    def run(self):
        data = dict()

        # Variance of ROI
        data = self.load_files(data, 'r2var')
        data['low_ci'], data['high_ci'] = tools.compute_confidence_interval(data['r2var'])
        del data['r2var']  # Save memory

        # Significance of ROI
        data = self.load_files(data, 'r2')
        data = self.load_files(data, 'r2null')
        data['p'] = tools.calculate_p(data['r2null'], data['r2'],
                                      n_perm_=len(data['r2null']), H0_='greater')
        del data['r2null'] #Save memory

        # Add all the necessary info and save
        data = self.add_info2data(data)
        self.save_results(data)
        print(f"r2 = {data['r2']:4f}")
        print(f"p = {data['p']:4f}")
        print(f"low_ci = {data['low_ci']:4f}")
        print(f"high_ci = {data['high_ci']:4f} \n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--category', type=str, default='affective')
    parser.add_argument('--hemi', type=str, default='rh')
    parser.add_argument('--roi', type=str, default='EVC')
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    ROICategory(args).run()

if __name__ == '__main__':
    main()
