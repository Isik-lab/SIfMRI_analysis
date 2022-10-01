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
    return img[mask]


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
        self.model = args.model
        if self.model is not None:
            self.model = self.model.replace('_', ' ')
        self.roi = args.roi
        self.contrast = roi2contrast(self.roi)
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.roi_mask = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*{self.contrast}*{self.hemi}*mask.nii.gz')[0]
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_model-{self.model}_roi-{self.roi}_hemi-{self.hemi}.pkl'
        print(vars(self))

    def get_file_name(self, name):
        top = f'{self.out_dir}/VoxelPermutation'
        if 'null' not in name and 'var' not in name:
            file_name = f'{top}/sub-{self.sid}_prediction-all_drop-{self.model}_single-None_method-{self.method}_{name}.nii.gz'
        else:
            file_name = f'{top}/dist/sub-{self.sid}_prediction-all_drop-{self.model}_single-None_method-{self.method}_{name}.npy'
        return file_name

    def load_files(self, data, key):
        file = self.get_file_name(key)
        if 'npy' in file:
            file = np.load(file)
        data[key] = mask_img(file, self.roi_mask).mean(axis=0)
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
        d['model'] = self.model
        return d

    def run(self):
        data = dict()

        # Variance of ROI
        data = self.load_files(data, 'r2var')
        print(f'{getsizeof(data) / (1024 * 1024):.2f} MB')
        data['low_ci'], data['high_ci'] = tools.compute_confidence_interval(data['r2var'])
        del data['r2var']  # Save memory

        # Significance of ROI
        data = self.load_files(data, 'r2')
        data = self.load_files(data, 'r2null')
        print(f'{getsizeof(data)/(1024*1024):.2f} MB')
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
    parser.add_argument('--model', type=str, default=None)
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
    ROIPrediction(args).run()

if __name__ == '__main__':
    main()
