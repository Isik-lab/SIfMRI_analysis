#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from src import tools


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
        self.model = args.model
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
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

    def get_file_name(self, var):
        top = f'{self.out_dir}/VoxelPermutation'
        if 'null' not in var and 'var' not in var:
            file_name = f'{top}/sub-{self.sid}_prediction-all_drop-{self.model}_single-None_method-{self.method}_{var}.nii.gz'
        else:
            file_name = f'{top}/dist/sub-{self.sid}_prediction-all_drop-{self.model}_single-None_method-{self.method}_{var}.nii.gz'
        return file_name

    def load_files(self):
        data = dict()
        for key in ['r2', 'r2var', 'r2null']:
            file = self.get_file_name(key)
            data[key] = mask_img(file, self.roi_mask)
            print(key)
            print(data[key].shape)
            print()
        return data

    def run(self):
        self.load_files()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--model', type=str, default='communication')
    parser.add_argument('--hemi', type=str, default='rh')
    parser.add_argument('--roi', type=str, default='SIpSTS')
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
