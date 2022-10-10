#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pickle


def mask_img(img, mask):
    if type(img) is str:
        img = nib.load(img)
        img = np.array(img.dataobj)

    if type(mask) is str:
        mask = nib.load(mask)

    mask = np.array(mask.dataobj, dtype=bool)
    return img[mask].squeeze()


class ROIBetas:
    def __init__(self, args):
        self.process = 'ROIBetas'
        self.sid = str(args.s_num).zfill(2)
        self.hemi = args.hemi
        self.model = args.model
        self.roi = args.roi
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        self.roi_file = glob.glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*{self.roi}*{self.hemi}*mask.nii.gz')[0]
        self.reliability_file = f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz'
        print(vars(self))
        self.roi_mask = mask_img(self.roi_file, self.reliability_file).astype('bool')
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.out_file_name = f'{self.out_dir}/{self.process}/sub-{self.sid}_model-{self.model}_roi-{self.roi}_hemi-{self.hemi}.pkl'

    def load_files(self, data, key):
        file = f'{self.out_dir}/PlotBetas/sub-{self.sid}_feature-{self.model}.nii.gz'
        reliable_data = mask_img(file, self.reliability_file)
        roi_data = reliable_data[self.roi_mask]
        print(f'{np.sum(self.roi_mask)} voxels in {self.sid} {self.hemi} {self.roi}')
        data['betas'] = roi_data.mean()
        data['betas_std'] = roi_data.std()
        data['betas_sem'] = data['betas_std']/np.sqrt(roi_data.shape[-1])
        data['low_ci'] = np.percentile(roi_data, 0.025)
        data['high_ci'] = np.percentile(roi_data, 0.975)
        data['low_sem'] = data['betas'] - data['betas_sem']
        data['high_sem'] = data['betas'] + data['betas_sem']
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

        data = self.load_files(data, 'betas')

        # Add all the necessary info and save
        data = self.add_info2data(data)
        self.save_results(data)


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
    ROIBetas(args).run()

if __name__ == '__main__':
    main()
