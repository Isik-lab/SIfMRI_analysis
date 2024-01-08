#!/usr/bin/env python
# coding: utf-8

from glob import glob
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from src import tools
import pickle


def mask_img(img, mask):
    if type(img) is str:
        img = nib.load(img)
        img = np.array(img.dataobj)

    if type(mask) is str:
        mask = nib.load(mask)

    mask = np.array(mask.dataobj, dtype=bool)
    return img[mask].squeeze()


class ROIPrediction:
    def __init__(self, args):
        self.process = 'ROIPrediction'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.feature = args.feature
        self.unique_variance = args.unique_variance
        self.include_nuisance = args.include_nuisance
        self.full_model = args.full_model
        assert (self.feature is None) or self.unique_variance, "not yet implemented"
        assert (not self.full_model) or (
                self.full_model and self.feature is None and self.category is None and not self.unique_variance), "no other inputs can be combined with the full model"
        self.roi = args.roi
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        print(vars(self))
        reliability_file = f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz'
        self.reliability_mask = nib.load(reliability_file).get_fdata().astype('bool')
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        self.in_file_prefix = ''
        self.out_file_name = ''

    def get_file_name_base(self):
        if self.full_model:
            self.in_file_prefix = f'sub-{self.sid}_full-model'
        else:
            if self.unique_variance and self.include_nuisance:
                if self.category is not None:
                    self.in_file_prefix = f'sub-{self.sid}_dropped-categorywithnuisance-{self.category}'
                else:  # self.feature is not None:
                    self.in_file_prefix = f'sub-{self.sid}_dropped-featurewithnuisance-{self.feature}'
            elif self.unique_variance and not self.include_nuisance:
                if self.category is not None:
                    self.in_file_prefix = f'sub-{self.sid}_dropped-category-{self.category}'
                else:  # self.feature is not None:
                    self.in_file_prefix = f'sub-{self.sid}_dropped-feature-{self.feature}'
            else:  # not self.unique_variance
                if self.category is not None:
                    # Regression with the categories without the other regressors
                    self.in_file_prefix = f'sub-{self.sid}_category-{self.category}'
                elif self.feature is not None:
                    self.in_file_prefix = f'sub-{self.sid}_feature-{self.feature}'
                else:  # This is the full regression model with all annotated features
                    self.in_file_prefix = f'sub-{self.sid}_all-features'
        print(self.in_file_prefix)
        self.out_file_name = f'{self.out_dir}/{self.process}/{self.in_file_prefix}_roi-{self.roi}.pkl'

    def load_roi_mask(self):
        # Loop through the two hemispheres
        roi_mask = []
        for hemi in ['lh', 'rh']:
            hemi_mask_file = glob(f'{self.data_dir}/localizers/sub-{self.sid}/sub-{self.sid}*roi-{self.roi}*hemi-{hemi}*mask.nii.gz')[0]
            hemi_mask = nib.load(hemi_mask_file).get_fdata().astype('bool')
            roi_mask.append(hemi_mask)
        roi_mask = np.sum(roi_mask, axis=0).astype('bool')
        return roi_mask[self.reliability_mask]
            
    def load_files(self, name, roi_mask):
        top = f'{self.out_dir}/VoxelPermutation'
        if 'null' in name or 'var' in name:
            file = f'{top}/dist/{self.in_file_prefix}_{name}.npy'
            permutation_data = np.load(file)
            roi_data = permutation_data[:, roi_mask].mean(axis=-1)
        else:
            file = f'{top}/{self.in_file_prefix}_{name}.nii.gz'
            permutation_data = nib.load(file).get_fdata()[self.reliability_mask]
            roi_data = permutation_data[roi_mask].mean()
        print(roi_data.shape)
        return roi_data

    def get_roi_data(self, name):
        roi_mask = self.load_roi_mask()
        return self.load_files(name, roi_mask)

    def get_variance(self, data):
        r2var = self.get_roi_data('r2var')
        data['r2var'] = r2var
        data['low_ci'], data['high_ci'] = np.percentile(r2var, [2.5, 97.5])

    def get_significance(self, data):
        data['r2'] = self.get_roi_data('r2')
        r2null = self.get_roi_data('r2null')
        data['r2null'] = r2null
        data['p'] = tools.calculate_p(r2null, data['r2'],
                                      n_perm_=len(r2null),
                                      H0_='greater')

    def add_info2data(self, data):
        data['sid'] = self.sid
        data['roi'] = self.roi
        data['category'] = self.category
        data['feature'] = self.feature
        data['unique_variance'] = self.unique_variance
        data['reliability'] = None

    def save_results(self, d):
        f = open(self.out_file_name, "wb")
        pickle.dump(d, f)
        f.close()

    def run(self):
        self.get_file_name_base()
        data = dict()
        self.get_variance(data)
        self.get_significance(data)
        self.add_info2data(data)
        self.save_results(data)
        print(f"r2 = {data['r2']:4f}")
        print(f"p = {data['p']:4f}")
        print(f"low_ci = {data['low_ci']:4f}")
        print(f"high_ci = {data['high_ci']:4f}")
        print(f"r2var len = {len(data['r2var'])}")
        print(f"r2null len = {len(data['r2null'])} \n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--roi', type=str, default='EVC')
    parser.add_argument('--full_model', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--include_nuisance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    ROIPrediction(args).run()


if __name__ == '__main__':
    main()
