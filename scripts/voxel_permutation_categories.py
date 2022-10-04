#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from src import tools
import nibabel as nib


class CategoryVoxelPermutation:
    def __init__(self, args):
        self.process = 'CategoryVoxelPermutation'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.step = args.step
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}/dist').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        im = nib.load(
            f'{self.data_dir}/betas_3mm_zscore/sub-{self.sid}/sub-{self.sid}_space-T1w_desc-train-{self.step}_data.nii.gz')
        self.im_shape = im.shape[:-1]
        self.affine = im.affine
        self.header = im.header
        del im

    def load(self):
        in_dir = f'{self.out_dir}/CategoryVoxelRegression/'
        pred = np.load(f'{in_dir}/sub-{self.sid}_category-{self.category}_y-pred.npy')
        test = np.load(f'{in_dir}/sub-{self.sid}_category-{self.category}_y-test.npy')
        return test, pred

    def load_anatomy(self):
        anat = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-preproc_T1w.nii.gz')
        brain_mask = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-brain_mask.nii.gz')
        return tools.mask_img(anat, brain_mask)

    def nib_transform(self, r_, nii=True):
        unmask = np.load(
            f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-{self.step}_reliability-mask.npy').astype(
            'bool')
        i = np.where(unmask)
        if r_.ndim < 2:
            r_unmasked = np.zeros(unmask.shape)
            r_unmasked[i] = r_
            r_unmasked = r_unmasked.reshape(self.im_shape)
        else:
            r_ = r_.T
            r_unmasked = np.zeros((unmask.shape + (r_.shape[-1],)))
            r_unmasked[i, ...] = r_
            r_unmasked = r_unmasked.reshape((self.im_shape + (r_.shape[-1],)))
            print(r_unmasked.shape)

        if nii:
            r_unmasked = nib.Nifti1Image(r_unmasked, self.affine, self.header)
        return r_unmasked

    def save_perm_results(self, d):
        print('Saving output')
        for key in d.keys():
            base = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}'
            nib.save(d[key], f'{base}_{key}.nii.gz')
            print(f'Saved {key} successfully')

    def run(self):
        y_true, y_pred = self.load()
        print(np.unique(y_true))

        # Run permutation
        r2, p, r2_null = tools.perm(y_true, y_pred, n_perm=self.n_perm)
        base = f'{self.out_dir}/{self.process}/dist/sub-{self.sid}_category-{self.category}'
        np.save(f'{base}_r2null.npy', self.nib_transform(r2_null, nii=False))
        del r2_null

        # Run bootstrap
        r2_var = tools.bootstrap(y_true, y_pred, n_perm=self.n_perm)
        np.save(f'{base}_r2var.npy', self.nib_transform(r2_var, nii=False))
        del r2_var

        # filter the rs based on the significant voxels
        r2_filtered, p_corrected = tools.filter_r(r2, p)

        # transform arrays to nii
        out_data = dict()
        for name, i in zip(['r2', 'r2filtered', 'p', 'pcorrected'],
                           [r2, r2_filtered, p, p_corrected]):
            out_data[name] = self.nib_transform(i)
        self.save_perm_results(out_data)
        print('Completed successfully!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--category', type=str, default='affective')
    parser.add_argument('--n_perm', type=int, default=10000)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    CategoryVoxelPermutation(args).run()


if __name__ == '__main__':
    main()
