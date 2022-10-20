#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from src import tools
import nibabel as nib


class VoxelPermutation:
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.feature = args.feature
        self.unique_variance = args.unique_variance
        self.full_model = args.full_model
        assert (self.feature is None) or self.unique_variance, "not yet implemented"
        assert (not self.full_model) or (
                    self.full_model and self.feature is None and self.category is None and not self.unique_variance), "no other inputs can be combined with the full model"
        self.step = args.step
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}/dist').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.in_file_prefix = ''
        self.out_file_prefix = ''
        self.dist_file_prefix = ''
        self.allfeature_file_prefix = ''
        im = nib.load(
            f'{self.data_dir}/betas_3mm_zscore/sub-{self.sid}/sub-{self.sid}_space-T1w_desc-train-{self.step}_data.nii.gz')
        self.im_shape = im.shape[:-1]
        self.affine = im.affine
        self.header = im.header
        del im

    def get_file_names(self):
        if self.full_model:
            base = f'sub-{self.sid}_full-model'
        else:
            if self.unique_variance:
                if self.category is not None:
                    base = f'sub-{self.sid}_dropped-category-{self.category}'
                else:  # self.feature is not None:
                    base = f'sub-{self.sid}_dropped-feature-{self.feature}'
            else:  # not self.unique_variance
                if self.category is not None:
                    # Regression with the categories without the other regressors
                    base = f'sub-{self.sid}_category-{self.category}'
                elif self.feature is not None:
                    base = f'sub-{self.sid}_feature-{self.feature}'
                else:  # This is the full regression model with all annotated features
                    base = f'sub-{self.sid}_all-features'
        self.allfeature_file_prefix = f'{self.out_dir}/VoxelRegression/sub-{self.sid}_all-features'
        self.in_file_prefix = f'{self.out_dir}/VoxelRegression/{base}'
        self.out_file_prefix = f'{self.out_dir}/{self.process}/{base}'
        self.dist_file_prefix = f'{self.out_dir}/{self.process}/dist/{base}'
        print(self.allfeature_file_prefix)
        print(self.in_file_prefix)
        print(self.out_file_prefix)
        print(self.dist_file_prefix)

    def load(self):
        test = np.load(f'{self.in_file_prefix}_y-test.npy')
        if self.unique_variance:
            pred = np.load(f'{self.allfeature_file_prefix}_y-pred.npy')
            loo = np.load(f'{self.in_file_prefix}_y-pred.npy')
        else:
            pred = np.load(f'{self.in_file_prefix}_y-pred.npy')
            loo = None
        return test, pred, loo

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
            nib.save(d[key], f'{self.out_file_prefix}_{key}.nii.gz')
            print(f'Saved {key} successfully')

    def save_dist(self, arr, name):
        np.save(f'{self.dist_file_prefix}_{name}.npy', arr)

    def run_permutation(self, y_true, y_pred):
        # Run permutation
        r2, p, r2_null = tools.perm(y_true, y_pred,
                                    n_perm=self.n_perm)
        r2_filtered, p_corrected = tools.filter_r(r2, p)

        # transform arrays to nii
        out_data = dict()
        for name, i in zip(['r2', 'r2filtered', 'p', 'pcorrected'],
                           [r2, r2_filtered, p, p_corrected]):
            out_data[name] = self.nib_transform(i)
        self.save_perm_results(out_data)
        self.save_dist(r2_null, 'r2null')
        print('Permutation testing done')

    def run_loo_permutation(self, y_true, y_pred, y_loo):
        # Run permutation
        r2, p, r2_null = tools.perm_unique_variance(y_true, y_pred, y_loo,
                                                    n_perm=self.n_perm)
        r2_filtered, p_corrected = tools.filter_r(r2, p)

        # transform arrays to nii
        out_data = dict()
        for name, i in zip(['r2', 'r2filtered', 'p', 'pcorrected'],
                           [r2, r2_filtered, p, p_corrected]):
            out_data[name] = self.nib_transform(i)
        self.save_perm_results(out_data)
        self.save_dist(r2_null, 'r2null')
        print('LOO permutation testing done')

    def run_bootstrap(self, y_true, y_pred):
        # Run bootstrap
        r2_var = tools.bootstrap(y_true, y_pred,
                                 n_perm=self.n_perm)
        self.save_dist(r2_var, 'r2var')
        print('Bootstrapping done')

    def run_loo_bootstrap(self, y_true, y_pred, y_loo):
        r2_var = tools.bootstrap_unique_variance(y_true, y_pred, y_loo,
                                                 n_perm=self.n_perm)
        self.save_dist(r2_var, 'r2var')
        print('LOO bootstrapping done')

    def run(self):
        self.get_file_names()
        if self.unique_variance:
            y_true, y_pred, y_loo = self.load()
            self.run_loo_permutation(y_true, y_pred, y_loo)
            self.run_loo_bootstrap(y_true, y_pred, y_loo)
        else:
            y_true, y_pred, _ = self.load()
            self.run_permutation(y_true, y_pred)
            self.run_bootstrap(y_true, y_pred)
        print('Completed successfully!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--full_model', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()


if __name__ == '__main__':
    main()
