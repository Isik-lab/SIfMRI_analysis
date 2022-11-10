#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from src import tools
import nibabel as nib
from scipy.stats import permutation_test, bootstrap
from timeit import default_timer as timer


def statistic(x, y, axis=0):
    x_m = x - np.expand_dims(np.mean(x, axis=axis), axis=1)
    y_m = y - np.expand_dims(np.mean(y, axis=axis), axis=1)

    numer = np.sum((x_m * y_m), axis=axis)
    denom = np.sqrt(np.sum((x_m * x_m), axis=axis) * np.sum((y_m * y_m), axis=axis))
    denom[np.isclose(denom, 0)] = np.nan
    r = numer / denom
    return np.sign(r) * (r ** 2)


class VoxelPermutationAlexNet:
    def __init__(self, args):
        self.process = 'VoxelPermutationAlexNet'
        self.sid = str(args.s_num).zfill(2)
        self.layer = args.layer
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
        file_prefix = f'{self.out_dir}/VoxelRegressionAlexNet/sub-{self.sid}_alexnet-conv{self.layer}'
        return np.load(f'{file_prefix}_y-test.npy'), np.load(f'{file_prefix}_y-pred.npy')

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
        if nii:
            r_unmasked = nib.Nifti1Image(r_unmasked, self.affine, self.header)
        return r_unmasked

    def save_perm_results(self, d):
        print('Saving output')
        file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_alexnet-conv{self.layer}'
        for key in d.keys():
            nib.save(d[key], f'{file_prefix}_{key}.nii.gz')
            print(f'Saved {key} successfully')

    def save_dist(self, arr, name):
        file = f'{self.out_dir}/{self.process}/dist/sub-{self.sid}_alexnet-conv{self.layer}_{name}.npy'
        np.save(file, arr)

    def run_permutation(self, y_true, y_pred):
        # Run permutation
        start = timer()
        res = permutation_test((y_true, y_pred), statistic,
                               vectorized=True,
                               axis=0,
                               permutation_type='pairings',
                               alternative='greater',
                               n_resamples=self.n_perm,
                               random_state=0)
        r2_filtered, p_corrected = tools.filter_r(res.statistic, res.pvalue)
        end = timer()
        print(end - start)

        # transform arrays to nii
        out_data = dict()
        for name, i in zip(['r2', 'r2filtered', 'p', 'pcorrected'],
                           [res.statistic, r2_filtered, res.pvalue, p_corrected]):
            out_data[name] = self.nib_transform(i)
        self.save_perm_results(out_data)
        self.save_dist(res.null_distribution, 'r2null')
        print('Permutation testing done')

    def run_bootstrap(self, y_true, y_pred):
        # Run bootstrap
        res = bootstrap((y_true, y_pred), statistic,
                        vectorized=True,
                        axis=0,
                        paired=True,
                        random_state=0)
        self.save_dist(res, 'r2var')
        print('Bootstrapping done')

    def run(self):
        y_true, y_pred = self.load()
        self.run_permutation(y_true, y_pred)
        self.run_bootstrap(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=2)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutationAlexNet(args).run()


if __name__ == '__main__':
    main()
