#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob
from src.tools import corr2d


class VoxelPermutation():
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.model = args.model.replace('_', ' ')
        self.sid = str(args.s_num).zfill(2)
        self.cross_validation = args.cross_validation
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        instance_variables = vars(self)
        print(instance_variables)

    def load(self):
        if self.cross_validation:
            pred_files = sorted(glob.glob(
                f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{self.model}_method-CV_loop*.npy'))
            true_files = sorted(
                glob.glob(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-CV_loop*.npy'))
            pred = None
            for i, (pred_file, true_file) in enumerate(zip(pred_files, true_files)):
                pred_file = np.load(pred_file)
                true_file = np.load(true_file)
                if pred is None:
                    pred = np.zeros((pred_file.shape[0], len(pred_files), pred_file.shape[1]))
                    true = np.zeros_like(pred)
                pred[:, i, :] = pred_file
                true[:, i, :] = true_file
        else:
            pred = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{self.model}_method-test.npy')
            true = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-test.npy')
        return true, pred

    def permutation_test_2d(self, a, b,
                            n_perm=int(5e3),
                            H0='greater'):
        if a.ndim == 3:
            r_true = corr2d(a.reshape(a.shape[0]*a.shape[1], a.shape[-1]),
                            b.reshape(b.shape[0]*b.shape[1], b.shape[-1]))
        else:
            r_true = corr2d(a, b)

        r_null = np.zeros((n_perm, a.shape[-1]))
        for i in tqdm(range(n_perm), total=n_perm):
            inds = np.random.default_rng(i).permutation(a.shape[0])
            if a.ndim == 3:
                a_shuffle = a[inds, :, :].reshape(a.shape[0]*a.shape[1], a.shape[-1])
                b_not_shuffled = b.reshape(b.shape[0]*b.shape[1], b.shape[-1])
            else: #a.ndim == 2:
                a_shuffle = a[inds, :]
                b_not_shuffled = b.copy()
            r_null[i, :] = corr2d(a_shuffle, b_not_shuffled)

        # Get the p-value depending on the type of test
        denominator = n_perm + 1
        if H0 == 'two_tailed':
            numerator = np.sum(np.abs(r_null) >= np.abs(r_true), axis=0) + 1
            p = numerator / denominator
        elif H0 == 'greater':
            numerator = np.sum(r_true > r_null, axis=0) + 1
            p = 1 - (numerator / denominator)
        else:  # H0 == 'less':
            numerator = np.sum(r_true < r_null, axis=0) + 1
            p = 1 - (numerator / denominator)
        return r_true, p, r_null

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_prediction-{self.model}_method-{self.method}'
        np.save(f'{base}_rs.npy', r_true)
        np.save(f'{base}_ps.npy', p)
        np.save(f'{self.out_dir}/{self.process}/rnull/sub-{self.sid}_prediction-{self.model}_method-{self.method}_rnull.npy', r_null)
        print('Completed successfully!')

    def run(self):
        y_true, y_pred = self.load()
        r_true, p, r_null = self.permutation_test_2d(y_true, y_pred, n_perm=self.n_perm)
        self.save_perm_results(r_true, p, r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--model', '-m', type=str, default='visual')
    parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()