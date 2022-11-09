#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from src import tools
from tqdm import tqdm
import pickle


def perm(a, b, n_perm, H0='greater'):
    r = tools.corr(a, b)
    r2 = np.sign(r) * (r ** 2)

    r2_null = np.zeros(n_perm)
    for i in tqdm(range(n_perm)):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        r = tools.corr(a[inds], b)
        r2_null[i] = np.sign(r) * (r ** 2)

    p = tools.calculate_p(r2_null, r2, n_perm, H0)
    return r2, p, r2_null


def bootstrap(a, b, n_perm):
    r2_var = np.zeros(n_perm)
    for i in tqdm(range(n_perm)):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        r = tools.corr(a[inds], b[inds])
        r2_var[i] = np.sign(r) * (r ** 2)
    return r2_var


class FeaturePermutation:
    def __init__(self, args):
        self.process = 'FeaturePermutation'
        self.feature = args.feature
        self.layer = args.layer
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.out_file_name = f'{self.out_dir}/{self.process}/feature-{self.feature}_alexnet-conv{self.layer}.pkl'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def load(self):
        file_prefix = f'{self.out_dir}/FeatureRegression/feature-{self.feature}_alexnet-conv{self.layer}'
        return np.load(f'{file_prefix}_y-test.npy'), np.load(f'{file_prefix}_y-pred.npy')

    def run_permutation(self, y_true, y_pred, out_data):
        # Run permutation
        r2, p, r2_null = perm(y_true, y_pred,
                              n_perm=self.n_perm)
        # r2_filtered, p_corrected = tools.filter_r(r2, p)

        # transform arrays to nii
        for name, i in zip(['r2', 'p'], [r2, p]):
            out_data[name] = i
        return out_data

    def run_bootstrap(self, y_true, y_pred, out_data):
        # Run bootstrap
        r2_var = bootstrap(y_true, y_pred,
                           n_perm=self.n_perm)
        out_data['low_ci'], out_data['high_ci'] = np.percentile(r2_var, [2.5, 97.5])
        return out_data

    def save_results(self, d):
        f = open(self.out_file_name, "wb")
        pickle.dump(d, f)
        f.close()

    def run(self):
        y_true, y_pred = self.load()
        out_data = {'feature': self.feature, 'layer': self.layer}
        out_data = self.run_permutation(y_true, y_pred, out_data)
        out_data = self.run_bootstrap(y_true, y_pred, out_data)
        print(out_data)
        self.save_results(out_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='expanse')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    FeaturePermutation(args).run()


if __name__ == '__main__':
    main()
