#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from src import tools
from tqdm import tqdm
import pickle
from scipy.stats import pearsonr, permutation_test, bootstrap


def statistic(x, y):
    return np.sign(pearsonr(x, y).statistic) * (pearsonr(x, y).statistic ** 2)


class FeaturePermutation:
    def __init__(self, args):
        self.process = 'FeaturePermutation'
        self.feature = args.feature
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.out_file_name = f'{self.out_dir}/{self.process}/feature-{self.feature}_motion-energy.pkl'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def load(self):
        file_prefix = f'{self.out_dir}/FeatureRegression/feature-{self.feature}_motion-energy'
        return np.load(f'{file_prefix}_y-test.npy'), np.load(f'{file_prefix}_y-pred.npy')

    def run_permutation(self, y_true, y_pred, out_data):
        # Run permutation
        res = permutation_test((y_true, y_pred), statistic, vectorized=False,
                               permutation_type='pairings', alternative='greater',
                               random_state=0, n_resamples=self.n_perm)

        # transform arrays to nii
        for name, i in zip(['r2', 'p'], [res.statistic[0], res.pvalue[0]]):
            out_data[name] = i
        return out_data

    def run_bootstrap(self, y_true, y_pred, out_data):
        # Run bootstrap
        res = bootstrap((y_true, y_pred), statistic,
                        vectorized=False, paired=True,
                        random_state=0, n_resamples=self.n_perm)
        ci = res.confidence_interval
        out_data['low_ci'], out_data['high_ci'] = ci[0][0], ci[1][0]
        return out_data

    def save_results(self, d):
        f = open(self.out_file_name, "wb")
        pickle.dump(d, f)
        f.close()

    def run(self):
        y_true, y_pred = self.load()
        out_data = {'feature': self.feature, 'layer': 'moten'}
        out_data = self.run_permutation(y_true, y_pred, out_data)
        out_data = self.run_bootstrap(y_true, y_pred, out_data)
        print(out_data)
        self.save_results(out_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='expanse')
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    FeaturePermutation(args).run()


if __name__ == '__main__':
    main()
