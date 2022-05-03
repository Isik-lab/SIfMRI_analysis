#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path


def compute_ps(r_true, r_null, H0='greater'):
    n_perm = r_null.shape[1]
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
    return p


class VoxelGroupResults():
    def __init__(self, args):
        self.model = args.model.replace('_', ' ')
        self.cross_validation = args.cross_validation
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.n_subjs = 4
        Path(f'{self.out_dir}/VoxelPermutation').mkdir(parents=True, exist_ok=True)
        instance_variables = vars(self)
        print(instance_variables)

    def load(self):
        rs = None
        for i in range(self.n_subjs):
            sid = str(i+1).zfill(2)
            r = np.load(f'{self.out_dir}/VoxelPermutation/sub-{sid}_prediction-{self.model}_method-{self.method}_rs.npy')
            null = np.load(
                f'{self.out_dir}/VoxelPermutation/rnull/sub-{sid}_prediction-{self.model}_method-{self.method}_rnull.npy')
            if rs is None:
                rs = r.copy()
                rs_null = null.copy()
            else:
                rs += rs
                rs_null += null
        rs /= self.n_subjs
        rs_null /= self.n_subjs
        return rs, rs_null

    def save(self, rs, ps):
        np.save(f'{self.out_dir}/VoxelPermutation/sub-all_prediction-{self.model}_method-{self.method}_rs.npy', rs)
        np.save(f'{self.out_dir}/VoxelPermutation/sub-all_prediction-{self.model}_method-{self.method}_ps.npy', ps)

    def run(self):
        rs, rs_null = self.load()
        ps = compute_ps(rs, rs_null)
        self.save(rs, ps)
        print("Completed successfully")


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
    VoxelGroupResults(args).run()

if __name__ == '__main__':
    main()