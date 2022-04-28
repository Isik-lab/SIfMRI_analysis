#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from src import tools
from pathlib import Path


class VoxelPermutationTest():
    def __init__(self, args):
        self.process = 'VoxelPermutationTest'
        self.model = args.model
        self.sid = str(args.s_num).zfill(2)
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def load(self):
        pred = np.load(f'{self.out_dir}/VoxelEncodingTest/sub-{self.sid}_{self.model}.npy')
        true = np.load(f'{self.out_dir}/VoxelEncodingTest/sub-{self.sid}_y-test.npy')
        return true, pred

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_{self.model}'
        np.save(f'{base}_rs.npy', r_true)
        np.save(f'{base}_ps.npy', p)
        np.save(f'{base}_rnull.npy', r_null)
        print('Completed successfully!')

    def run(self):
        y_true, y_pred = self.load()
        print(y_true.shape)
        print(y_pred.shape)
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred, n_perm=self.n_perm)
        self.save_perm_results(r_true, p, r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutationTest(args).run()

if __name__ == '__main__':
    main()