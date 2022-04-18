#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from src import tools, regress
from pathlib import Path


class VoxelPermutation():
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.y_pred = args.y_pred
        if '/' in self.y_pred:
            self.y_pred = self.y_pred.split('/')[-1]
        self.pred_feature = args.pred_feature
        print(self.pred_feature)
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def load(self):
        base = f'{self.out_dir}/VoxelEncoding/{self.y_pred}'
        pred = np.load(base)
        true = np.load(base.replace('y_pred', 'y_true'))
        indices = np.load(base.replace('y_pred', 'indices'))
        return true, pred, indices

    def load_by_feature(self):
        feature_index = regress.get_feature_inds(self.pred_feature)
        true, pred, indices = self.load()
        return true, pred[feature_index, ...].squeeze(), indices

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/{self.y_pred}'
        np.save(base.replace('y_pred', 'rs'), r_true)
        np.save(base.replace('y_pred', 'ps'), p)
        np.save(base.replace('y_pred', 'r_null'), r_null)
        print('Completed successfully!')

    def run(self):
        if 'predict-features' in self.y_pred:
            y_true, y_pred, test_inds = self.load_by_feature()
            print('not completing by feature prediction')
        else:
            y_true, y_pred, test_inds = self.load()
        print(y_true.shape)
        print(y_pred.shape)
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred,
                                                      test_inds=test_inds, n_perm=self.n_perm)
        self.save_perm_results(r_true, p, r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_pred', type=str)
    parser.add_argument('--pred_feature', type=str)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()