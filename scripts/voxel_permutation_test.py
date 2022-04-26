#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from src import tools, regress
from pathlib import Path


class VoxelPermutationTest():
    def __init__(self, args):
        self.process = 'VoxelPermutationTest'
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
        base = f'{self.out_dir}/VoxelEncodingTest/{self.y_pred}'
        pred = np.load(base)
        true = np.load(base.replace('_y_pred', '_y_true'))
        return true, pred

    def load_by_feature(self):
        feature_index = regress.get_feature_inds(self.pred_feature)
        true, pred = self.load()
        return true, pred[feature_index, ...].squeeze()

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/{self.y_pred}'
        if 'predict-features' in self.y_pred:
            base = base.replace('predict-features', f'predict-{self.pred_feature}')
        np.save(base.replace('_y_pred', '_rs'), r_true)
        np.save(base.replace('_y_pred', '_ps'), p)
        np.save(base.replace('_y_pred', '_r_null'), r_null)
        print('Completed successfully!')

    def run(self):
        if 'predict-features' in self.y_pred:
            print('by feature prediction')
            y_true, y_pred = self.load_by_feature()
        else:
            print('not completing by feature prediction')
            y_true, y_pred = self.load()
        print(y_true.shape)
        print(y_pred.shape)
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred, n_perm=self.n_perm)
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
    VoxelPermutationTest(args).run()

if __name__ == '__main__':
    main()