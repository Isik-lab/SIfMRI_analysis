#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
from src import tools

class VoxelPermutation():
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.n_subjs = args.n_subjs
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.by_feature = args.by_feature
        if self.by_feature:
            assert args.feature is not None, "You must define feature to use by_feature mode"
            self.feature_snake = args.feature.replace("_", " ")
            self.feature = self.feature_snake.replace("_", " ")
        else:
            self.feature_snake = 'all'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def get_feature_index(self):
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        return features.index(self.feature)

    def load(self):
        pred = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_y_pred.npy')
        true = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_y_true.npy')
        indices = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_indices.npy')
        return true, pred, indices

    def load_by_feature(self):
        feature_index = self.get_feature_index()
        true, pred, indices = self.load()
        return true, pred[feature_index, ...], indices

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_feature-{self.feature_snake}_rs.npy', r_true)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_feature-{self.feature_snake}_ps.npy', p)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_feature-{self.feature_snake}_rs-nulldist.npy', r_null)
        print('Completed successfully!')

    def run(self):
        if self.by_feature:
            y_true, y_pred, test_inds = self.load_regress_by_feature()
        else:
            y_true, y_pred, test_inds = self.load()
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred, test_inds=test_inds)
        self.save_perm_results(r_true, p, r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--n_subjs', '-n', type=int, default=4)
    parser.add_argument('--by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--feature', '-f', type=str, default=None)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()