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
        self.sid = str(args.s_num).zfill(2)
        self.feature =  args.feature
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def get_feature_index(self):
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        return features.index(self.feature)

    def load_regression_output(self, feature_index):
        pred = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_y_pred.npy')
        true = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_y_true.npy')
        indices = np.load(f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_indices.npy')
        return true, pred[feature_index, ...], indices

    def run(self):
        feature_index = self.get_feature_index()
        y_true, y_pred, test_inds = self.load_regression_output(feature_index)
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred, test_inds=test_inds)

        print('Saving output')
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_r_true.npy', r_true)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_p.npy', p)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_r_null.npy', r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--feature', '-f', type=int, default=None)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()