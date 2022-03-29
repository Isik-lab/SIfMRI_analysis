#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import numpy as np
import pandas as pd
from src import regress
from sklearn.model_selection import KFold


class VoxelEncoding:
    def __init__(self, args):
        self.process = 'VoxelEncoding'
        self.n_subjs = args.n_subjs
        self.set = args.set
        self.include_control = args.include_control
        self.layer = args.layer
        if self.include_control:
            assert self.layer is not None, "AlexNet layer must be defined"
            self.control_name = f'conv{self.layer}'
        else:
            self.control_name = 'none'
        self.by_feature = args.by_feature
        self.pca_before_regression = args.pca_before_regression
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def run(self, regression_splits=10, random_state=1):
        if self.include_control:
            control_model = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}.npy')
        else:
            control_model = None
        X = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-{self.set}.npy')

        # Get the feature names for the annotated model
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        n_features = len(features)

        # Initialize KF splitter
        kf = KFold(n_splits=regression_splits, shuffle=True, random_state=random_state)

        # load the beta values and mask to reliable voxels
        if self.sid == 'all':
            beta_map = None
            for i in range(self.n_subjs):
                sid = str(i+1).zfill(2)
                if beta_map is None:
                    beta_map = np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_train-data.npy')
                else:
                    beta_map += np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_train-data.npy')
            beta_map /= self.n_subjs
        else:
            beta_map = np.load(f'{self.out_dir}/grouped_runs/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        mask = np.load(f'{self.out_dir}/Reliability/sub-all_reliability-mask.npy')
        indices = np.where(mask)[0]
        beta_map = beta_map[indices, :]

        # Run the regression and print out the timing
        print('Starting regression')
        start = time.time()
        y_true, y_pred, indices = regress.cross_validated_ridge(X, control_model, beta_map,
                                                                n_features, kf,
                                                                include_control=self.include_control,
                                                                by_feature=self.by_feature,
                                                                pca_before_regression=self.pca_before_regression)
        print(f'Finished regression in {(time.time() - start) / 60:.2f} minutes')

        # Save the outputs of the code
        print('Saving outputs')
        start = time.time()
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_by_feature-{self.by_feature}_control-{self.control_name}_pca_before_regression-{self.pca_before_regression}'
        np.save(f'{base}_y_true.npy', y_true)
        np.save(f'{base}_y_pred.npy', y_pred)
        np.save(f'{base}_indices.npy', indices)
        print(f'Finished saving in {(time.time() - start) / 60:.2f} minutes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--layer', '-l', type=str, default=None)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--include_control', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_subjs', '-n', type=int, default=4)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelEncoding(args).run()


if __name__ == '__main__':
    main()
