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
        self.sid = str(args.s_num).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.feature_num = args.feature_num
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def run(self):
        control_model = np.load(f'{self.out_dir}/generate_models/control_model.npy')
        X = np.load(f'{self.out_dir}/generate_models/annotated_model.npy')

        # Get the feature names for the annotated model
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        n_features = len(features)

        # Initialize KF splitter
        kf = KFold(n_splits=10, shuffle=True, random_state=1)

        # load the beta values and mask to reliable voxels
        beta_map = np.load(f'{self.out_dir}/grouped_runs/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        mask = np.load(f'{self.out_dir}/group_reliability/sub-all_reliability-mask.npy')
        beta_map = beta_map[mask, :]

        # Run the regression and print out the timing
        print('Starting regression')
        start = time.time()
        y_true, y_pred, indices = regress.outer_ridge_2d(X, control_model, beta_map,
                                                   n_features, kf)
        print(f'Finished regression in {(time.time() - start) / 60:.2f} minutes')

        # Save the outputs of the code
        print('Saving outputs')
        start = time.time()
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y_true.npy', y_true)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y_pred.npy', y_pred)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_indices.npy', indices)
        print(f'Finished saving in {(time.time() - start) / 60:.2f} minutes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--feature_num', '-f', type=int, default=None)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelEncoding(args).run()


if __name__ == '__main__':
    main()