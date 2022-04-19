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
        self.predict_by_feature = args.predict_by_feature
        self.model_by_feature = args.model_by_feature
        self.predict_grouped_features = args.predict_grouped_features
        self.model_features = args.model_features
        self.predict_features = args.predict_features
        if self.model_by_feature:
            assert self.model_features, "features must be defined to model by feature"
            regress_name = 'model-'
            for feature in self.model_features:
                regress_name += feature.replace(' ', '').capitalize()
            predict_name = 'predict-all'
        elif self.predict_by_feature:
            regress_name = 'model-full'
            predict_name = 'predict-features'
        elif self.predict_grouped_features:
            regress_name = 'model-full'
            predict_name = 'predict-'
            for feature in self.predict_features:
                predict_name += feature.replace(' ', '').capitalize()
        else:
            regress_name = 'model-full'
            predict_name = 'predict-all'
        self.regress_name = regress_name
        self.predict_name = predict_name
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
            control_model = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-{self.set}.npy')
        else:
            control_model = None
        X = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-{self.set}.npy')
        if self.model_by_feature:
            model_indices = regress.get_feature_inds(self.model_features)
            X = X[:, model_indices]

        if self.predict_features:
            pred_indices = regress.get_feature_inds(self.predict_features)
        else:
            pred_indices = None

        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_{self.regress_name}_{self.predict_name}_control-{self.control_name}_pca-{self.pca_before_regression}'
        print(base)

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
                    beta_map = np.load(f'{self.out_dir}/GroupRuns/sub-{sid}/sub-{sid}_train-data.npy')
                else:
                    beta_map += np.load(f'{self.out_dir}/GroupRuns/sub-{sid}/sub-{sid}_train-data.npy')
            beta_map /= self.n_subjs
        else:
            beta_map = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        mask = np.load(f'{self.out_dir}/Reliability/sub-{self.sid}_set-test_reliability-mask.npy')
        indices = np.where(mask)[0]
        print(len(indices))
        beta_map = beta_map[indices, :]

        # Run the regression and print out the timing
        print('Starting regression')
        start = time.time()
        y_true, y_pred, indices = regress.cross_validated_ridge(X, control_model, beta_map,
                                                                n_features, kf,
                                                                include_control=self.include_control,
                                                                predict_by_feature=self.predict_by_feature,
                                                                pca_before_regression=self.pca_before_regression,
                                                                inds=pred_indices)
        print(f'Finished regression in {(time.time() - start) / 60:.2f} minutes')

        # Save the outputs of the code
        print('Saving outputs')
        start = time.time()
        np.save(f'{base}_y_true.npy', y_true)
        np.save(f'{base}_y_pred.npy', y_pred)
        np.save(f'{base}_indices.npy', indices)
        print(f'Finished saving in {(time.time() - start) / 60:.2f} minutes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default="1")
    parser.add_argument('--layer', '-l', type=str, default="2")
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--include_control', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--predict_by_feature', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model_by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--predict_grouped_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model_features', '-m', action='append', default=[])
    parser.add_argument('--predict_features', '-p', action='append', default=[])
    parser.add_argument('--n_subjs', '-n', type=int, default=4)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelEncoding(args).run()


if __name__ == '__main__':
    main()
