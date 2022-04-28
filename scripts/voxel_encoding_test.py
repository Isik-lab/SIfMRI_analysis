#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from src import regress


class VoxelEncodingTest:
    def __init__(self, args):
        self.process = 'VoxelEncodingTest'
        self.n_subjs = args.n_subjs
        self.include_control = args.include_control
        self.layer = args.layer
        if self.include_control:
            assert self.layer is not None, "AlexNet layer must be defined"
            self.control_name = f'conv{self.layer}'
        else:
            self.control_name = 'none'
        self.predict_by_feature = args.predict_by_feature
        self.predict_grouped_features = args.predict_grouped_features
        self.feature_group = args.feature_group
        self.predict_features = None
        self.pred_indices = None
        if self.predict_by_feature:
            regress_name = 'model-full'
            predict_name = 'predict-features'
        elif self.predict_grouped_features:
            assert self.feature_group is not None, "Feature group must be defined for grouped prediction"
            regress_name = 'model-full'
            predict_name = f'predict-{self.feature_group}'
            d = {'social': ['joint action', 'communication', 'cooperation',
                            'dominance', 'intimacy', 'valence', 'arousal'],
                 'visual': ['indoor', 'expanse', 'transitivity'],
                 'socialprimitive': ['agent distance', 'facingness']}
            assert self.feature_group in d.keys(), \
                f'Valid group inputs are "social", "visual", or "socialprimitive", not {self.feature_group}'
            self.predict_features = d[self.feature_group]
            self.pred_indices = regress.get_feature_inds(self.predict_features)
        else:
            regress_name = 'model-full'
            predict_name = 'predict-all'
        self.regress_name = regress_name
        self.predict_name = predict_name
        self.sid = str(args.s_num).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)

    def mask_neural(self, set):
        mask = np.load(f'{self.out_dir}/Reliability/sub-all_set-test_reliability-mask.npy')
        path = f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_{set}-data.npy'
        beta_map = np.load(path)
        indices = np.where(mask)[0]
        return beta_map[indices, :].T

    def run(self):
        # load the control model if used
        if self.include_control:
            train_control = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-train.npy')
            # test_control = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-test.npy')
        else:
            train_control = None

        # Load the annotation model and filter if modeling by feature
        # X = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-train.npy')
        # X_test = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-test.npy')
        X = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-train.npy')
        X_test = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-test.npy')

        # Se tthe base name for the output files
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_{self.regress_name}_{self.predict_name}_control-{self.control_name}'
        print(f'{base}_y_pred.npy')

        # Get the feature names for the annotated model
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        n_features = len(features)

        # load the beta values and mask to reliable voxels
        y_train = self.mask_neural('train')
        y_test = self.mask_neural('test')

        # Run the regression and print out the timing
        print('Starting regression')
        start = time.time()
        y_pred, y_true, betas = regress.ridge(X, train_control, X_test,
                                              y_train, y_test,
                                              include_control=self.include_control,
                                              predict_by_feature=self.predict_by_feature,
                                              inds=self.pred_indices)
        print(f'Finished regression in {(time.time() - start) / 60:.2f} minutes')

        # Save the outputs of the code
        print('Saving outputs')
        start = time.time()
        np.save(f'{base}_y_true.npy', y_true)
        np.save(f'{base}_y_pred.npy', y_pred)
        np.save(f'{base}_betas.npy', betas)
        print(f'Finished saving in {(time.time() - start) / 60:.2f} minutes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--layer', '-l', type=str, default=2)
    parser.add_argument('--include_control', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--predict_by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--predict_grouped_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--feature_group', '-p', type=str, default=None)
    parser.add_argument('--n_subjs', '-n', type=int, default=4)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelEncodingTest(args).run()


if __name__ == '__main__':
    main()
