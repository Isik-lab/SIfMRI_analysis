#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
from src import tools, regress


class VoxelPermutation():
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.n_subjs = args.n_subjs
        if args.s_num == 'all':
            self.sid = args.s_num
        else:
            self.sid = str(int(args.s_num)).zfill(2)
        self.predict_by_feature = args.predict_by_feature
        self.model_by_feature = args.model_by_feature
        self.control = args.control
        self.pca_before_regression = args.pca_before_regression
        self.predict_features = args.predict_features
        self.predict_grouped_features = args.predict_grouped_features
        self.model_features = args.model_features
        self.n_perm = args.n_perm
        if self.model_by_feature:
            assert self.model_features, "features must be defined to model by feature"
            regress_name = 'model-'
            for feature in self.model_features:
                regress_name += feature.capitalize()
            in_predict_name = 'predict-all'
            out_predict_name = 'predict-all'
        elif self.predict_by_feature:
            regress_name = 'model-full'
            assert len(self.predict_features) == 1, "only 1 feature can be defined"
            in_predict_name = 'predict-features'
            out_predict_name = f'predict-{self.predict_features[0]}'
        elif self.predict_grouped_features:
            regress_name = 'model-full'
            in_predict_name = 'predict-'
            for feature in self.predict_features:
                in_predict_name += feature.replace(' ', '').capitalize()
            out_predict_name = in_predict_name
        else:
            regress_name = 'model-full'
            in_predict_name = 'predict-all'
            out_predict_name = 'predict-all'
        self.regress_name = regress_name
        self.in_predict_name = in_predict_name
        self.out_predict_name = out_predict_name
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def load(self):
        base = f'{self.out_dir}/VoxelEncoding/sub-{self.sid}_{self.regress_name}_{self.in_predict_name}_control-{self.control}_pca-{self.pca_before_regression}'
        pred = np.load(f'{base}_y_pred.npy')
        true = np.load(f'{base}_y_true.npy')
        indices = np.load(f'{base}_indices.npy')
        return true, pred, indices

    def load_by_feature(self):
        feature_index = regress.get_feature_inds(self.predict_features)
        true, pred, indices = self.load()
        return true, pred[feature_index, ...].squeeze(), indices

    def save_perm_results(self, r_true, p, r_null):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_{self.regress_name}_{self.out_predict_name}_control-{self.control}_pca_before_regression-{self.pca_before_regression}'
        np.save(f'{base}_rs.npy', r_true)
        np.save(f'{base}_ps.npy', p)
        np.save(f'{base}_rs-nulldist.npy', r_null)
        print('Completed successfully!')

    def run(self):
        if self.predict_by_feature:
            y_true, y_pred, test_inds = self.load_by_feature()
        else:
            y_true, y_pred, test_inds = self.load()
        print(y_true.shape)
        print(y_pred.shape)
        r_true, p, r_null = tools.permutation_test_2d(y_true, y_pred,
                                                      test_inds=test_inds, n_perm=self.n_perm)
        self.save_perm_results(r_true, p, r_null)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str)
    parser.add_argument('--predict_features', '-p', action='append', default=[])
    parser.add_argument('--model_features', '-m', action='append', default=[])
    parser.add_argument('--n_subjs', type=int, default=4)
    parser.add_argument('--predict_by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--predict_grouped_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--control', type=str, default='conv2')
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model_by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()