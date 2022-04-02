#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from src import regress, tools
from sklearn.model_selection import KFold
import nibabel as nib

from src.custom_plotting import feature_categories


class ROIEncoding:
    def __init__(self, args):
        self.process = 'ROIEncoding'
        self.n_subjects = args.n_subjects
        self.set = args.set
        self.n_samples = args.n_samples
        self.roi = args.roi
        self.include_control = args.include_control
        self.layer = args.layer
        if self.include_control:
            assert self.layer is not None, "AlexNet layer must be defined"
            self.control_name = f'conv{self.layer}'
        else:
            self.control_name = 'none'
        self.by_feature = args.by_feature
        self.pca_before_regression = args.pca_before_regression
        self.sid = str(args.s_num).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def load_mask(self):
        file = glob.glob(f'{self.data_dir}/ROI_masks/sub-{self.sid}/sub-{self.sid}*{self.roi}*nooverlap.nii.gz')[0]
        im = nib.load(file)
        mask = np.array(im.dataobj).flatten().astype('bool')
        indices = np.where(mask)[0]
        return indices

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        return df.drop('video_name', axis=1).columns.to_numpy()

    def save_arrays(self, rs, rs_null, rs_var):
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_rs.npy', rs)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_rs_null.npy', rs_null)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_rs_var.npy', rs_null)

    def reorganize_data(self, features, y_pred, y_true, indices):
        categories = feature_categories()
        df = pd.DataFrame()
        rs = np.ones(len(features))
        rs_null = np.ones((len(features), self.n_samples))
        rs_var = np.ones_like(rs_null)
        for ifeature, feature in enumerate(features):
            r, p, r_null = tools.permutation_test(y_true, y_pred[ifeature, :], test_inds=indices)
            r_var = tools.bootstrap(y_pred[ifeature, :], y_true, indices)
            f = pd.DataFrame({'Subjects': [f'sub-{self.sid}'],
                              'ROI': [self.roi],
                              'Features': [feature],
                              'Feature category': [categories[feature]],
                              'Pearson r': [r],
                              'p value': [p],
                              'low sem': [r - r_var.std()],
                              'high sem': [r + r_var.std()],
                              'Explained variance': [r ** 2]})
            # self.plotting_dists(r, p, r_null, f'sub-{sid}_roi-{self.roi}_feature-{feature}')
            df = pd.concat([df, f])
            rs[ifeature] = r
            rs_null[ifeature, :] = r_null
            rs_var[ifeature, :] = r_var
        self.save_arrays(rs, rs_null, rs_var)
        df['Feature category'] = pd.Categorical(df['Feature category'],
                                                categories=['scene', 'object', 'social primitive', 'social'],
                                                ordered=True)
        df.to_csv(f'{self.out_dir}/{self.process}/sub-{self.sid}_roi-{self.roi}_encoding.csv', index=False)

    def run(self, regression_splits=10, random_state=1):
        if self.include_control:
            control_model = np.load(f'{self.out_dir}/GenerateModels/control_model_conv{self.layer}_set-{self.set}.npy')
        else:
            control_model = None
        X = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-{self.set}.npy')

        # Get the feature names for the annotated model
        features = self.load_features()
        n_features = len(features)

        # Initialize KF splitter
        kf = KFold(n_splits=regression_splits, shuffle=True, random_state=random_state)

        # load the beta values and mask to reliable voxels
        beta_map = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        indices = self.load_mask()
        y = beta_map[indices, :].mean(axis=0)

        # Run the regression and print out the timing
        print('Starting regression')
        start = time.time()
        y_true, y_pred, indices = regress.cross_validated_ridge(X, control_model, y,
                                                                n_features, kf,
                                                                include_control=self.include_control,
                                                                by_feature=self.by_feature,
                                                                pca_before_regression=self.pca_before_regression)
        self.reorganize_data(features, y_pred, y_true, indices)
        print(f'Finished regression in {(time.time() - start) / 60:.2f} minutes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--roi', '-r', type=str, default='pSTS')
    parser.add_argument('--layer', '-l', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--include_control', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--by_feature', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pca_before_regression', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_subjects', '-n', type=int, default=4)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    ROIEncoding(args).run()


if __name__ == '__main__':
    main()
