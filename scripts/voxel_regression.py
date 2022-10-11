#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import LinearRegression


def category_map(category=None):
    d = dict()
    d['scene_object'] = ['indoor', 'expanse', 'transitivity']
    d['social_primitive'] = ['agent distance', 'facingness']
    d['social'] = ['joint action', 'communication']
    d['affective'] = ['valence', 'arousal']
    return d[category]


def scale(train_, test_):
    mean = np.nanmean(train_, axis=0).squeeze()
    variance = np.nanstd(train_, axis=0).squeeze()
    variance[np.isclose(variance, 0.)] = np.nan
    train_ = (train_ - mean) / variance
    return train_, (test_ - mean) / variance


def regress(X_train_, y_train_, X_test_):
    # ols
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_, y_train_)
    return lr.predict(X_test_)


def preprocess(X_train_, X_test_,
               y_train_, y_test_):
    X_train_, X_test_ = scale(X_train_, X_test_)
    y_train_, y_test_ = scale(y_train_, y_test_)
    return X_train_, X_test_, y_train_, y_test_


class VoxelRegression:
    def __init__(self, args):
        self.process = 'VoxelRegression'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.feature = args.feature
        self.unique_variance = args.unique_variance
        self.step = args.step
        self.space = args.space
        self.zscore_ses = args.zscore_ses
        self.smooth = args.smooth
        if self.smooth:
            if self.zscore_ses:
                self.beta_path = 'betas_3mm_zscore'
            else: #self.smoothing and not self.zscore_ses:
                self.beta_path = 'betas_3mm_nozscore'
        else:
            if self.zscore_ses:
                self.beta_path = 'betas_0mm_zscore'
            else: #not self.smoothing and not self.zscore_ses
                self.beta_path = 'betas_0mm_nozscore'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file_prefix = ''
        print(vars(self))

    def load_neural(self, dataset):
        mask = np.load(f'{self.out_dir}/Reliability/sub-{self.sid}_space-{self.space}_desc-test-{self.step}_reliability-mask.npy').astype('bool')
        neural = nib.load(f'{self.data_dir}/{self.beta_path}/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-{dataset}-{self.step}_data.nii.gz')
        neural = np.array(neural.dataobj).reshape((-1, neural.shape[-1])).T
        return neural[:, mask]

    def get_regression_features(self, df):
        if self.unique_variance:
            if self.category is not None:
                columns = df.columns.to_list()
                [columns.remove(i) for i in category_map(self.category)]
                self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_dropped-category-{self.category}'
            else: #self.feature is not None:
                columns = df.columns.to_list()
                columns.remove(self.feature)
                self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_dropped-feature-{self.feature}'
        else: #not self.unique_variance
            if self.category is not None:
                # Regression with the categories without the other regressors
                columns = category_map(self.category)
                self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}'
            elif self.feature is not None:
                columns = [self.feature.replace('_', ' ')]
                self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_feature-{self.feature}'
            else:  # This is the full regression model with all annotated features
                columns = df.columns.to_list()
                self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_all-features'
        return columns

    def load_annotated_features(self, dataset):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        set_names = pd.read_csv(f'{self.data_dir}/annotations/{dataset}.csv')
        df = df.merge(set_names)
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name', 'cooperation',
                         'dominance', 'intimacy'], inplace=True)
        columns = self.get_regression_features(df)
        print(columns)
        df = df[columns]
        return df.to_numpy()

    def load(self):
        X_train_ = self.load_annotated_features('train')
        X_test_ = self.load_annotated_features('test')
        y_train_ = self.load_neural('train')
        y_test_ = self.load_neural('test')
        return X_train_, X_test_, y_train_, y_test_

    def save_results(self, y_test_, y_pred_):
        np.save(f'{self.out_file_prefix}_y-test.npy', y_test_)
        np.save(f'{self.out_file_prefix}_y-pred.npy', y_pred_)

    def regression(self, X_train_, X_test_, y_train_, y_test_):
        X_train_, X_test_, y_train, y_test = preprocess(X_train_, X_test_, y_train_, y_test_)
        y_pred = regress(X_train_, y_train_, X_test_)
        self.save_results(y_pred, y_test)

    def run(self):
        X_train, X_test, y_train, y_test = self.load()
        self.regression(X_train, X_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--unique_variance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--zscore_ses', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--smooth', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelRegression(args).run()


if __name__ == '__main__':
    main()
