#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import LinearRegression
from src import tools


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


def regress(X_train_, y_train_):
    # ols
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_, y_train_)
    return lr.coef_.T


def predict(X_test_, betas_):
    return X_test_ @ betas_


def preprocess(X_train_, X_test_,
               y_train_, y_test_):
    X_train_, X_test_ = scale(X_train_, X_test_)
    y_train_, y_test_ = scale(y_train_, y_test_)
    return X_train_, X_test_, y_train_, y_test_


class CategoryVoxelRegression:
    def __init__(self, args):
        self.process = 'CategoryVoxelRegression'
        self.sid = str(args.s_num).zfill(2)
        self.category = args.category
        self.step = args.step
        self.space = args.space
        self.zscore_ses = args.zscore_ses
        self.smooth = args.smooth
        if self.smooth:
            if self.zscore_ses:
                self.beta_path = 'betas_3mm_zscore'
            else: #self.smoothing and not self.zscore_ses:
                self.beta_path  = 'betas_3mm_nozscore'
        else:
            if self.zscore_ses:
                self.beta_path  = 'betas_0mm_zscore'
            else: #not self.smoothing and not self.zscore_ses
                self.beta_path  = 'betas_0mm_nozscore'
        self.cross_validation = args.CV
        assert not self.cross_validation, "CV not yet implemented"
        self.n_PCs = args.n_PCs
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def load_neural(self, dataset):
        mask = np.load(f'{self.out_dir}/Reliability/sub-{self.sid}_space-{self.space}_desc-test-{self.step}_reliability-mask.npy').astype('bool')
        neural = nib.load(f'{self.data_dir}/{self.beta_path}/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-{dataset}-{self.step}_data.nii.gz')
        neural = np.array(neural.dataobj).reshape((-1, neural.shape[-1])).T
        return neural[:, mask]

    def load_annotated_features(self, dataset):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        set_names = pd.read_csv(f'{self.data_dir}/annotations/{dataset}.csv')
        df = df.merge(set_names)
        df.sort_values(by=['video_name'], inplace=True)
        columns = category_map(self.category)
        df = df[columns]
        return df.to_numpy()

    def load(self):
        X_train_ = self.load_annotated_features('train')
        X_test_ = self.load_annotated_features('test')
        y_train_ = self.load_neural('train')
        y_test_ = self.load_neural('test')
        return X_train_, X_test_, y_train_, y_test_

    def save_results(self, betas_, y_test_, y_pred_):
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}_betas.npy', betas_)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}_y-test.npy', y_test_)
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_category-{self.category}_y-pred.npy', y_pred_)

    def train_test_regression(self, X_train_, X_test_, y_train_, y_test_):
        X_train_, X_test_, y_train, y_test = preprocess(X_train_, X_test_, y_train_, y_test_)
        betas = regress(X_train_, y_train_)
        y_pred = predict(X_test_, betas)
        self.save_results(betas, y_pred, y_test)

    def run(self):
        X_train, X_test, y_train, y_test = self.load()
        self.train_test_regression(X_train, X_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--category', type=str, default='scene_object')
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--zscore_ses', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smooth', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_PCs', type=int, default=8)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    CategoryVoxelRegression(args).run()


if __name__ == '__main__':
    main()
