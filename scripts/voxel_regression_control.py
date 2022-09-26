#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def pca(a, b=None, n_components=8):
    pca_ = PCA(svd_solver='full', whiten=True, n_components=n_components)
    a_out = pca_.fit_transform(a)
    if b is not None:
        b_out = pca_.transform(b)
    else:
        b_out = None
    return a_out, b_out


def scale(train_, test_):
    n_features = test_.shape[-1]
    mean = np.nanmean(train_, axis=0).squeeze()
    var = np.nanstd(train_, axis=0).squeeze()
    var[np.isclose(var, 0.)] = np.nan
    train_ = (train_ - mean) / var
    return train_, (test_ - mean[:n_features]) / var[:n_features]


def zero_inds(arr, inds_, max_val):
    out = arr.copy()
    if inds_ is not None:
        for i in range(max_val):
            if i not in inds_:
                out[i, :] = 0.
    return out


def regress(X_train_, y_train_):
    # ols
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_, y_train_)
    return lr.coef_.T


class VoxelRegressionControl:
    def __init__(self, args):
        self.process = 'VoxelRegressionControl'
        self.sid = str(args.s_num).zfill(2)
        self.step = args.step
        self.space = args.space
        self.unique_model = args.unique_model
        self.single_model = args.single_model
        assert (self.unique_model is None or self.single_model is None) or (
                    self.unique_model is None and self.single_model is None)
        if self.unique_model is not None:
            self.unique_model = self.unique_model.replace('_', ' ')
        if self.single_model is not None:
            self.single_model = self.single_model.replace('_', ' ')
        self.cross_validation = args.CV
        self.n_PCs = args.n_PCs
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def mk_models(self):
        if self.unique_model is not None:
            models = {'all': None}
        elif self.single_model is not None:
            models = {'all': [0]}
        else:
            models = {'all': None}
        return models

    def preprocess(self, X_train_, X_test_,
                   y_train_, y_test_):
        X_train_PCs, X_test_PCs = pca(X_train_, X_test_,
                                      n_components=self.n_PCs)
        X_train_, X_test_ = scale(X_train_PCs, X_test_PCs)
        y_train_, y_test_ = scale(y_train_, y_test_)
        return X_train_, X_test_, y_train_, y_test_

    def prediction(self, X_test_, betas_, y_test_, i=None):
        if (self.unique_model is None) and (self.single_model is None):
            if i is not None:
                np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_betas_method-CV_loop-{i}.npy', betas_)
                np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y-test_method-CV_loop-{i}.npy', y_test_)
            else:
                np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_betas_method-test.npy', betas_)
                np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y-test_method-test.npy', y_test_)

        models = self.mk_models()
        for key in models:
            name_base = f'{self.out_dir}/{self.process}/sub-{self.sid}_prediction-{key}'
            cur_betas_ = zero_inds(betas_, models[key], X_test_.shape[-1])
            y_pred = X_test_ @ cur_betas_

            if i is not None:
                np.save(f'{name_base}_drop-{self.unique_model}_single-{self.single_model}_method-CV_loop-{i}.npy',
                        y_pred)
            else:
                np.save(f'{name_base}_drop-{self.unique_model}_single-{self.single_model}_method-test.npy', y_pred)
                print(f'{name_base}_drop-{self.unique_model}_single-{self.single_model}_method-test.npy')

    def load_neural(self):
        mask = np.load(
            f'{self.out_dir}/Reliability/sub-{self.sid}_space-{self.space}_desc-test-{self.step}_reliability-mask.npy').astype(
            'bool')
        train = nib.load(
            f'{self.data_dir}/betas/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-train-{self.step}_data.nii.gz')
        train = np.array(train.dataobj).reshape((-1, train.shape[-1])).T
        test = nib.load(
            f'{self.data_dir}/betas/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-test-{self.step}_data.nii.gz')
        test = np.array(test.dataobj).reshape((-1, test.shape[-1])).T
        return train[:, mask], test[:, mask]

    def load_nuissance_regressors(self, dataset):
        alexnet = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{dataset}_avgframe.npy').T
        # of = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{dataset}.npy')
        return alexnet  # np.hstack([alexnet, of])

    def load_features(self):
        X_train_ = self.load_nuissance_regressors('train')
        X_test_ = self.load_nuissance_regressors('test')
        return X_train_, X_test_

    def load(self):
        X_train_, X_test_ = self.load_features()
        y_train_, y_test_ = self.load_neural()
        if self.cross_validation:
            return X_train_, y_train_
        else:
            return X_train_, X_test_, y_train_, y_test_

    def cross_validated_regression(self, X_, y_,
                                   n_splits=10, random_state=0):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X_)), total=n_splits):
            X_train_, X_test_ = X_[train_index], X_[test_index]
            y_train_, y_test_ = y_[train_index], y_[test_index]

            # Preprocess
            X_train_, X_test_, y_train_, y_test_ = self.preprocess(X_train_, X_test_,
                                                                   y_train_, y_test_)

            # Regression
            betas = regress(X_train_, y_train_)

            # predictions
            self.prediction(X_test_, betas, y_test_, i)

    def train_test_regression(self, X_train_, X_test_,
                              y_train_, y_test_):
        # Preprocess
        X_train_, X_test_, y_train, y_test = self.preprocess(X_train_, X_test_,
                                                             y_train_, y_test_)

        # Regression
        betas = regress(X_train_, y_train_)

        # predictions
        self.prediction(X_test_, betas, y_test_)

    def run(self):
        if self.cross_validation:
            X, y = self.load()
            self.cross_validated_regression(X, y)
        else:
            X_train, X_test, y_train, y_test = self.load()
            self.train_test_regression(X_train, X_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--unique_model', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_PCs', type=float, default=8)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelRegressionControl(args).run()


if __name__ == '__main__':
    main()
