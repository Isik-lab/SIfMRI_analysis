#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def outer_ridge(X_train_, y_train_, alpha):
    # Fit the Ridge model with the selected alpha
    lr = Ridge(fit_intercept=False, alpha=alpha)
    lr.fit(X_train_, y_train_)
    return lr.coef_


def pca(X_train_, n_components=40):
    pca_ = PCA(svd_solver='full', whiten=True, n_components=n_components)
    return pca_.fit_transform(X_train_)


def inner_ridge(X_train_, y_train_, n_splits=4,
                random_state=0):
    # Find clf.alpha_ with RidgeCV
    alphas = 10. ** np.arange(start=-1., stop=6.)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = RidgeCV(cv=kf, scoring="r2", alphas=alphas)
    clf.fit(X_train_, y_train_)
    return clf.alpha_


def scale(train_, test_):
    n_features = test_.shape[-1]
    scaler = StandardScaler()
    train_ = scaler.fit_transform(train_)
    mean = scaler.mean_[:n_features].squeeze()
    var = scaler.var_[:n_features].squeeze()
    var[np.isclose(var, 0.)] = np.nan
    return train_, (test_ - mean) / var


def preprocess(X_train_, X_test_, X_control_, y_train_, y_test_):
    X_control_PCs_ = pca(X_control_)
    X_train_ = np.hstack([X_train_, X_control_PCs_])
    X_train_, X_test_ = scale(X_train_, X_test_)
    y_train_, y_test_ = scale(y_train_, y_test_)
    return X_train_, X_test_, y_train_, y_test_


def zero_inds(arr, inds_, max_val):
    for i in range(max_val):
        if i not in inds_:
            arr[:, i] = 0.
    return arr


def regress(X_train_, y_train_, n_features):
    # inner ridge for alpha
    alpha = inner_ridge(X_train_, y_train_)
    print(alpha)

    # outer ridge
    betas_ = outer_ridge(X_train_, y_train_, alpha)
    betas_ = betas_[:, :n_features].T
    return betas_


class VoxelRegression():
    def __init__(self, args):
        self.process = 'VoxelRegression'
        self.sid = str(args.s_num).zfill(2)
        self.cross_validation = args.cross_validation
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def mk_models(self):
        models = {'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  'visual': [0, 1, 2],
                  'socialprimitive': [3, 4],
                  'social': [5, 6, 7, 8, 9, 10, 11]}
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        features = features.sort_values(by=['video_name']).drop(columns=['video_name']).columns.to_numpy()
        for ifeature, feature in enumerate(features):
            models[feature] = [ifeature]
        return models

    def prediction(self, X_test_, betas_, y_test_, i=None):
        if i is not None:
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_betas_method-CV_loop-{i}.npy', betas_)
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-CV_loop-{i}.npy', y_test_)
        else:
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_betas_method-test.npy', betas_)
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-test.npy', y_test_)

        models = self.mk_models()
        for key in models:
            cur_X_test_ = zero_inds(X_test_, models[key], X_test_.shape[-1])
            cur_betas_ = zero_inds(betas_, models[key], X_test_.shape[-1])
            y_pred = cur_X_test_ @ cur_betas_

            if i is not None:
                np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{key}_method-CV_loop-{i}.npy',
                        y_pred)
            else:
                np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{key}_method-test.npy', y_pred)

    def load_neural(self):
        mask = np.load(f'{self.out_dir}/Reliability/sub-all_set-test_reliability-mask.npy').astype('bool')
        train = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        test = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_test-data.npy')
        return train[:, mask], test[:, mask]

    def load_features(self):
        alexnet = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-train_avgframe.npy').T
        of = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-train.npy')
        X_train_ = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-train.npy')
        X_test_ = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-test.npy')
        return X_train_, X_test_, np.hstack([alexnet, of])

    def load(self):
        X_train_, X_test_, X_control_ = self.load_features()
        y_train_, y_test_ = self.load_neural()
        if self.cross_validation:
            return X_train_, X_control_, y_train_
        else:
            return X_train_, X_test_, X_control_, y_train_, y_test_

    def cross_validated_regression(self, X_, X_control_, y_,
                                   n_splits=10, random_state=0):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X_)), total=n_splits):
            X_train_, X_test_ = X_[train_index], X_[test_index]
            y_train_, y_test_ = y_[train_index], y_[test_index]

            # Preprocess
            X_train_, X_test_, y_train_, y_test_ = preprocess(X_train_, X_test_, X_control_[train_index],
                                                              y_train_, y_test_)

            # Regression
            betas = regress(X_train_, y_train_, X_test_.shape[-1])

            # predictions
            self.prediction(X_test_, betas, y_test_, i)

    def train_test_regression(self, X_train_, X_test_, X_control_, y_train_, y_test_):
        # Preprocess
        X_train_, X_test_, y_train, y_test = preprocess(X_train_, X_test_, X_control_, y_train_, y_test_)

        # Regression
        betas = regress(X_train_, y_train_, X_test_.shape[-1])

        # predictions
        self.prediction(X_test_, betas, y_test_)

    def run(self):
        if self.cross_validation:
            X, X_control, y = self.load()
            self.cross_validated_regression(X, X_control, y)
        else:
            X_train, X_test, X_control, y_train, y_test = self.load()
            self.train_test_regression(X_train, X_test, X_control, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    VoxelRegression(args).run()


if __name__ == '__main__':
    main()
