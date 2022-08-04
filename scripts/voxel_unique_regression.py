#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
    scaler = StandardScaler()
    train_ = scaler.fit_transform(train_)
    mean = scaler.mean_[:n_features].squeeze()
    var = scaler.var_[:n_features].squeeze()
    var[np.isclose(var, 0.)] = np.nan
    return train_, (test_ - mean) / var


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


class VoxelUniqueRegression:
    def __init__(self, args):
        self.process = 'VoxelUniqueRegression'
        self.sid = str(args.s_num).zfill(2)
        print(f'sub-{self.sid}')
        self.cross_validation = args.cross_validation
        self.n_PCs = args.n_PCs
        self.n_annotated_features = 9
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def mk_models(self):
        models = {'all': None,
                  'visual': [0, 1, 2],
                  'primitive': [3, 4],
                  'social': [5, 6],
                  'affective': [7, 8]}
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        features.drop(columns=['cooperation', 'dominance', 'intimacy'], inplace=True)
        features = features.sort_values(by=['video_name']).drop(columns=['video_name']).columns.to_numpy()
        inds = np.arange(self.n_annotated_features + self.n_PCs)
        # This makes the visual model the annotated visual dimensions and the highlevel
        models['nuissance'] = list(np.delete(inds,
                                             [j for i in models if models[i] is not None for j in models[i]]))
        models['lowhighvis'] = list(np.delete(inds,
                                                  [models['primitive'], models['social'], models['affective']]))
        models['annotated'] = list(np.arange(self.n_annotated_features))
        for ifeature, feature in enumerate(features):
            models[feature] = [ifeature]
        return models

    def preprocess(self, X_train_, X_test_,
                   X_control_train_, X_control_test_,
                   y_train_, y_test_):
        X_control_train_PCs_, X_control_test_PCs_ = pca(X_control_train_, X_control_test_,
                                                        n_components=self.n_PCs)
        X_train_ = np.hstack([X_train_, X_control_train_PCs_])
        X_test_ = np.hstack([X_test_, X_control_test_PCs_])
        X_train_, X_test_ = scale(X_train_, X_test_)
        y_train_, y_test_ = scale(y_train_, y_test_)
        return X_train_, X_test_, y_train_, y_test_

    def prediction(self, X_test_, betas_, y_test_, i=None):
        if i is not None:
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_betas_method-CV_loop-{i}.npy', betas_)
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-CV_loop-{i}.npy', y_test_)
        else:
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_betas_method-test.npy', betas_)
            np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-test.npy', y_test_)

        models = self.mk_models()
        for key in models:
            cur_betas_ = zero_inds(betas_, models[key], X_test_.shape[-1])
            y_pred = X_test_ @ cur_betas_

            if i is not None:
                np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{key}_method-CV_loop-{i}.npy',
                        y_pred)
            else:
                np.save(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{key}_method-test.npy', y_pred)

    def load_neural(self):
        mask = np.load(f'{self.out_dir}/Reliability/sub-{self.sid}_set-test_reliability-mask.npy').astype('bool')
        train = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        test = np.load(f'{self.out_dir}/GroupRuns/sub-{self.sid}/sub-{self.sid}_test-data.npy')
        return train[:, mask], test[:, mask]

    def load_nuissance_regressors(self, dataset):
        alexnet = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{dataset}_avgframe.npy').T
        of = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{dataset}.npy')
        return np.hstack([alexnet, of])

    def load_features(self):
        X_control_train_ = self.load_nuissance_regressors('train')
        X_control_test_ = self.load_nuissance_regressors('test')
        X_train_ = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-train.npy')
        X_test_ = np.load(f'{self.out_dir}/GenerateModels/annotated_model_set-test.npy')
        return X_train_, X_test_, X_control_train_, X_control_test_

    def load(self):
        X_train_, X_test_, X_control_train_, X_control_test_ = self.load_features()
        y_train_, y_test_ = self.load_neural()
        if self.cross_validation:
            return X_train_, X_control_train_, y_train_
        else:
            return X_train_, X_test_, X_control_train_, X_control_test_, y_train_, y_test_

    def cross_validated_regression(self, X_, X_control_, y_,
                                   n_splits=10, random_state=0):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X_)), total=n_splits):
            X_train_, X_test_ = X_[train_index], X_[test_index]
            y_train_, y_test_ = y_[train_index], y_[test_index]
            X_control_train_, X_control_test_ = X_control_[train_index], X_control_[test_index]

            # Preprocess
            X_train_, X_test_, y_train_, y_test_ = self.preprocess(X_train_, X_test_,
                                                                   X_control_train_, X_control_test_,
                                                                   y_train_, y_test_)

            # Regression
            betas = regress(X_train_, y_train_)

            # predictions
            self.prediction(X_test_, betas, y_test_, i)

    def train_test_regression(self, X_train_, X_test_,
                              X_control_train_, X_control_test_,
                              y_train_, y_test_):
        # Preprocess
        X_train_, X_test_, y_train, y_test = self.preprocess(X_train_, X_test_,
                                                             X_control_train_, X_control_test_,
                                                             y_train_, y_test_)

        # Regression
        betas = regress(X_train_, y_train_)

        # predictions
        self.prediction(X_test_, betas, y_test_)

    def run(self):
        if self.cross_validation:
            X, X_control, y = self.load()
            self.cross_validated_regression(X, X_control, y)
        else:
            X_train, X_test, X_control_train, X_control_test, y_train, y_test = self.load()
            self.train_test_regression(X_train, X_test, X_control_train, X_control_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_PCs', type=int, default=8)
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