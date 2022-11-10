#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def scale(train_, test_):
    # first remove activations that have 0 variance
    variance = np.nanstd(train_, axis=0).squeeze()
    train_ = train_[:, np.invert(np.isclose(variance, 0.))]
    test_ = test_[:, np.invert(np.isclose(variance, 0.))]
    print(f'{np.sum(np.isclose(variance, 0.))} channels had a variance of zero')

    # now scale the data
    mean = np.nanmean(train_, axis=0).squeeze()
    variance = np.nanstd(train_, axis=0).squeeze()
    train_ = (train_ - mean) / variance
    return train_, (test_ - mean) / variance


def regress(X_train_, y_train_, X_test_):
    # ols
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_, y_train_)
    return lr.predict(X_test_)


def pca(X_train_, X_test_):
    pca = PCA(whiten=False, n_components=0.95)
    X_train_ = pca.fit_transform(X_train_)
    print(X_train_.shape)
    return X_train_, pca.transform(X_test_)


def preprocess(X_train_, X_test_,
               y_train_, y_test_):
    X_train_, X_test_ = scale(X_train_, X_test_)
    X_train_, X_test_ = pca(X_train_, X_test_)
    y_train_, y_test_ = scale(y_train_, y_test_)
    return X_train_, X_test_, y_train_, y_test_


class FeatureRegression:
    def __init__(self, args):
        self.process = 'FeatureRegression'
        self.feature = args.feature
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file_prefix = f'{self.out_dir}/{self.process}/feature-{self.feature}_motion-energy'
        print(vars(self))

    def load_X(self, dataset):
        return np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{dataset}.npy')

    def load_y(self, dataset):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        df = df.merge(pd.read_csv(f'{self.data_dir}/annotations/{dataset}.csv'))
        return df[self.feature.replace('_', ' ')].to_numpy()

    def load(self):
        X_train_ = self.load_X('train')
        X_test_ = self.load_X('test')
        y_train_ = self.load_y('train')
        y_test_ = self.load_y('test')
        return X_train_, X_test_, y_train_, y_test_

    def save_results(self, y_test, y_pred):
        np.save(f'{self.out_file_prefix}_y-test.npy', y_test)
        np.save(f'{self.out_file_prefix}_y-pred.npy', y_pred)

    def regression(self, X_train_, X_test_, y_train_, y_test_):
        X_train_normed, X_test_normed, y_train_normed, y_test_normed = preprocess(X_train_, X_test_,
                                                                                  y_train_, y_test_)
        y_pred = regress(X_train_normed, y_train_normed, X_test_normed)
        self.save_results(y_test_normed, y_pred)

    def run(self):
        X_train, X_test, y_train, y_test = self.load()
        self.regression(X_train, X_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='expanse')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    FeatureRegression(args).run()


if __name__ == '__main__':
    main()
