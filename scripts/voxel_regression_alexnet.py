#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
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
    pca = PCA(whiten=False)
    X_train_ = pca.fit_transform(X_train_)
    return X_train_, pca.transform(X_test_)


def preprocess(X_train_, X_test_,
               y_train_, y_test_):
    X_train_, X_test_ = scale(X_train_, X_test_)
    X_train_, X_test_ = pca(X_train_, X_test_)
    y_train_, y_test_ = scale(y_train_, y_test_)
    return X_train_, X_test_, y_train_, y_test_


class VoxelRegressionAlexNet:
    def __init__(self, args):
        self.process = 'VoxelRegressionAlexNet'
        self.sid = str(args.s_num).zfill(2)
        self.layer = args.layer
        self.step = args.step
        self.space = args.space
        self.zscore_ses = args.zscore_ses
        self.smooth = args.smooth
        if self.smooth:
            if self.zscore_ses:
                self.beta_path = 'betas_3mm_zscore'
            else:  # self.smoothing and not self.zscore_ses:
                self.beta_path = 'betas_3mm_nozscore'
        else:
            if self.zscore_ses:
                self.beta_path = 'betas_0mm_zscore'
            else:  # not self.smoothing and not self.zscore_ses
                self.beta_path = 'betas_0mm_nozscore'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file_prefix = f'{self.out_dir}/{self.process}/sub-{self.sid}_alexnet-conv{self.layer}'
        print(vars(self))

    def load_X(self, dataset):
        return np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv{self.layer}_set-{dataset}_avgframe.npy')

    def load_y(self, dataset):
        mask = np.load(
            f'{self.out_dir}/Reliability/sub-{self.sid}_space-{self.space}_desc-test-{self.step}_reliability-mask.npy').astype(
            'bool')
        neural = nib.load(
            f'{self.data_dir}/{self.beta_path}/sub-{self.sid}/sub-{self.sid}_space-{self.space}_desc-{dataset}-{self.step}_data.nii.gz')
        neural = np.array(neural.dataobj).reshape((-1, neural.shape[-1])).T
        return neural[:, mask]

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
        X_train_, X_test_, y_train, y_test = preprocess(X_train_, X_test_,
                                                             y_train_, y_test_)
        y_pred = regress(X_train_, y_train_, X_test_)
        self.save_results(y_test, y_pred)

    def run(self):
        X_train, X_test, y_train, y_test = self.load()
        self.regression(X_train, X_test, y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=2)
    parser.add_argument('--step', type=str, default='fracridge')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--layer', type=int, default='3')
    parser.add_argument('--zscore_ses', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smooth', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelRegressionAlexNet(args).run()


if __name__ == '__main__':
    main()
