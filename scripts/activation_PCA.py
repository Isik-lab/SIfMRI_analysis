#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np

from sklearn.decomposition import PCA
import pandas as pd


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


def pca(X_train_, X_test_, n_PCs):
    pca = PCA(whiten=False, n_components=n_PCs)
    X_train_PCs = pca.fit_transform(X_train_)
    return X_train_PCs, pca.transform(X_test_)


class ActivationPCA:
    def __init__(self, args):
        self.process = 'ActivationPCA'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.model = args.model
        self.out_prefix = ''
        if 'alexnet' in self.model:
            self.n_PCs = 20
        else: #'moten'
            self.n_PCs = 3
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)

    def get_highD_data(self, dataset):
        if 'moten' in self.model:
            X = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{dataset}.npy')
            self.out_prefix = f'{self.out_dir}/{self.process}/moten_PCs'
        else: #if 'alexnet' in self.model:
            X = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv2_set-{dataset}.npy')
            self.out_prefix = f'{self.out_dir}/{self.process}/alexnet_PCs'
        return X

    def save(self, train_, test_):
        np.save(f'{self.out_prefix}_set-train.npy', train_)
        np.save(f'{self.out_prefix}_set-test.npy', test_)

    def save_df(self, train_, test_):
        out_df = []
        for data, split in zip([train_, test_], ['train', 'test']):
            df = pd.read_csv(f'{self.data_dir}/annotations/{split}.csv')
            df.sort_values(by=['video_name'], inplace=True)
            data_df = pd.DataFrame({f'{self.model}_PC{i}': data[:, i] for i in range(data.shape[-1])})
            df = pd.concat([df, data_df], axis=1)
            df['split'] = split
            out_df.append(df)
        out_df = pd.concat(out_df)
        out_df.to_csv(f'{self.out_prefix}.csv', index=False)

    def run(self):
        train = self.get_highD_data('train')
        test = self.get_highD_data('test')
        train, test = scale(train, test)
        train, test = pca(train, test, self.n_PCs)
        self.save_df(train, test)
        print(train.shape)
        print(test.shape)
        self.save(train, test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    ActivationPCA(args).run()


if __name__ == '__main__':
    main()
