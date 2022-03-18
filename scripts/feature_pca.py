#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(X):
    model = PCA(whiten=True)
    model.fit(X)
    return model.components_, model.explained_variance_ratio_


class FeaturePCA():
    def __init__(self, args):
        self.process = 'FeaturePCA'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def load_features(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name'], inplace=True)
        feature_names = np.array(df.columns)
        return feature_names, df.to_numpy()

    def summarize_results(self, components, explained_variance, feature_names):
        df = pd.DataFrame()
        n_PCs = components.shape[0]
        n_features = components.shape[1]
        for iPC in range(n_PCs):
            indices = np.flip(np.argsort(components[iPC, :]))
            ordered_features = feature_names[indices]
            ordered_components = components[iPC, indices]
            d = {'Explained variance': [explained_variance[iPC]]}
            for ifeature in range(n_features):
                d[f'Feature{str(ifeature+1).zfill(2)}'] = ordered_features[ifeature]
            for ifeature in range(n_features):
                d[f'Feature{str(ifeature+1).zfill(2)}_weight'] = ordered_components[ifeature]
            df = pd.concat([df, pd.DataFrame(d)])
        return df.reset_index(drop=True)

    def plot(self, vals):
        print()
        fig, ax = plt.subplots()
        ax.plot(vals, '.-r')
        ax.plot(np.ones(len(vals))*0.8, 'k', alpha=0.3)
        ax.plot(np.ones(11) * np.sum(vals <= 0.8), np.arange(0, 1.1, 0.1), '--k')
        ax.plot(np.ones(len(vals)) * 0.95, 'k', alpha=0.3)
        ax.plot(np.ones(11) * np.sum(vals <= 0.95), np.arange(0, 1.1, 0.1), '--k')
        plt.ylim([0, 1.05])
        plt.xlabel('PCs')
        plt.ylabel('Explained variance')
        plt.savefig(f'{self.figure_dir}/explained_variance.pdf')

    def run(self):
        feature_names, X = self.load_features()
        components, explained_variance = pca(X)
        self.plot(explained_variance.cumsum())
        df = self.summarize_results(components, explained_variance, feature_names)
        print(df.head())
        df.to_csv(f'{self.out_dir}/PCs.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeaturePCA(args).run()

if __name__ == '__main__':
    main()
