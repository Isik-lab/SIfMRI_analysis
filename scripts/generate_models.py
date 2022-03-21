#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import zscore

import matplotlib.pyplot as plt

class generate_models():
    def __init__(self, args):
        self.process = 'generate_models'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}'
        self.layer = args.layer
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def control_model(self):
        # AlexNet
        alexnet = np.load(f'{self.out_dir}/alexnet_activations/alexnet_conv{self.layer}_avgframe.npy')
        np.save(f'{self.out_dir}/alexnet_activations/alexnet_conv{self.layer}_avg.npy', alexnet.mean(axis=0))
        alexnet = zscore(alexnet, axis=-1)

        pca = PCA(svd_solver='full', n_components=20)
        alexnet = pca.fit_transform(alexnet.T)

        _, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_.cumsum())
        plt.savefig(f'{self.figure_dir}/alexnet_conv{self.layer}_pcs.pdf')
        
        # Optical flow
        of = np.load(f'{self.out_dir}/of_activations/of_adelsonbergen.npy')
        np.save(f'{self.out_dir}/of_activations/of_adelsonbergen_avg.npy', of.mean(axis=1))
        of = zscore(of, axis=0)
        pca = PCA(svd_solver='full', n_components=20)
        of = pca.fit_transform(of)

        _, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_.cumsum())
        plt.savefig(f'{self.figure_dir}/of_pcs.pdf')
        
        # Combine
        control_model = np.concatenate((alexnet, of), axis=-1)
        np.save(f'{self.out_dir}/{self.process}/control_model_conv{self.layer}.npy', control_model)
    
    def annotated_model(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        features = df.columns.to_list()
        features.remove('video_name')
        
        model = []
        for feature in features:
            arr = zscore(df[feature].to_numpy())
            arr = np.expand_dims(arr, axis=0)
            if type(model) == list:
                model = arr.copy()
            else:
                model = np.append(model, arr, axis=0) 
        np.save(f'{self.out_dir}/{self.process}/annotated_model.npy', model.T)
        
    def run(self):
        self.control_model()
        self.annotated_model()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=str)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    times = generate_models(args).run()

if __name__ == '__main__':
    main()

