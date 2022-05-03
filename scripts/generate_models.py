#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

class GenerateModels():
    def __init__(self, args):
        self.process = 'GenerateModels'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}'
        self.layer = args.layer
        self.set = args.set
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def control_model(self):
        alexnet = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv{self.layer}_set-{self.set}_avgframe.npy')
        of = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{self.set}.npy')
        control_model = np.hstack([alexnet.T, of])

        pca = PCA(svd_solver='full')
        pca.fit(control_model)
        _, ax = plt.subplots(2, 1)
        ax[0].plot(pca.explained_variance_ratio_.cumsum())
        ax[1].plot(pca.explained_variance_ratio_, '.')
        plt.savefig(f'{self.figure_dir}/alexnet_and_motion_set-{self.set}_pcs.pdf')
        
        # Combine
        pca = PCA(svd_solver='full', n_components=15)
        pca.fit(control_model)
        np.save(f'{self.out_dir}/{self.process}/control_model_conv{self.layer}_set-{self.set}.npy', control_model)
    
    def annotated_model(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        set_names = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(set_names)
        print(df.head())
        df.sort_values(by=['video_name'], inplace=True)
        df.drop(columns=['video_name'], inplace=True)
        model = df.to_numpy()
        np.save(f'{self.out_dir}/{self.process}/annotated_model_set-{self.set}.npy', model)
        
    def run(self):
        self.control_model()
        self.annotated_model()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', '-l', type=str, default='2')
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    GenerateModels(args).run()

if __name__ == '__main__':
    main()

