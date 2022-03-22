

#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def euclidean_distance(a):
    dist = pdist(a, 'euclidean')
    return dist, squareform(dist)


def correlation_distance(a):
    dist = pdist(a, 'correlation')
    return dist, squareform(dist)


class FeatureRDMs():
    def __init__(self, args):
        self.process = 'FeatureRDMs'
        self.set = args.set
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def load_annotations(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/{self.set}.csv')
        df = df.merge(train)
        df = df.drop(columns=['video_name'])
        return df

    def load_models(self, model='of', layer=None):
        if model == 'of':
            activation = np.load(f'{self.out_dir}/MotionEnergyActivations/motion_energy_set-{self.set}.npy')
        else:
            activation = np.load(f'{self.out_dir}/AlexNetActivations/alexnet_conv{layer}_set-{self.set}_avgframe.npy').T
        return activation

    def plot(self, matrix, feature):
        _, ax = plt.subplots()
        ax.imshow(matrix)
        plt.savefig(f'{self.figure_dir}/{feature}.pdf')

    def run(self):
        df = pd.DataFrame()
        features = self.load_annotations()
        for feature in features.columns:
            arr = np.expand_dims(features[feature], axis=1)
            vector, matrix = euclidean_distance(arr)
            self.plot(matrix, feature)
            np.save(f'{self.out_dir}/{self.process}/{feature}_set-{self.set}.npy', vector)
            df[feature] = vector

        for name, (model, layer) in zip(['motion energy', 'AlexNet conv2', 'AlexNet conv5'],
                                        zip(['of', 'alexnet', 'alexnet'], [None, 2, 5])):
            arr = self.load_models(model=model, layer=layer)
            vector, matrix = correlation_distance(arr)
            self.plot(matrix, name)
            np.save(f'{self.out_dir}/{self.process}/{name}_set-{self.set}.npy', vector)
            df[name] = vector
        df.to_csv(f'{self.out_dir}/{self.process}/rdms_set-{self.set}.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    FeatureRDMs(args).run()

if __name__ == '__main__':
    main()
