import argparse
import os
import glob
from natsort import natsorted
from tqdm import tqdm
import itertools

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import zscore

import matplotlib.pyplot as plt
import seaborn as sns

class generate_models():
    def __init__(self, args):
        self.process = 'generate_models'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def control_model(self):
        # AlexNet
        layer = 2
        alexnet = zscore(np.load(f'{self.out_dir}/alexnet_activations/alexnet_conv{layer}_avgframe.npy'), axis=-1)

        pca = PCA(svd_solver='full', n_components=20)
        alexnet = pca.fit_transform(alexnet.T)

        fig, ax = plt.subplots()
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.savefig(f'{self.figure_dir}/alexnet_pcs.pdf')
        
        # Optical flow
        of = zscore(np.load(f'{self.out_dir}/of_activations/of_adelsonbergen.npy'), axis=0)
        pca = PCA(svd_solver='full', n_components=20)
        of = pca.fit_transform(of)

        fig, ax = plt.subplots()
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.savefig(f'{self.figure_dir}/of_pcs.pdf')
        
        # Combine
        control_model = np.concatenate((alexnet, of), axis=-1)
        np.save(f'{self.out_dir}/{self.process}/control_model.npy', control_model)
    
    def annotated_model(self):
        df = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv')
        train = pd.read_csv(f'{self.data_dir}/annotations/train.csv')
        df = df.merge(train)
        df.sort_values(by=['video_name'], inplace=True)
        
        feature_categories = ['visual', 'visual', 'visual', 'visual', 'visual',
                     'social detection', 'social detection', 
                     'social evaluation', 'social evaluation', 'social evaluation', 'social evaluation', 'social evaluation']
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
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    times = generate_models(args).run()

if __name__ == '__main__':
    main()

