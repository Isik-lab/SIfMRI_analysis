#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import pandas as pd
import numpy as np

import regress
from sklearn.model_selection import KFold

class voxelwise_encoding():
    def __init__(self, args):
        self.process = 'voxelwise_encoding'
        self.sid = str(args.s_num).zfill(2)
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')

    def run(self, reliability_thresh=0.279):
        control_model = np.load(f'{self.out_dir}/generate_models/control_model.npy')
        X = np.load(f'{self.out_dir}/generate_models/annotated_model.npy')

        # Get the feature names for the annotated model
        features = pd.read_csv(f'{self.data_dir}/annotations/annotations.csv').columns.to_list()
        features.remove('video_name')
        n_features = len(features)


        # Initialize KF splitter
        kf = KFold(n_splits=10, shuffle=True, random_state=1)

        # load the beta values and filter to reliable voxels
        beta_map = np.load(f'{self.out_dir}/grouped_runs/sub-{self.sid}/sub-{self.sid}_train-data.npy')
        im_arr = np.load(f'{self.out_dir}/subject_reliability/sub-{self.sid}/sub-{self.sid}_stat-rho_statmap.npy')
        beta_map = beta_map[im_arr > reliability_thresh, :]

        print('Starting regression')
        start = time.time()
        true, pred, inds = regress.outer_ridge_2d(X, control_model, beta_map,
                                                           n_features, kf)
        print(f'Finished regression in {(time.time() - start)/60:.2f} minutes')

        ytrue = np.zeros((beta_map.shape[-1], im_arr.shape[0]))
        ytrue[:, im_arr > reliability_thresh] = true
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y-true.npy', ytrue)
        
        ypred = np.zeros((n_features, beta_map.shape[-1], im_arr.shape[0]))
        ypred[:, :, im_arr > reliability_thresh] = pred
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_y-pred.npy', ypred)
        
        np.save(f'{self.out_dir}/{self.process}/sub-{self.sid}_test-inds.npy', inds)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int)
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    args = parser.parse_args()
    times = voxelwise_encoding(args).run()

if __name__ == '__main__':
    main()