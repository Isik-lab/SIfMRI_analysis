#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import nibabel as nib
import src.custom_plotting as cm

class ROIEncoding():
    def __init__(self, args):
        self.process = 'ROIEncoding'
        self.data_dir = args.data_dir
        self.out_dir = f'{args.out_dir}/{self.process}'
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

    def get_ROIactivation(self, path, z_map):
        # activity in ROI
        mask = nib.load(path)
        mask = np.array(mask.dataobj, dtype='bool').flatten()
        roi_activation = z_map[mask, :]

        # Remove nan values (these are voxels that do not vary across the different videos)
        inds = ~np.any(np.isnan(roi_activation), axis=1)
        roi_activation = roi_activation[inds, ...]
        return roi_activation.mean(axis=0)

    def reorganize_data(self, features):
        categories = cm.feature_categories()
        for ifeature, features in enumerate(features):
            r_var = bootstrap(y_pred[ifeature, :, :].flatten(),
                              y_true.flatten(), test_inds, n_samples=null_samples)
            df = df.append({'Subjects': f'sub-{sid}',
                            'Features': feature,
                            'Feature category': category,
                            'ROIs': roi_name,
                            'Pearson r': r,
                            'p value': p,
                            'low sem': r-r_var.std(),
                            'high sem': r+r_var.std(),
                            'Explained variance': r ** 2},
                            ignore_index=True)
            rs[sid_, iroi, ifeature] = r
            rs_null[sid_, iroi, ifeature, :] = r_null
            rs_var[sid_, iroi, ifeature, :] = r_var

    def run(self):
        #Do it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    ROIEncoding(args).run()

if __name__ == '__main__':
    main()
