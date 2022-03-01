#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
from natsort import natsorted
from tqdm import tqdm
import itertools

import pandas as pd
import numpy as np

from nilearn import image
import matplotlib.pyplot as plt
import seaborn as sns

class roi_reliability():
    def __init__(self, args):
        self.process = 'roi_reliability'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        if not os.path.exists(f'{self.out_dir}/{self.process}'):
            os.mkdir(f'{self.out_dir}/{self.process}')
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)
            
            
    def run(self):
        rois = ['EVC', 'MT', 'EBA', 'PPA', 'TOS', 'RSC', 'LOC',  'FFA', 'OFA', '-STS', 'biomotion', 'SIpSTS', 'TPJ', 'DMPFC']
        roi_names = ['EVC', 'MT', 'EBA', 'PPA', 'OPA', 'RSC', 'LOC',  'FFA', 'OFA', 'faceSTS', 'biomotion', 'SIpSTS', 'TPJ', 'DMPFC']
        n_subjs = 4
        
        reliability = np.zeros((len(rois), n_subjs))
        for sid_ in range(n_subjs):
            sid = str(sid_+1).zfill(2)
            r_map = np.load(f'{self.out_dir}/subject_reliability/sub-{sid}/sub-{sid}_stat-rho_statmap.npy')
            variance = np.load(f'{self.out_dir}/grouped_runs/sub-{sid}/sub-{sid}_test-data.npy').mean(axis=-1).std(axis=-1)
                            
            for iroi, (roi, roi_name) in enumerate(zip(rois, roi_names)):
                #activity in ROI
                print(f'{self.data_dir}/ROI_masks/sub-{sid}/*{roi}*nooverlap.nii.gz')
                mask = image.load_img(glob.glob(f'{self.data_dir}/ROI_masks/sub-{sid}/*{roi}*nooverlap.nii.gz')[0])
                mask = np.array(mask.dataobj, dtype='bool').flatten()
                roi_r = r_map[mask]

                #Remove nan values (these are voxels that do not vary across the different videos)
                inds = ~np.isnan(variance[mask])
                reliability[iroi, sid_] = roi_r[inds].mean()
                            
        subjects = [f'sub-{i+1:02d}' for i in range(n_subjs)]

        noise_ceiling = pd.DataFrame(reliability.T, columns=roi_names, index=subjects)
        noise_ceiling.reset_index(inplace=True)
        noise_ceiling.rename(columns={'index': 'Subjects'}, inplace=True)
        noise_ceiling = pd.melt(noise_ceiling, id_vars='Subjects', 
                 var_name='ROIs',
                 value_vars=roi_names, value_name='Pearson r')

        noise_ceiling['Explained variance'] = noise_ceiling['Pearson r']**2
        noise_ceiling.to_csv(f'{self.out_dir}/{self.process}/noise_ceiling.csv', index=False)
                            
        sns.set(style='white', context='paper')
        fig, ax = plt.subplots()
        sns.barplot(x='ROIs', y='Pearson r',
                data=noise_ceiling, ax=ax, color='#ADDDF2')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        sns.despine(left=True)
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}/roi_noise_ceiling.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/input_data')
    parser.add_argument('--out_dir', '-output', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/output_data')
    parser.add_argument('--figure_dir', '-figures', type=str, default='/Users/emcmaho7/Dropbox/projects/SI_fmri/fmri/figures')
    args = parser.parse_args()
    roi_reliability(args).run()

if __name__ == '__main__':
    main()

