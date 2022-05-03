#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path


class VoxelGroupResults():
    def __init__(self, args):
        self.cross_validation = args.cross_validation
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.n_subjs = 4
        Path(f'{self.out_dir}/VoxelPermutation').mkdir(parents=True, exist_ok=True)
        instance_variables = vars(self)
        print(instance_variables)

    def load(self):
        betas = None
        for i in range(self.n_subjs):
            sid = str(i+1).zfill(2)
            arr = np.load(f'{self.out_dir}/VoxelRegression/sub-{sid}_betas_method-{self.method}.npy')
            if betas is None:
                betas = arr.copy()
            else:
                betas += arr
        betas /= self.n_subjs
        return betas

    def save(self, betas):
        np.save(f'{self.out_dir}/VoxelRegression/sub-all_betas_method-{self.method}.npy', betas)

    def run(self):
        betas = self.load()
        self.save(betas)
        print("Completed successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    args = parser.parse_args()
    VoxelGroupResults(args).run()

if __name__ == '__main__':
    main()